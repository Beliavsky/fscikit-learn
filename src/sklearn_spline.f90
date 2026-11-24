! SplineTransformer contracts: sklearn 1.6.1 preprocessing/_polynomial.py.
! BSD-3-Clause attribution retained in LICENSES/scikit-learn-COPYING.
! Native Cox-de Boor evaluation and derivative recurrence replace scipy BSpline.
module sklearn_spline
  use sklearn_common, only: dp,require,check_x,argsort
  use sklearn_sparse,only:csr_matrix
  use sklearn_feature_extraction,only:feature_string
  use sklearn_ieee_math,only:sub=>ieee_difference,mul=>ieee_product,div=>ieee_quotient
  use,intrinsic::ieee_arithmetic
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  private
  public :: spline_transformer
  type :: spline_transformer
    integer :: n_knots=5,degree=3,n_features_in_=0,n_features_out_=0
    character(16) :: knots='uniform',extrapolation='constant'
    logical :: include_bias=.true.
    type(feature_string),allocatable::feature_names_in_(:)
    real(dp), allocatable :: base_knots_(:,:),knots_(:,:)
    integer, private :: fitted_degree=3,n_splines=0
    character(16), private :: fitted_extrapolation='constant'
  contains
    procedure :: fit=>spline_fit
    procedure :: transform=>spline_transform
    procedure :: fit_transform=>spline_fit_transform
    procedure::transform_csr=>spline_csr
    procedure::fit_transform_csr=>spline_fit_csr
    procedure::get_feature_names_out=>spline_names
  end type
contains
  subroutine spline_fit(self,x,knots,sample_weight,feature_names)
    class(spline_transformer), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(in), optional :: knots(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    type(feature_string),optional,intent(in)::feature_names(:)
    real(dp), allocatable :: base(:,:),sorted(:),cdf(:)
    integer, allocatable :: order(:)
    real(dp) :: q,fraction,period,first_step,last_step
    integer(int64) :: raw,width,total
    integer :: p,m,j,k,lo,hi,mid,i,n
    real(dp)::target,delta,step
    call check_x(x)
    call require(size(x,1)>=2,'SplineTransformer needs at least two samples')
    call require(self%n_knots>=2,'SplineTransformer n_knots must be at least two')
    call require(self%degree>=0,'SplineTransformer degree must be nonnegative')
    call require(self%extrapolation=='constant'.or.self%extrapolation=='linear'.or. &
      self%extrapolation=='continue'.or.self%extrapolation=='periodic'.or.self%extrapolation=='error', &
      'unknown SplineTransformer extrapolation')
    p=size(x,2); m=self%n_knots;n=size(x,1)
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'SplineTransformer weight count mismatch')
      call require(all(ieee_is_finite(sample_weight)),'SplineTransformer weights must be finite')
    end if
    if(allocated(self%feature_names_in_)) deallocate(self%feature_names_in_)
    if(present(feature_names)) then
      call require(size(feature_names)==p,'SplineTransformer feature name count mismatch')
      do i=1,p
        call require(allocated(feature_names(i)%text),'SplineTransformer feature name is unallocated')
      end do
      self%feature_names_in_=feature_names
    end if
    if(present(knots)) m=size(knots,1)
    total=int(m,int64)+2_int64*self%degree
    raw=int(m,int64)+self%degree-1
    if(self%extrapolation=='periodic') raw=int(m,int64)-1
    width=int(p,int64)*(raw-merge(0,1,self%include_bias))
    call require(total<=huge(0).and.width<=huge(0),'SplineTransformer output dimension overflow')
    if(present(knots)) then
      call check_x(knots)
      call require(size(knots,1)>=2,'SplineTransformer explicit knots need at least two rows')
      call require(size(knots,2)==p,'SplineTransformer knot feature count mismatch')
      base=knots
    else
      call require(self%knots=='uniform'.or.self%knots=='quantile','unknown SplineTransformer knots mode')
      allocate(base(self%n_knots,p))
      do j=1,p
        order=argsort(x(:,j)); sorted=x(order,j)
        if(self%knots=='uniform'.and.present(sample_weight)) then
          call require(any(sample_weight>0),'SplineTransformer uniform knots need a positive-weight sample')
          sorted=pack(x(:,j),sample_weight>0);sorted=sorted(argsort(sorted))
        end if
        if(self%knots=='quantile'.and.present(sample_weight)) then
          cdf=sample_weight(order)
          do i=2,n
            cdf(i)=sub(cdf(i-1),-cdf(i))
          end do
        end if
        delta=sub(sorted(size(sorted)),sorted(1));step=div(delta,real(m-1,dp))
        do k=1,m
          fraction=real(k-1,dp)/real(m-1,dp)
          if(self%knots=='uniform') then
            base(k,j)=sub(sorted(1),-mul(real(k-1,dp),step))
          else if(present(sample_weight)) then
            target=mul(div(mul(100._dp,fraction),100._dp),cdf(n))
            if(target==0) target=nearest(0._dp,1._dp)
            lo=1;hi=n+1
            do while(lo<hi)
              mid=lo+(hi-lo)/2
              if(cdf(mid)<target) then
                lo=mid+1
              else
                hi=mid
              end if
            end do
            base(k,j)=sorted(min(lo,n))
          else
            ! NumPy percentile computes the virtual index after percent/100.
            q=div(mul(100._dp,fraction),100._dp)*real(n-1,dp)
            lo=floor(q)+1;hi=min(lo+1,n);q=q-floor(q)
            delta=sub(sorted(hi),sorted(lo))
            if(q>=.5_dp) then
              base(k,j)=sub(sorted(hi),mul(1-q,delta))
            else
              base(k,j)=sub(sorted(lo),-mul(q,delta))
            end if
          end if
        end do
        if(self%knots=='uniform') then
          base(1,j)=sorted(1);base(m,j)=sorted(size(sorted))
        end if
      end do
    end if
    m=size(base,1)
    if(present(knots)) call require(all(base(2:,:)>base(:m-1,:)), &
      'SplineTransformer knots must be strictly increasing')
    if(self%extrapolation=='periodic') &
      call require(m>self%degree,'periodic SplineTransformer requires degree < knot count')
    total=int(m,int64)+2_int64*self%degree
    raw=int(m,int64)+self%degree-1
    if(self%extrapolation=='periodic') raw=m-1
    width=int(p,int64)*(raw-merge(0,1,self%include_bias))
    call require(total<=huge(0).and.width<=huge(0),'SplineTransformer output dimension overflow')
    self%base_knots_=base
    if(allocated(self%knots_)) deallocate(self%knots_)
    allocate(self%knots_(int(total),p)); self%knots_(self%degree+1:self%degree+m,:)=base
    do j=1,p
      period=sub(base(m,j),base(1,j)); first_step=sub(base(2,j),base(1,j)); last_step=sub(base(m,j),base(m-1,j))
      do k=1,self%degree
        if(self%extrapolation=='periodic') then
          self%knots_(k,j)=sub(base(m-self%degree+k-1,j),period)
          self%knots_(self%degree+m+k,j)=sub(base(k+1,j),-period)
        else
          self%knots_(k,j)=sub(base(1,j),mul(real(self%degree,dp),first_step))
          self%knots_(self%degree+m+k,j)=sub(base(m,j),-mul(real(self%degree,dp),last_step))
          if(self%degree>1) then
            delta=sub(sub(base(1,j),first_step),self%knots_(k,j))
            self%knots_(k,j)=sub(self%knots_(k,j),-mul(real(k-1,dp),div(delta,real(self%degree-1,dp))))
            delta=sub(self%knots_(self%degree+m+k,j),sub(base(m,j),-last_step))
            self%knots_(self%degree+m+k,j)=sub(sub(base(m,j),-last_step), &
              -mul(real(k-1,dp),div(delta,real(self%degree-1,dp))))
          else
            self%knots_(self%degree+m+k,j)=sub(base(m,j),-last_step)
          end if
          if(k==self%degree) then
            self%knots_(k,j)=sub(base(1,j),first_step)
            self%knots_(self%degree+m+k,j)=sub(base(m,j),-mul(real(self%degree,dp),last_step))
          end if
        end if
      end do
    end do
    self%n_features_in_=p; self%n_features_out_=int(width); self%n_splines=int(raw)
    self%fitted_degree=self%degree; self%fitted_extrapolation=self%extrapolation
  end subroutine

  subroutine spline_basis(t,degree,x,basis,derivative,stored,previous)
    real(dp),intent(in)::t(:),x
    integer,intent(in)::degree
    real(dp),allocatable,intent(out)::basis(:),derivative(:)
    logical,allocatable,intent(out)::stored(:)
    integer,optional,intent(inout)::previous
    real(dp)::h(degree+1),hh(degree+1),a,b,w
    integer::n,span,ell,j,k,order,ind
    n=size(t)-degree-1;ell=degree
    allocate(basis(n),derivative(n),stored(n));basis=0;derivative=0;stored=.false.
    if(ieee_is_nan(x)) then
      basis=ieee_value(0._dp,ieee_quiet_nan);stored=.true.
      if(present(previous)) previous=0
      return
    end if
    if(present(previous)) then
      if(previous>degree+1.and.previous<=n) ell=previous-1
    end if
    do while(ell/=degree)
      if(x>=t(ell+1)) exit
      ell=ell-1
    end do
    ell=ell+1
    do while(ell/=n)
      if(x<t(ell+1)) exit
      ell=ell+1
    end do
    span=ell;if(present(previous)) previous=span
    stored(span-degree:span)=.true.
    ! SciPy 1.15.3 fitpack _deBoor_D, with owned compact work vectors.
    do order=0,1
      h=0;h(1)=1
      do j=1,degree-order
        hh=h;h(1)=0
        do k=1,j
          ind=span+k;a=t(ind-j);b=t(ind)
          if(b==a) then
            h(k+1)=0;cycle
          end if
          w=div(hh(k),sub(b,a))
          h(k)=sub(h(k),-mul(w,sub(b,x)));h(k+1)=mul(w,sub(x,a))
        end do
      end do
      if(order==1) then
        j=degree;hh=h;h(1)=0
        do k=1,j
          ind=span+k;a=t(ind-j);b=t(ind)
          if(b==a) then
            h(2)=0;cycle
          end if
          w=div(mul(real(j,dp),hh(k)),sub(b,a));h(k)=sub(h(k),w);h(k+1)=w
        end do
      end if
      if(order==0) then
        basis(span-degree:span)=h
      else
        derivative(span-degree:span)=h
      end if
    end do
  end subroutine
  logical function nonzero(x)
    real(dp),intent(in)::x
    nonzero=.true.;if(ieee_is_nan(x)) return
    nonzero=x/=0
  end function
  subroutine block(self,value,j,degree,sparse,basis,stored,previous)
    class(spline_transformer),intent(in)::self
    real(dp),intent(in)::value
    integer,intent(in)::j,degree
    logical,intent(in)::sparse
    real(dp),allocatable,intent(out)::basis(:)
    logical,allocatable,intent(out)::stored(:)
    integer,optional,intent(inout)::previous
    real(dp),allocatable::raw(:),derivative(:),edge(:),slope(:)
    logical,allocatable::slots(:),unused(:)
    real(dp)::left,right,point,period
    integer::k,column,d,n,fitdegree,edge_column,side,used
    logical::outside,allow
    fitdegree=self%fitted_degree;n=size(self%knots_,1)-fitdegree-1
    call require(degree>=0.and.degree<size(self%knots_,1),'SplineTransformer live degree index out of range')
    d=degree;point=value;outside=.false.
    allow=self%fitted_extrapolation=='continue'.or.self%fitted_extrapolation=='periodic'
    left=self%knots_(d+1,j);right=self%knots_(size(self%knots_,1)-d,j)
    if(self%extrapolation=='periodic') then
      left=self%knots_(fitdegree+1,j);right=self%knots_(n+1,j);period=sub(right,left)
      if(period==0) then
        point=ieee_value(0._dp,ieee_quiet_nan)
      else
        point=sub(left,-modulo(sub(value,left),period))
      end if
    else if(self%extrapolation/='continue'.and.self%extrapolation/='error') then
      outside=value<left.or.value>right
      if(outside) point=left
    end if
    if(sparse) call require(ieee_is_finite(point),'SplineTransformer sparse evaluation point must be finite')
    if(sparse.and..not.allow) call require(point>=self%knots_(fitdegree+1,j).and. &
      point<=self%knots_(n+1,j),'SplineTransformer sparse input outside knot range')
    if(outside.and..not.sparse) then
      call spline_basis(self%knots_(:,j),fitdegree,point,raw,derivative,slots)
    else
      call spline_basis(self%knots_(:,j),fitdegree,point,raw,derivative,slots,previous)
    end if
    if(.not.allow.and..not.sparse) then
      if(point<self%knots_(fitdegree+1,j).or.point>self%knots_(n+1,j)) then
        raw=ieee_value(0._dp,ieee_quiet_nan)
        if(present(previous)) previous=0
      end if
    end if
    if(self%extrapolation=='error') &
      call require(.not.any(ieee_is_nan(raw)),'SplineTransformer input outside knot range')
    if(sparse) then
      basis=raw;stored=slots
      if(self%extrapolation=='periodic') then
        call require(d/=0,'SplineTransformer degree-zero sparse periodic shape mismatch')
        ! sklearn LIL addition drops zeros in the first degree columns.
        d=min(d,n)
        do k=1,d
          basis(k)=sub(raw(k),-raw(n-d+k));stored(k)=nonzero(basis(k))
        end do
        if(d==0) then
          basis=basis(:0);stored=stored(:0)
        else
          basis=basis(:n-d);stored=stored(:n-d)
        end if
      end if
    else
      allocate(basis(self%n_splines),stored(self%n_splines));basis=0;stored=.true.
      do k=1,n
        column=k
        if(self%fitted_extrapolation=='periodic'.and.column>self%n_splines) column=column-self%n_splines
        call require(column<=size(basis),'SplineTransformer fitted coefficient shape mismatch')
        basis(column)=sub(basis(column),-raw(k))
      end do
    end if
    if(outside) then
      basis=0;stored=.false.
      if(self%extrapolation=='constant'.or.self%extrapolation=='linear') then
        used=d
        if(self%extrapolation=='linear'.and.used<=1) used=used+1
        do side=1,2
          if(side==1.and..not.(value<left)) cycle
          if(side==2.and..not.(value>right)) cycle
          point=left;if(side==2) point=right
          call spline_basis(self%knots_(:,j),fitdegree,point,edge,slope,unused)
          if(.not.allow) then
            if(point<self%knots_(fitdegree+1,j).or.point>self%knots_(n+1,j)) then
              edge=ieee_value(0._dp,ieee_quiet_nan);slope=edge
            end if
          end if
          if(self%fitted_extrapolation=='periodic') then
            do k=self%n_splines+1,size(edge)
              edge(k-self%n_splines)=sub(edge(k-self%n_splines),-edge(k))
              slope(k-self%n_splines)=sub(slope(k-self%n_splines),-slope(k))
            end do
          end if
          d=used
          if(side==2.and.d==0) then
            call require(sparse,'SplineTransformer degree-zero constant right extrapolation shape mismatch')
            d=size(basis)
          end if
          call require(d<=self%n_splines,'SplineTransformer live boundary coefficient shape mismatch')
          do k=1,d
            column=k;edge_column=k
            if(side==2) then
              column=size(basis)-d+k;edge_column=self%n_splines-d+k
              if(self%extrapolation=='linear') column=edge_column
            end if
            basis(column)=edge(edge_column)
            if(self%extrapolation=='linear') &
              basis(column)=sub(basis(column),-mul(sub(value,point),slope(edge_column)))
            stored(column)=nonzero(basis(column))
          end do
        end do
      end if
    end if
  end subroutine
  subroutine query_check(self,x)
    class(spline_transformer),intent(in)::self
    real(dp),intent(in)::x(:,:)
    call require(allocated(self%knots_),'SplineTransformer is not fitted')
    call check_x(x)
    call require(size(x,2)==self%n_features_in_,'SplineTransformer feature count mismatch')
  end subroutine
  integer function live_degree(self,j) result(d)
    class(spline_transformer),intent(in)::self
    integer,intent(in)::j
    d=self%degree
    if(self%extrapolation=='linear'.and.d<=1) d=min(2,d+j-1)
  end function
  function spline_transform(self,x) result(z)
    class(spline_transformer),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),allocatable::z(:,:),basis(:),full(:,:)
    logical,allocatable::stored(:)
    integer,allocatable::columns(:)
    integer::i,j,nout,previous
    call query_check(self,x)
    nout=self%n_features_out_+size(x,2)*merge(0,1,self%include_bias)
    call require(nout>=size(x,2)*self%n_splines,'SplineTransformer live bias output shape mismatch')
    allocate(full(size(x,1),nout));full=0
    do j=1,size(x,2)
      previous=self%fitted_degree+1
      do i=1,size(x,1)
        call block(self,x(i,j),j,live_degree(self,j),.false.,basis,stored,previous)
        full(i,(j-1)*self%n_splines+1:j*self%n_splines)=basis
      end do
    end do
    if(self%include_bias) then
      call move_alloc(full,z)
    else
      columns=pack([(i,i=1,nout)],mod([(i,i=1,nout)],self%n_splines)/=0);z=full(:,columns)
    end if
  end function
  function spline_csr(self,x) result(z)
    class(spline_transformer),intent(in)::self
    real(dp),intent(in)::x(:,:)
    type(csr_matrix)::z
    real(dp),allocatable::basis(:),values(:)
    logical,allocatable::stored(:)
    integer,allocatable::indices(:),widths(:),previous(:)
    integer(int64)::capacity
    integer::i,j,k,col,pos,total,nraw,d
    call query_check(self,x)
    call require(all(ieee_is_finite(self%knots_)),'SplineTransformer sparse knots must be finite')
    call require(all(self%knots_(2:,:)>=self%knots_(:size(self%knots_,1)-1,:)), &
      'SplineTransformer sparse knots must be sorted')
    nraw=size(self%knots_,1)-self%fitted_degree-1
    allocate(widths(size(x,2)),previous(size(x,2)));widths=nraw;previous=self%fitted_degree+1
    if(self%extrapolation=='periodic') then
      do j=1,size(widths)
        d=live_degree(self,j);widths(j)=max(0,nraw-d);if(d==0) widths(j)=0
      end do
    end if
    total=sum(widths);z%n_rows=size(x,1);z%n_cols=total
    if(.not.self%include_bias) z%n_cols=total-total/self%n_splines
    capacity=int(size(x,1),int64)*size(x,2)*min(nraw,max(self%fitted_degree+1,self%degree+2))
    call require(capacity<int(huge(1),int64),'SplineTransformer sparse stored count overflow')
    allocate(values(int(capacity)),indices(int(capacity)),z%indptr(size(x,1)+1));pos=1;z%indptr(1)=1
    do i=1,size(x,1)
      total=0
      do j=1,size(x,2)
        call block(self,x(i,j),j,live_degree(self,j),.true.,basis,stored,previous(j))
        do k=1,size(basis)
          if(.not.stored(k)) cycle
          col=total+k
          if(.not.self%include_bias) then
            if(mod(col,self%n_splines)==0) cycle
            col=col-(col-1)/self%n_splines
          end if
          call require(pos<=size(values),'SplineTransformer sparse capacity exceeded')
          values(pos)=basis(k);indices(pos)=col;pos=pos+1
        end do
        total=total+widths(j)
      end do
      z%indptr(i+1)=pos
    end do
    z%data=values(:pos-1);z%indices=indices(:pos-1);call z%validate()
  end function
  function spline_names(self,input_features) result(names)
    class(spline_transformer),intent(in)::self
    type(feature_string),optional,intent(in)::input_features(:)
    type(feature_string),allocatable::names(:),input(:)
    integer::i,j,k,width
    character(32)::number
    call require(allocated(self%knots_),'SplineTransformer is not fitted')
    if(present(input_features)) then
      call require(size(input_features)==self%n_features_in_,'SplineTransformer input name count mismatch')
      do i=1,size(input_features)
        call require(allocated(input_features(i)%text),'SplineTransformer feature name is unallocated')
        if(allocated(self%feature_names_in_)) then
          call require(len(input_features(i)%text)==len(self%feature_names_in_(i)%text), &
            'SplineTransformer input feature names differ')
          call require(input_features(i)%text==self%feature_names_in_(i)%text,'SplineTransformer input feature names differ')
        end if
      end do
      input=input_features
    else if(allocated(self%feature_names_in_)) then
      input=self%feature_names_in_
    else
      allocate(input(self%n_features_in_))
      do i=1,size(input)
        write(number,'(i0)') i-1;input(i)%text='x'//trim(number)
      end do
    end if
    width=self%n_splines-merge(0,1,self%include_bias);allocate(names(width*size(input)));k=0
    do i=1,size(input)
      do j=0,width-1
        k=k+1;write(number,'(i0)') j;names(k)%text=input(i)%text//'_sp_'//trim(number)
      end do
    end do
  end function
  function spline_fit_transform(self,x,knots,sample_weight,feature_names) result(z)
    class(spline_transformer),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    real(dp),optional,intent(in)::knots(:,:),sample_weight(:)
    type(feature_string),optional,intent(in)::feature_names(:)
    real(dp),allocatable::z(:,:)
    call self%fit(x,knots,sample_weight,feature_names);z=self%transform(x)
  end function
  function spline_fit_csr(self,x,knots,sample_weight,feature_names) result(z)
    class(spline_transformer),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    real(dp),optional,intent(in)::knots(:,:),sample_weight(:)
    type(feature_string),optional,intent(in)::feature_names(:)
    type(csr_matrix)::z
    call self%fit(x,knots,sample_weight,feature_names);z=self%transform_csr(x)
  end function
end module
