! Dense kernel feature maps following sklearn/kernel_approximation.py.
! See THIRD_PARTY_NOTICES.md for upstream BSD-3-Clause attribution.
module sklearn_kernel_approximation
  use sklearn_nystroem_svd,only:kernel_svd
  use sklearn_polynomial_count_sketch, only: polynomial_count_sketch
  use sklearn_sparse, only: csr_matrix
  use sklearn_stateless_preprocessing,only:sorted_sparse
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_ieee_math, only: ieee_difference,ieee_product,ieee_quotient
  use sklearn_common
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value,ieee_quiet_nan,ieee_fma, &
    ieee_negative_inf,ieee_positive_inf
  implicit none
  private
  public :: polynomial_count_sketch
  public :: rbf_sampler,nystroem,additive_chi2_sampler,skewed_chi2_sampler
  ! Chi2 map contracts follow sklearn 1.6.1 kernel_approximation.py.
  type :: additive_chi2_sampler
    integer :: sample_steps=2,n_features_in_=0
    real(dp) :: sample_interval=-1
  contains
    procedure,private::fit_dense=>additive_fit
    procedure,private::fit_csr=>additive_fit_csr
    generic::fit=>fit_dense,fit_csr
    procedure,private::transform_dense=>additive_transform
    procedure,private::transform_csr=>additive_transform_csr
    generic::transform=>transform_dense,transform_csr
    procedure,private::fit_transform_dense=>additive_fit_transform
    procedure,private::fit_transform_csr=>additive_fit_transform_csr
    generic::fit_transform=>fit_transform_dense,fit_transform_csr
  end type
  type :: skewed_chi2_sampler
    integer :: n_components=100,random_state=0,n_features_in_=0
    real(dp) :: skewedness=1
    real(dp), allocatable :: random_weights_(:,:),random_offset_(:)
  contains
    procedure :: fit=>skewed_fit
    procedure :: transform=>skewed_transform
    procedure :: fit_transform=>skewed_fit_transform
  end type
  type :: rbf_sampler
    integer :: n_components=100,random_state=0,n_features_in_=0
    real(dp) :: gamma=1,gamma_=1
    logical :: gamma_scale=.false.
    real(dp), allocatable :: random_weights_(:,:),random_offset_(:)
  contains
    procedure, private :: fit_dense=>rbfs_fit
    procedure, private :: fit_csr=>rbfs_fit_csr
    generic :: fit=>fit_dense,fit_csr
    procedure, private :: transform_dense=>rbfs_transform
    procedure, private :: transform_csr=>rbfs_transform_csr
    generic :: transform=>transform_dense,transform_csr
    procedure, private :: fit_transform_dense=>rbfs_fit_transform
    procedure, private :: fit_transform_csr=>rbfs_fit_transform_csr
    generic :: fit_transform=>fit_transform_dense,fit_transform_csr
  end type
  abstract interface
    function nystroem_kernel(a,b) result(value)
      import dp
      real(dp),intent(in)::a(:),b(:)
      real(dp)::value
    end function
  end interface
  public::nystroem_kernel
  type :: nystroem
    integer :: n_components=100,random_state=0,degree=3,n_features_in_=0
    character(16) :: kernel='rbf'
    real(dp) :: gamma=-1,coef0=1,gamma_=0
    real(dp), allocatable :: components_(:,:),normalization_(:,:)
    integer, allocatable :: component_indices_(:)
    type(csr_matrix),allocatable::components_csr_
    procedure(nystroem_kernel),pointer,nopass::kernel_callback=>null()
  contains
    procedure,private::fit_dense=>ny_fit
    procedure,private::fit_csr=>ny_fit_csr
    generic::fit=>fit_dense,fit_csr
    procedure,private::transform_dense=>ny_transform
    procedure,private::transform_csr=>ny_transform_csr
    generic::transform=>transform_dense,transform_csr
    procedure,private::fit_transform_dense=>ny_fit_transform
    procedure,private::fit_transform_csr=>ny_fit_transform_csr
    generic::fit_transform=>fit_transform_dense,fit_transform_csr
  end type
contains
  recursive real(dp) function rbf_numpy_sum(a) result(total)
    real(dp), intent(in) :: a(:)
    real(dp) :: partial(8),left,right
    integer :: n,i,j,whole,split
    n=size(a);total=-0._dp
    if(n<8) then
      do i=1,n
        total=ieee_difference(total,-a(i))
      end do
    else if(n<=128) then
      partial=a(:8);whole=n-mod(n,8)
      do i=9,whole,8
        do j=1,8
          partial(j)=ieee_difference(partial(j),-a(i+j-1))
        end do
      end do
      left=ieee_difference(ieee_difference(partial(1),-partial(2)),-ieee_difference(partial(3),-partial(4)))
      right=ieee_difference(ieee_difference(partial(5),-partial(6)),-ieee_difference(partial(7),-partial(8)))
      total=ieee_difference(left,-right)
      do i=whole+1,n
        total=ieee_difference(total,-a(i))
      end do
    else
      split=n/2;split=split-mod(split,8)
      total=ieee_difference(rbf_numpy_sum(a(:split)),-rbf_numpy_sum(a(split+1:)))
    end if
  end function
  subroutine rbf_controls(self)
    class(rbf_sampler),intent(in)::self
    call require(self%n_components>0,'RBF sampler n_components must be positive')
    if(.not.self%gamma_scale) then
      call require(ieee_is_finite(self%gamma),'RBF sampler gamma must be finite')
      call require(self%gamma>=0,'invalid RBF sampler gamma')
    end if
    call require(self%random_state>=0,'RBF sampler random_state must be nonnegative')
  end subroutine
  subroutine rbf_csr_input(x)
    type(csr_matrix),intent(in)::x
    call x%validate()
    call require(x%n_rows>0.and.x%n_cols>0,'RBF sampler CSR input must be nonempty')
    call require(all(ieee_is_finite(x%data)),'RBF sampler CSR input contains NaN or infinity')
  end subroutine
  subroutine rbf_generate(self,p,variance)
    class(rbf_sampler),intent(inout)::self
    integer,intent(in)::p
    real(dp),intent(in)::variance
    type(legacy_random_state)::rng
    real(dp)::scale,square
    integer::i,j
    self%n_features_in_=p;self%gamma_=self%gamma
    if(self%gamma_scale) then
      self%gamma_=1
      if(ieee_is_nan(variance)) then
        self%gamma_=variance
      else if(variance/=0) then
        self%gamma_=ieee_quotient(1._dp,ieee_product(real(p,dp),variance))
      end if
    end if
    square=ieee_product(2._dp,self%gamma_);scale=ieee_value(0._dp,ieee_quiet_nan)
    if(.not.ieee_is_nan(square)) then
      if(square>=0) scale=sqrt(square)
    end if
    if(allocated(self%random_weights_)) deallocate(self%random_weights_)
    if(allocated(self%random_offset_)) deallocate(self%random_offset_)
    allocate(self%random_weights_(p,self%n_components),self%random_offset_(self%n_components))
    call rng%seed(self%random_state)
    do i=1,p
      do j=1,self%n_components
        self%random_weights_(i,j)=ieee_product(scale,rng%normal())
      end do
    end do
    do j=1,self%n_components
      self%random_offset_(j)=rng%uniform()*(2*acos(-1._dp))
    end do
  end subroutine
  subroutine rbfs_fit(self,x)
    class(rbf_sampler),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    real(dp),allocatable::values(:)
    real(dp)::variance,mean,total
    integer::i
    call check_x(x);call rbf_controls(self);variance=0
    if(self%gamma_scale) then
      values=reshape(transpose(x),[size(x)]);total=real(size(values),dp)
      mean=ieee_quotient(rbf_numpy_sum(values),total)
      do i=1,size(values)
        values(i)=ieee_difference(values(i),mean)
        values(i)=ieee_product(values(i),values(i))
      end do
      variance=ieee_quotient(rbf_numpy_sum(values),total)
    end if
    call rbf_generate(self,size(x,2),variance)
  end subroutine
  subroutine rbfs_fit_csr(self,x)
    class(rbf_sampler),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::row_values(:),means(:),squares(:)
    integer,allocatable::seen(:),columns(:)
    real(dp)::factor,mean,variance,value
    integer::i,k,j,n,previous,index_,first,last,step
    logical::canonical
    call rbf_csr_input(x);call rbf_controls(self);variance=0
    if(self%gamma_scale) then
      factor=1/(real(x%n_rows,dp)*real(x%n_cols,dp))
      allocate(row_values(x%n_cols),seen(x%n_cols),columns(x%n_cols),means(x%n_rows),squares(x%n_rows))
      seen=0;canonical=.true.
      do i=1,x%n_rows
        previous=0
        do k=x%indptr(i),x%indptr(i+1)-1
          if(x%indices(k)<=previous) canonical=.false.
          previous=x%indices(k)
        end do
      end do
      do i=1,x%n_rows
        n=0;means(i)=0;squares(i)=0
        do k=x%indptr(i),x%indptr(i+1)-1
          j=x%indices(k)
          if(seen(j)/=i) then
            n=n+1;columns(n)=j;seen(j)=i;row_values(j)=0
          end if
          row_values(j)=ieee_difference(row_values(j),-x%data(k))
          means(i)=ieee_difference(means(i),-ieee_product(x%data(k),factor))
        end do
        first=1;last=n;step=1
        if(.not.canonical) then
          first=n;last=1;step=-1
        end if
        do index_=first,last,step
          value=row_values(columns(index_));value=ieee_product(value,value)
          squares(i)=ieee_difference(squares(i),-ieee_product(value,factor))
        end do
      end do
      mean=rbf_numpy_sum(means)
      variance=ieee_difference(rbf_numpy_sum(squares),ieee_product(mean,mean))
    end if
    call rbf_generate(self,x%n_cols,variance)
  end subroutine
  subroutine rbf_finish(self,z)
    class(rbf_sampler),intent(in)::self
    real(dp),intent(inout)::z(:,:)
    real(dp)::phase,scale
    integer::i,j
    call require(self%n_components>0,'RBF sampler transform n_components must be positive')
    scale=sqrt(2._dp/self%n_components)
    do j=1,size(z,2)
      do i=1,size(z,1)
        phase=ieee_difference(z(i,j),-self%random_offset_(j))
        z(i,j)=ieee_value(0._dp,ieee_quiet_nan)
        if(ieee_is_finite(phase)) z(i,j)=cos(phase)*scale
      end do
    end do
  end subroutine
  function fourier_dot(x,weights) result(z)
    real(dp),intent(in)::x(:,:),weights(:,:)
    real(dp),allocatable::z(:,:)
    real(dp)::bound
    integer::i,j,k
    logical::fast,fused
    fast=.false.;fused=.false.
    if(all(ieee_is_finite(weights)).and.all(ieee_is_finite(x))) then
      bound=ieee_product(ieee_product(maxval(abs(x)),maxval(abs(weights))),real(size(x,2),dp))
      ! Large phases amplify last-bit reassociation into different cosine values.
      ! Keep ordinary maps on MATMUL and control summation for sensitive phases.
      fast=.false.
      if(ieee_is_finite(bound)) then
        fast=bound<=1024._dp;fused=.not.fast
      end if
    end if
    if(fast) then
      z=matmul(x,weights)
    else
      allocate(z(size(x,1),size(weights,2)));z=0
      do j=1,size(z,2)
        do i=1,size(z,1)
          do k=1,size(x,2)
            if(fused) then
              z(i,j)=ieee_fma(x(i,k),weights(k,j),z(i,j))
            else
              z(i,j)=ieee_difference(z(i,j),-ieee_product(x(i,k),weights(k,j)))
            end if
          end do
        end do
      end do
    end if
  end function
  function rbfs_transform(self,x) result(z)
    class(rbf_sampler),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),allocatable::z(:,:)
    call require(allocated(self%random_weights_),'RBF sampler is not fitted')
    call check_xy(transpose(x),size(self%random_weights_,1))
    z=fourier_dot(x,self%random_weights_)
    call rbf_finish(self,z)
  end function
  function rbfs_transform_csr(self,x) result(z)
    class(rbf_sampler),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::z(:,:)
    call require(allocated(self%random_weights_),'RBF sampler is not fitted')
    call rbf_csr_input(x)
    call require(x%n_cols==size(self%random_weights_,1),'RBF sampler CSR feature count mismatch')
    z=x%matmul(self%random_weights_);call rbf_finish(self,z)
  end function
  function rbfs_fit_transform(self,x) result(z)
    class(rbf_sampler),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    real(dp),allocatable::z(:,:)
    call self%fit(x);z=self%transform(x)
  end function
  function rbfs_fit_transform_csr(self,x) result(z)
    class(rbf_sampler),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::z(:,:)
    call self%fit(x);z=self%transform(x)
  end function
  subroutine ny_controls(self)
    class(nystroem),intent(in)::self
    call require(any(self%kernel==[character(16)::'rbf','linear','poly','polynomial','cosine', &
      'laplacian','sigmoid','chi2','additive_chi2','precomputed','callable']),'unsupported Nystroem kernel')
    call require(ieee_is_finite(self%gamma),'invalid Nystroem gamma')
    call require(self%gamma>=0.or.self%gamma==-1,'invalid Nystroem gamma')
    call require(self%degree>=1.and.ieee_is_finite(self%coef0),'invalid Nystroem polynomial parameters')
    if(self%kernel=='callable'.or.self%kernel=='precomputed') then
      call require(self%gamma==-1.and.self%degree==3.and.self%coef0==1, &
        'Nystroem callable/precomputed kernels require default gamma, degree and coef0')
    end if
    if(self%kernel=='callable') call require(associated(self%kernel_callback),'Nystroem kernel callback is not associated')
  end subroutine
  function ny_cosine_input(x) result(z)
    real(dp),intent(in)::x(:,:)
    real(dp),allocatable::z(:,:)
    real(dp)::norm
    integer::i,j
    allocate(z(size(x,1),size(x,2)))
    do i=1,size(x,1)
      norm=0
      do j=1,size(x,2)
        norm=ieee_difference(norm,-ieee_product(x(i,j),x(i,j)))
      end do
      norm=sqrt(norm)
      if(norm<10*epsilon(1._dp)) norm=1
      do j=1,size(x,2)
        z(i,j)=ieee_quotient(x(i,j),norm)
      end do
    end do
  end function
  function kernel_matrix(self,x,y,symmetric) result(k)
    class(nystroem),intent(in)::self
    real(dp),intent(in)::x(:,:),y(:,:)
    logical,intent(in),optional::symmetric
    real(dp),allocatable::k(:,:),nx(:),ny(:)
    real(dp)::g,value,base,denom,difference
    integer::i,j,t
    logical::sym
    call ny_controls(self)
    g=self%gamma
    if(g==-1) then
      g=1._dp/size(x,2)
      if(self%kernel=='chi2') g=1
    end if
    sym=.false.;if(present(symmetric)) sym=symmetric
    select case(self%kernel)
    case('rbf')
      k=fourier_dot(x,transpose(y));allocate(nx(size(x,1)),ny(size(y,1)));nx=0;ny=0
      do i=1,size(x,1)
        do t=1,size(x,2)
          nx(i)=ieee_difference(nx(i),-ieee_product(x(i,t),x(i,t)))
        end do
      end do
      do j=1,size(y,1)
        do t=1,size(y,2)
          ny(j)=ieee_difference(ny(j),-ieee_product(y(j,t),y(j,t)))
        end do
      end do
      do j=1,size(y,1)
        do i=1,size(x,1)
          value=ieee_product(-2._dp,k(i,j));value=ieee_difference(value,-nx(i))
          value=ieee_difference(value,-ny(j))
          if(.not.ieee_is_nan(value)) value=max(value,0._dp)
          if(sym.and.i==j) value=0
          value=ieee_product(-g,value);k(i,j)=value
          if(.not.ieee_is_nan(value)) k(i,j)=exp(value)
        end do
      end do
    case('linear');k=fourier_dot(x,transpose(y))
    case('poly','polynomial')
      k=fourier_dot(x,transpose(y))
      do j=1,size(y,1)
        do i=1,size(x,1)
          base=ieee_difference(ieee_product(g,k(i,j)),-self%coef0);value=1
          do t=1,self%degree
            value=ieee_product(value,base)
          end do
          k(i,j)=value
        end do
      end do
    case('cosine');k=fourier_dot(ny_cosine_input(x),transpose(ny_cosine_input(y)))
    case('laplacian')
      allocate(k(size(x,1),size(y,1)))
      do j=1,size(y,1)
        do i=1,size(x,1)
          value=0
          do t=1,size(x,2)
            value=ieee_difference(value,-abs(ieee_difference(x(i,t),y(j,t))))
          end do
          value=ieee_product(-g,value);k(i,j)=value
          if(.not.ieee_is_nan(value)) k(i,j)=exp(value)
        end do
      end do
    case('sigmoid')
      k=fourier_dot(x,transpose(y))
      do j=1,size(y,1)
        do i=1,size(x,1)
          value=ieee_difference(ieee_product(g,k(i,j)),-self%coef0);k(i,j)=value
          if(.not.ieee_is_nan(value)) k(i,j)=tanh(value)
        end do
      end do
    case('chi2','additive_chi2')
      call require(all(x>=0).and.all(y>=0),'Nystroem chi-squared kernels require nonnegative inputs')
      allocate(k(size(x,1),size(y,1)))
      do j=1,size(y,1)
        do i=1,size(x,1)
          value=0
          do t=1,size(x,2)
            denom=ieee_difference(x(i,t),-y(j,t))
            if(denom==0) cycle
            difference=ieee_difference(x(i,t),y(j,t))
            base=ieee_quotient(ieee_product(difference,difference),denom)
            value=ieee_difference(value,-base)
          end do
          k(i,j)=-value
          if(self%kernel=='chi2') then
            value=ieee_product(g,k(i,j));k(i,j)=value
            if(.not.ieee_is_nan(value)) k(i,j)=exp(value)
          end if
        end do
      end do
    case('precomputed')
      call require(size(x,2)==size(y,1),'Nystroem precomputed kernel shape mismatch')
      k=x
    case('callable')
      allocate(k(size(x,1),size(y,1)))
      do i=1,size(x,1)
        do j=1,size(y,1)
          if(sym.and.i>j) cycle
          k(i,j)=self%kernel_callback(x(i,:),y(j,:))
          if(sym) k(j,i)=k(i,j)
        end do
      end do
    end select
  end function
  subroutine ny_csr_input(x)
    type(csr_matrix),intent(in)::x
    call x%validate()
    call require(x%n_rows>0.and.x%n_cols>0,'Nystroem CSR input must be nonempty')
    call require(all(ieee_is_finite(x%data)),'Nystroem CSR input contains NaN or infinity')
  end subroutine
  function ny_select_csr(x,rows) result(basis)
    use,intrinsic::iso_fortran_env,only:int64
    type(csr_matrix),intent(in)::x
    integer,intent(in)::rows(:)
    type(csr_matrix)::basis
    integer(int64)::total
    integer::i,k,last,first,count
    total=0
    do i=1,size(rows)
      total=total+x%indptr(rows(i)+1)-x%indptr(rows(i))
    end do
    call require(total<huge(1),'Nystroem CSR landmark stored count overflow')
    basis%n_rows=size(rows);basis%n_cols=x%n_cols
    allocate(basis%indptr(size(rows)+1),basis%indices(int(total)),basis%data(int(total)))
    basis%indptr(1)=1;k=1
    do i=1,size(rows)
      first=x%indptr(rows(i));last=x%indptr(rows(i)+1)-1;count=last-first+1
      basis%indices(k:k+count-1)=x%indices(first:last);basis%data(k:k+count-1)=x%data(first:last)
      k=k+count;basis%indptr(i+1)=k
    end do
  end function
  function ny_sparse_row(a,row) result(values)
    type(csr_matrix),intent(in)::a
    integer,intent(in)::row
    real(dp),allocatable::values(:)
    integer::k,j
    allocate(values(a%n_cols));values=0
    do k=a%indptr(row),a%indptr(row+1)-1
      j=a%indices(k);values(j)=ieee_difference(values(j),-a%data(k))
    end do
  end function
  function ny_sparse_norms(a) result(norms)
    type(csr_matrix),intent(in)::a
    real(dp)::norms(a%n_rows)
    integer::i,k
    norms=0
    do i=1,a%n_rows
      do k=a%indptr(i),a%indptr(i+1)-1
        norms(i)=ieee_difference(norms(i),-ieee_product(a%data(k),a%data(k)))
      end do
    end do
  end function
  function ny_dense_norms(a) result(norms)
    real(dp),intent(in)::a(:,:)
    real(dp)::norms(size(a,1))
    integer::i,j
    norms=0
    do i=1,size(a,1)
      do j=1,size(a,2)
        norms(i)=ieee_difference(norms(i),-ieee_product(a(i,j),a(i,j)))
      end do
    end do
  end function
  function ny_sparse_kernel(self,sx,sy,dx,dy,symmetric) result(k)
    class(nystroem),intent(in)::self
    type(csr_matrix),intent(in),optional::sx,sy
    real(dp),intent(in),optional::dx(:,:),dy(:,:)
    logical,intent(in),optional::symmetric
    real(dp),allocatable::k(:,:),nx(:),ny(:),rowx(:),rowy(:)
    type(csr_matrix)::cx,cy
    real(dp)::g,value,a,b,base
    integer::n,m,p,i,j,ix,iy,col,lx,ly,rx,ry,t
    logical::sym,cosine
    call ny_controls(self)
    call require(self%kernel/='chi2'.and.self%kernel/='additive_chi2', &
      'Nystroem chi-squared kernels require dense inputs in sklearn')
    call require(self%kernel/='precomputed','Nystroem precomputed CSR output is not a source numeric array')
    if(present(sx)) then
      n=sx%n_rows;p=sx%n_cols
    else
      n=size(dx,1);p=size(dx,2)
    end if
    if(present(sy)) then
      m=sy%n_rows
    else
      m=size(dy,1)
    end if
    g=self%gamma;if(g==-1) g=1._dp/p
    sym=.false.;if(present(symmetric)) sym=symmetric
    allocate(k(n,m));cosine=self%kernel=='cosine'
    allocate(nx(n),ny(m));nx=1;ny=1
    if(self%kernel=='rbf'.or.cosine) then
      if(present(sx)) then
        nx=ny_sparse_norms(sx)
      else
        nx=ny_dense_norms(dx)
      end if
      if(present(sy)) then
        ny=ny_sparse_norms(sy)
      else
        ny=ny_dense_norms(dy)
      end if
      if(cosine) then
        nx=sqrt(nx);ny=sqrt(ny)
        if(present(sx)) then
          where(nx==0) nx=1
        else
          where(nx<10*epsilon(1._dp)) nx=1
        end if
        if(present(sy)) then
          where(ny==0) ny=1
        else
          where(ny<10*epsilon(1._dp)) ny=1
        end if
      end if
    end if
    if(self%kernel=='laplacian') then
      if(present(sx)) cx=sorted_sparse(sx,.true.)
      if(present(sy)) cy=sorted_sparse(sy,.true.)
    end if
    do i=1,n
      if(self%kernel=='callable') then
        if(present(sx)) then
          rowx=ny_sparse_row(sx,i)
        else
          rowx=dx(i,:)
        end if
      end if
      do j=1,m
        if(sym.and.self%kernel=='callable'.and.i>j) cycle
        if(self%kernel=='callable') then
          if(present(sy)) then
            rowy=ny_sparse_row(sy,j)
          else
            rowy=dy(j,:)
          end if
          k(i,j)=self%kernel_callback(rowx,rowy)
          if(sym) k(j,i)=k(i,j)
          cycle
        end if
        value=0
        if(self%kernel=='laplacian') then
          if(present(sx).and.present(sy)) then
            ix=cx%indptr(i);rx=cx%indptr(i+1);iy=cy%indptr(j);ry=cy%indptr(j+1)
            do while(ix<rx.or.iy<ry)
              if(ix==rx) then
                value=ieee_difference(value,-abs(cy%data(iy)));iy=iy+1
              else if(iy==ry) then
                value=ieee_difference(value,-abs(cx%data(ix)));ix=ix+1
              else if(cx%indices(ix)<cy%indices(iy)) then
                value=ieee_difference(value,-abs(cx%data(ix)));ix=ix+1
              else if(cx%indices(ix)>cy%indices(iy)) then
                value=ieee_difference(value,-abs(cy%data(iy)));iy=iy+1
              else
                value=ieee_difference(value,-abs(ieee_difference(cx%data(ix),cy%data(iy))))
                ix=ix+1;iy=iy+1
              end if
            end do
          else
            if(present(sx)) then
              rowx=ny_sparse_row(cx,i);rowy=dy(j,:)
            else
              rowx=dx(i,:);rowy=ny_sparse_row(cy,j)
            end if
            do col=1,p
              value=ieee_difference(value,-abs(ieee_difference(rowx(col),rowy(col))))
            end do
          end if
          value=ieee_product(-g,value);k(i,j)=value
          if(.not.ieee_is_nan(value)) k(i,j)=exp(value)
          cycle
        end if
        if(present(sx)) then
          do ix=sx%indptr(i),sx%indptr(i+1)-1
            col=sx%indices(ix);a=sx%data(ix)
            if(cosine) a=ieee_quotient(a,nx(i))
            if(present(sy)) then
              do iy=sy%indptr(j),sy%indptr(j+1)-1
                if(sy%indices(iy)/=col) cycle
                b=sy%data(iy);if(cosine) b=ieee_quotient(b,ny(j))
                value=ieee_difference(value,-ieee_product(a,b))
              end do
            else
              b=dy(j,col);if(cosine) b=ieee_quotient(b,ny(j))
              value=ieee_difference(value,-ieee_product(a,b))
            end if
          end do
        else
          do iy=sy%indptr(j),sy%indptr(j+1)-1
            col=sy%indices(iy);a=dx(i,col);b=sy%data(iy)
            if(cosine) then
              a=ieee_quotient(a,nx(i));b=ieee_quotient(b,ny(j))
            end if
            value=ieee_difference(value,-ieee_product(a,b))
          end do
        end if
        select case(self%kernel)
        case('linear','cosine');k(i,j)=value
        case('rbf')
          value=ieee_product(-2._dp,value);value=ieee_difference(value,-nx(i));value=ieee_difference(value,-ny(j))
          if(.not.ieee_is_nan(value)) value=max(value,0._dp)
          if(sym.and.i==j) value=0
          value=ieee_product(-g,value);k(i,j)=value
          if(.not.ieee_is_nan(value)) k(i,j)=exp(value)
        case('poly','polynomial')
          base=ieee_difference(ieee_product(g,value),-self%coef0);value=1
          do t=1,self%degree
            value=ieee_product(value,base)
          end do
          k(i,j)=value
        case('sigmoid')
          value=ieee_difference(ieee_product(g,value),-self%coef0);k(i,j)=value
          if(.not.ieee_is_nan(value)) k(i,j)=tanh(value)
        end select
      end do
    end do
  end function
  subroutine ny_normalization(self,gram)
    class(nystroem),intent(inout)::self
    real(dp),intent(in)::gram(:,:)
    real(dp),allocatable::values(:),u(:,:),v(:,:)
    integer::k,status
    call require(all(ieee_is_finite(gram)),'Nystroem nonfinite kernel matrix')
    k=size(gram,1);allocate(values(k),u(k,k),v(k,k));call kernel_svd(gram,u,values,v,status,240)
    call require(status==0,'Nystroem Jacobi SVD did not converge')
    values=1/sqrt(max(values,1.e-12_dp))
    self%normalization_=matmul(u*spread(values,1,k),transpose(v))
  end subroutine
  subroutine ny_fit(self,x)
    class(nystroem),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    type(legacy_random_state)::rng
    real(dp),allocatable::gram(:,:)
    integer,allocatable::order(:)
    integer::n,k,i,j,tmp
    call check_x(x);n=size(x,1)
    call require(self%n_components>0,'Nystroem n_components must be positive')
    call require(self%random_state>=0,'Nystroem random_state must be nonnegative')
    call ny_controls(self);self%n_features_in_=size(x,2)
    self%gamma_=self%gamma
    if(self%gamma_==-1) then
      self%gamma_=1._dp/size(x,2)
      if(self%kernel=='chi2') self%gamma_=1
    end if
    k=min(n,self%n_components);order=[(i,i=1,n)];call rng%seed(self%random_state)
    do i=n,2,-1
      j=rng%integer_index(i);tmp=order(i);order(i)=order(j);order(j)=tmp
    end do
    self%component_indices_=order(:k);self%components_=x(order(:k),:)
    if(allocated(self%components_csr_)) deallocate(self%components_csr_)
    gram=kernel_matrix(self,self%components_,self%components_,.true.)
    call ny_normalization(self,gram)
  end subroutine
  function ny_transform(self,x) result(z)
    class(nystroem),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    real(dp),allocatable::z(:,:)
    call require(allocated(self%normalization_),'Nystroem is not fitted')
    call check_xy(transpose(x),self%n_features_in_)
    if(allocated(self%components_csr_)) then
      if(self%kernel=='laplacian') self%components_csr_=sorted_sparse(self%components_csr_,.true.)
      z=fourier_dot(ny_sparse_kernel(self,dx=x,sy=self%components_csr_),transpose(self%normalization_))
    else
      z=fourier_dot(kernel_matrix(self,x,self%components_),transpose(self%normalization_))
    end if
  end function
  function ny_fit_transform(self,x) result(z)
    class(nystroem),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    real(dp),allocatable::z(:,:)
    call self%fit(x);z=self%transform(x)
  end function
  subroutine ny_fit_csr(self,x)
    class(nystroem),intent(inout)::self
    type(csr_matrix),intent(in)::x
    type(legacy_random_state)::rng
    integer,allocatable::order(:)
    real(dp),allocatable::gram(:,:)
    integer::n,k,i,j,tmp
    call ny_csr_input(x);call ny_controls(self)
    call require(self%n_components>0,'Nystroem n_components must be positive')
    call require(self%random_state>=0,'Nystroem random_state must be nonnegative')
    call require(self%kernel/='precomputed','Nystroem source SVD rejects precomputed CSR input')
    self%n_features_in_=x%n_cols;n=x%n_rows;k=min(n,self%n_components);order=[(i,i=1,n)]
    self%gamma_=self%gamma;if(self%gamma_==-1) self%gamma_=1._dp/x%n_cols
    call rng%seed(self%random_state)
    do i=n,2,-1
      j=rng%integer_index(i);tmp=order(i);order(i)=order(j);order(j)=tmp
    end do
    self%component_indices_=order(:k);self%components_csr_=ny_select_csr(x,order(:k))
    if(allocated(self%components_)) deallocate(self%components_)
    if(self%kernel=='laplacian') self%components_csr_=sorted_sparse(self%components_csr_,.true.)
    gram=ny_sparse_kernel(self,sx=self%components_csr_,sy=self%components_csr_,symmetric=.true.)
    call ny_normalization(self,gram)
  end subroutine
  function ny_transform_csr(self,x) result(z)
    class(nystroem),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::z(:,:)
    call require(allocated(self%normalization_),'Nystroem is not fitted')
    call ny_csr_input(x);call require(x%n_cols==self%n_features_in_,'Nystroem CSR feature count mismatch')
    if(allocated(self%components_csr_)) then
      if(self%kernel=='laplacian') self%components_csr_=sorted_sparse(self%components_csr_,.true.)
      z=fourier_dot(ny_sparse_kernel(self,sx=x,sy=self%components_csr_),transpose(self%normalization_))
    else
      z=fourier_dot(ny_sparse_kernel(self,sx=x,dy=self%components_),transpose(self%normalization_))
    end if
  end function
  function ny_fit_transform_csr(self,x) result(z)
    class(nystroem),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::z(:,:)
    call self%fit(x);z=self%transform(x)
  end function
  real(dp) function additive_interval(self) result(h)
    class(additive_chi2_sampler), intent(in) :: self
    call require(self%sample_steps>=1,'AdditiveChi2 sample_steps must be positive')
    call require(ieee_is_finite(self%sample_interval),'nonfinite AdditiveChi2 sample_interval')
    call require(self%sample_interval==-1.or.self%sample_interval>=0,'invalid AdditiveChi2 sample_interval')
    h=self%sample_interval
    if(h==-1) then
      select case(self%sample_steps)
      case(1); h=.8_dp
      case(2); h=.5_dp
      case(3); h=.4_dp
      case default; call require(.false.,'AdditiveChi2 needs an explicit interval for these steps')
      end select
    end if
  end function

  subroutine additive_fit(self,x)
    class(additive_chi2_sampler), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: interval
    interval=additive_interval(self)
    call check_x(x); call require(all(x>=0),'AdditiveChi2 requires nonnegative input')
    self%n_features_in_=size(x,2)
  end subroutine

  real(dp) function additive_sqrt(value) result(result_)
    real(dp),intent(in)::value
    result_=ieee_value(0._dp,ieee_quiet_nan)
    if(ieee_is_nan(value)) return
    if(value>=0) result_=sqrt(value)
  end function
  real(dp) function additive_cosh(value) result(result_)
    real(dp),intent(in)::value
    real(dp)::half
    result_=ieee_value(0._dp,ieee_quiet_nan)
    if(ieee_is_nan(value)) return
    if(.not.ieee_is_finite(value)) then
      result_=ieee_value(0._dp,ieee_positive_inf)
    else if(abs(value)<=log(huge(1._dp))) then
      result_=cosh(value)
    else if(abs(value)>2*log(huge(1._dp))) then
      result_=ieee_value(0._dp,ieee_positive_inf)
    else
      half=exp(abs(value)/2)
      result_=ieee_difference(ieee_product(.5_dp*half,half),-.5_dp*exp(-abs(value)))
    end if
  end function
  subroutine additive_harmonic(value,h,k,c,s)
    real(dp),intent(in)::value,h
    integer,intent(in)::k
    real(dp),intent(out)::c,s
    real(dp)::factor,phase,argument,step
    argument=ieee_product(ieee_product(acos(-1._dp),real(k,dp)),h)
    step=ieee_product(ieee_product(2._dp,value),h)
    factor=additive_sqrt(ieee_quotient(step,additive_cosh(argument)))
    phase=ieee_product(real(k,dp),ieee_product(h,skewed_log(value)))
    c=ieee_value(0._dp,ieee_quiet_nan);s=c
    if(ieee_is_finite(phase)) then
      c=ieee_product(factor,cos(phase));s=ieee_product(factor,sin(phase))
    end if
  end subroutine
  function additive_transform(self,x) result(z)
    use,intrinsic::iso_fortran_env,only:int64
    class(additive_chi2_sampler),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),allocatable::z(:,:)
    real(dp)::h,c,s
    integer::n,p,i,j,k,offset,width
    integer(int64)::wide
    h=additive_interval(self);call check_x(x)
    call require(all(x>=0),'AdditiveChi2 requires nonnegative input')
    n=size(x,1);p=size(x,2)
    if(self%n_features_in_>0) call require(p==self%n_features_in_,'AdditiveChi2 feature count mismatch')
    wide=(2_int64*self%sample_steps-1)*int(p,int64)
    call require(wide<=huge(1),'AdditiveChi2 output width overflow')
    width=int(wide);allocate(z(n,width));z=0
    do j=1,p
      do i=1,n
        if(x(i,j)==0) cycle
        z(i,j)=additive_sqrt(ieee_product(x(i,j),h))
        do k=1,self%sample_steps-1
          call additive_harmonic(x(i,j),h,k,c,s)
          offset=(2*k-1)*p;z(i,offset+j)=c;z(i,offset+p+j)=s
        end do
      end do
    end do
  end function
  subroutine additive_csr_input(self,x)
    class(additive_chi2_sampler),intent(in)::self
    type(csr_matrix),intent(in)::x
    call x%validate()
    call require(x%n_rows>0.and.x%n_cols>0,'AdditiveChi2 CSR input must be nonempty')
    call require(all(ieee_is_finite(x%data)),'AdditiveChi2 CSR input contains NaN or infinity')
    call require(all(x%data>=0),'AdditiveChi2 requires nonnegative input')
  end subroutine
  subroutine additive_fit_csr(self,x)
    class(additive_chi2_sampler),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp)::h
    h=additive_interval(self);call additive_csr_input(self,x)
    self%n_features_in_=x%n_cols
  end subroutine
  function additive_transform_csr(self,x) result(z)
    use,intrinsic::iso_fortran_env,only:int64
    class(additive_chi2_sampler),intent(in)::self
    type(csr_matrix),intent(in)::x
    type(csr_matrix)::z
    real(dp)::h,c,s
    integer(int64)::factor,width,count
    integer::i,k,j,start,nrow,entry,slot,cos_slot,sin_slot,p
    h=additive_interval(self);call additive_csr_input(self,x);p=x%n_cols
    if(self%n_features_in_>0) call require(p==self%n_features_in_,'AdditiveChi2 feature count mismatch')
    factor=2_int64*self%sample_steps-1;width=factor*p;count=factor*size(x%data,kind=int64)
    call require(width<=huge(1),'AdditiveChi2 output width overflow')
    call require(count<huge(1),'AdditiveChi2 CSR stored count overflow')
    z%n_rows=x%n_rows;z%n_cols=int(width)
    allocate(z%indptr(z%n_rows+1),z%indices(int(count)),z%data(int(count)))
    z%indptr(1)=1;start=1
    do i=1,x%n_rows
      nrow=x%indptr(i+1)-x%indptr(i)
      do j=1,nrow
        entry=x%indptr(i)+j-1;slot=start+j-1
        z%indices(slot)=x%indices(entry);z%data(slot)=additive_sqrt(ieee_product(x%data(entry),h))
      end do
      if(nrow>0) then
        do k=1,self%sample_steps-1
          do j=1,nrow
            entry=x%indptr(i)+j-1;cos_slot=start+(2*k-1)*nrow+j-1;sin_slot=cos_slot+nrow
            call additive_harmonic(x%data(entry),h,k,c,s)
            z%indices(cos_slot)=x%indices(entry)+(2*k-1)*p;z%data(cos_slot)=c
            z%indices(sin_slot)=x%indices(entry)+2*k*p;z%data(sin_slot)=s
          end do
        end do
      end if
      start=start+int(factor*nrow);z%indptr(i+1)=start
    end do
  end function
  function additive_fit_transform_csr(self,x) result(z)
    class(additive_chi2_sampler),intent(inout)::self
    type(csr_matrix),intent(in)::x
    type(csr_matrix)::z
    call self%fit(x);z=self%transform(x)
  end function

  function additive_fit_transform(self,x) result(z)
    class(additive_chi2_sampler), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x); z=self%transform(x)
  end function

  real(dp) function skewed_log(value) result(result_)
    real(dp),intent(in)::value
    result_=ieee_value(0._dp,ieee_quiet_nan)
    if(ieee_is_nan(value)) return
    if(value==0) then
      result_=ieee_value(0._dp,ieee_negative_inf)
    else if(value>0) then
      result_=log(value)
    end if
  end function
  subroutine skewed_fit(self,x)
    class(skewed_chi2_sampler),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    type(legacy_random_state)::rng
    real(dp)::pi,u
    integer::i,j
    call check_x(x)
    call require(self%n_components>=1,'SkewedChi2 n_components must be positive')
    call require(ieee_is_finite(self%skewedness),'nonfinite SkewedChi2 skewedness')
    call require(self%random_state>=0,'SkewedChi2 random_state must be nonnegative')
    self%n_features_in_=size(x,2);pi=acos(-1._dp)
    if(allocated(self%random_weights_)) deallocate(self%random_weights_)
    if(allocated(self%random_offset_)) deallocate(self%random_offset_)
    allocate(self%random_weights_(size(x,2),self%n_components),self%random_offset_(self%n_components))
    call rng%seed(self%random_state)
    do i=1,size(x,2)
      do j=1,self%n_components
        u=rng%uniform()
        self%random_weights_(i,j)=(1/pi)*skewed_log(tan((pi/2)*u))
      end do
    end do
    do j=1,self%n_components
      self%random_offset_(j)=rng%uniform()*(2*pi)
    end do
  end subroutine
  function skewed_transform(self,x) result(z)
    class(skewed_chi2_sampler),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),allocatable::z(:,:),shifted(:,:)
    real(dp)::phase,scale
    integer::i,j
    call require(allocated(self%random_weights_),'SkewedChi2 is not fitted')
    call require(ieee_is_finite(self%skewedness),'nonfinite SkewedChi2 skewedness')
    call check_xy(transpose(x),self%n_features_in_)
    call require(all(x>-self%skewedness),'SkewedChi2 input must exceed -skewedness')
    call require(self%n_components>0,'SkewedChi2 transform n_components must be positive')
    allocate(shifted(size(x,1),size(x,2)))
    do j=1,size(x,2)
      do i=1,size(x,1)
        shifted(i,j)=skewed_log(ieee_difference(x(i,j),-self%skewedness))
      end do
    end do
    z=fourier_dot(shifted,self%random_weights_)
    scale=sqrt(2._dp)/sqrt(real(self%n_components,dp))
    do j=1,size(z,2)
      do i=1,size(z,1)
        phase=ieee_difference(z(i,j),-self%random_offset_(j))
        z(i,j)=ieee_value(0._dp,ieee_quiet_nan)
        if(ieee_is_finite(phase)) z(i,j)=cos(phase)*scale
      end do
    end do
  end function

  function skewed_fit_transform(self,x) result(z)
    class(skewed_chi2_sampler), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x); z=self%transform(x)
  end function
end module
