! KBinsDiscretizer: sklearn 1.6.1 preprocessing/_discretization.py (BSD).
! NumPy RNG/reductions and weighted percentiles: retained notices in LICENSES/.
module sklearn_discretization
 use sklearn_common,only:dp,require,check_x
 use sklearn_numpy_ordering,only:argsort=>numpy_argsort,numpy_argpartition
 use sklearn_sparse,only:csr_matrix
 use sklearn_feature_extraction,only:feature_string
 use sklearn_legacy_rng,only:legacy_random_state,seed_legacy_uint32
 use sklearn_ieee_math,only:sub=>ieee_difference,mul=>ieee_product,div=>ieee_quotient
 use,intrinsic::iso_fortran_env,only:int64
 use,intrinsic::ieee_arithmetic
 implicit none
 private
 public::kbins_discretizer
 type::bin_edge_column
  real(dp),allocatable::values(:)
 end type
 type::kbins_discretizer
  integer::n_bins=5,n_features_out_=0,n_features_in_=0,subsample=200000
  integer(int64)::random_state=0_int64
  character(16)::strategy='quantile',encode='onehot-dense'
  integer,allocatable::n_bins_(:),n_iter_(:)
  type(bin_edge_column),allocatable::bin_edges_(:)
  type(feature_string),allocatable::feature_names_in_(:)
  logical::converged_=.false.
  integer,allocatable,private::encoder_bins_(:)
 contains
  procedure::fit=>bins_fit
  procedure::transform=>bins_transform
  procedure::transform_csr=>bins_csr
  procedure::fit_transform=>bins_fit_transform
  procedure::fit_transform_csr=>bins_fit_csr
  procedure,private::bins_inverse,bins_inverse_csr
  generic::inverse_transform=>bins_inverse,bins_inverse_csr
  procedure::get_feature_names_out=>bins_names
 end type
contains
 logical function le(a,b)
  real(dp),intent(in)::a,b
  le=.false.;if(ieee_is_nan(a).or.ieee_is_nan(b)) return
  le=a<=b
 end function
 recursive real(dp) function total(x) result(v)
  real(dp),intent(in)::x(:)
  real(dp)::r(8)
  integer::i,j,n,half,whole
  n=size(x);v=-0._dp
  if(n<8) then
   do i=1,n
    v=sub(v,-x(i))
   end do
  else if(n<=128) then
   r=x(:8);whole=n-mod(n,8)
   do i=9,whole,8
    do j=1,8
     r(j)=sub(r(j),-x(i+j-1))
    end do
   end do
   v=sub(sub(sub(r(1),-r(2)),-sub(r(3),-r(4))),-sub(sub(r(5),-r(6)),-sub(r(7),-r(8))))
   do i=whole+1,n
    v=sub(v,-x(i))
   end do
  else
   half=n/2;half=half-mod(half,8);v=sub(total(x(:half)),-total(x(half+1:)))
  end if
 end function
 function linspace(low,high,k) result(x)
  real(dp),intent(in)::low,high
  integer,intent(in)::k
  real(dp)::x(k+1),delta,step
  integer::i
  delta=sub(high,low);step=div(delta,real(k,dp))
  do i=0,k
   if(step==0) then
    x(i+1)=sub(low,-mul(real(i,dp)/k,delta))
   else
    x(i+1)=sub(low,-mul(real(i,dp),step))
   end if
  end do
  x(1)=low;x(k+1)=high
 end function
 function permutation(n,rng) result(ids)
  integer,intent(in)::n
  type(legacy_random_state),intent(inout)::rng
  integer::ids(n),i,j,t
  ids=[(i,i=1,n)]
  do i=n,2,-1
   j=rng%integer_index(i);t=ids(i);ids(i)=ids(j);ids(j)=t
  end do
 end function
  ! Weighted 1D Lloyd iterations with deterministic uniform-midpoint starts.
  subroutine bin_kmeans(x,w,k,edges,n_iter,converged)
    real(dp), intent(in) :: x(:),w(:)
    integer, intent(in) :: k
    real(dp), intent(out) :: edges(k+1)
    integer, intent(out) :: n_iter
    logical, intent(out) :: converged
    real(dp) :: col(size(x)),centers(k),next(k),mass(k),dist(size(x)),average,variance,shift,initial(k+1)
    integer :: labels(size(x)),previous(size(x)),order(k),far(size(x)),empty(k)
    integer :: i,j,it,pick,donor,n_empty,largest
    average=total(x)/size(x); col=x-average; variance=total(col**2)/size(x)
    initial=linspace(minval(x),maxval(x),k)
    do j=1,k
      centers(j)=(initial(j+1)+initial(j))*.5_dp-average
    end do
    previous=-1; converged=.false.
    do it=1,300
      next=0; mass=0
      do i=1,size(x)
        labels(i)=minloc((col(i)-centers)**2,dim=1)
        dist(i)=(col(i)-centers(labels(i)))**2
        next(labels(i))=next(labels(i))+w(i)*col(i)
        mass(labels(i))=mass(labels(i))+w(i)
      end do
      n_empty=0
      do j=1,k
        if(mass(j)/=0) cycle
        n_empty=n_empty+1; empty(n_empty)=j
      end do
      if(n_empty>0.and.maxval(dist)>0) then
        far=numpy_argpartition(dist,size(x)-n_empty+1)
        do j=1,n_empty
          pick=far(size(x)-j+1); donor=labels(pick)
          next(donor)=next(donor)-w(pick)*col(pick); mass(donor)=mass(donor)-w(pick)
          next(empty(j))=w(pick)*col(pick); mass(empty(j))=w(pick)
        end do
      end if
      largest=maxloc(mass,dim=1)
      do j=1,k
        if(mass(j)>0) then
          next(j)=next(j)*(1._dp/mass(j))
        else
          next(j)=next(largest)
        end if
      end do
      shift=sum((next-centers)**2); centers=next
      if(all(labels==previous).or.shift<=1.e-4_dp*variance) then
        converged=.true.; exit
      end if
      previous=labels
    end do
    n_iter=min(it,300); order=argsort(centers); centers=centers(order)+average
    edges(1)=minval(x); edges(k+1)=maxval(x)
    edges(2:k)=(centers(:k-1)+centers(2:))*0.5_dp
  end subroutine

 subroutine bins_fit(self,x,n_bins,sample_weight,feature_names,rng)
  class(kbins_discretizer),intent(inout)::self
  real(dp),intent(in)::x(:,:)
  integer,optional,intent(in)::n_bins(:)
  real(dp),optional,intent(in)::sample_weight(:)
  type(feature_string),optional,intent(in)::feature_names(:)
  type(legacy_random_state),optional,intent(inout)::rng
  type(legacy_random_state)::local_rng
  real(dp),allocatable::a(:,:),w(:),edges(:),sorted(:),cdf(:),percent(:)
  integer,allocatable::ids(:),order(:)
  logical,allocatable::keep(:)
  logical::converged,less
  real(dp)::low,high,q,t,delta,target
  integer::n,p,j,k,i,left,right,mid
  call check_x(x);n=size(x,1);p=size(x,2)
  call require(self%strategy=='uniform'.or.self%strategy=='quantile'.or.self%strategy=='kmeans', &
   'invalid KBinsDiscretizer strategy')
  call require(self%encode=='ordinal'.or.self%encode=='onehot'.or.self%encode=='onehot-dense', &
   'invalid KBinsDiscretizer encode')
  call require(self%subsample>=0,'KBinsDiscretizer subsample must be positive or zero for None')
  call require(self%random_state>=0.and.self%random_state<=int(z'FFFFFFFF',int64), &
   'KBinsDiscretizer random_state must fit uint32')
  self%n_features_in_=p
  if(allocated(self%feature_names_in_)) deallocate(self%feature_names_in_)
  if(present(feature_names)) then
   call require(size(feature_names)==p,'KBinsDiscretizer feature name count mismatch')
   do j=1,p
    call require(allocated(feature_names(j)%text),'KBinsDiscretizer feature name is unallocated')
   end do
   self%feature_names_in_=feature_names
  end if
  if(present(sample_weight)) call require(self%strategy/='uniform','uniform binning does not support sample weights')
  a=x
  if(self%subsample>0.and.n>self%subsample) then
   if(present(rng)) then
    ids=permutation(n,rng)
   else
    call seed_legacy_uint32(local_rng,self%random_state);ids=permutation(n,local_rng)
   end if
   a=x(ids(:self%subsample),:);n=size(a,1)
  end if
  self%n_bins_=spread(self%n_bins,1,p)
  if(present(n_bins)) then
   call require(size(n_bins)==p,'KBinsDiscretizer bin count shape mismatch');self%n_bins_=n_bins
  end if
  call require(all(self%n_bins_>=2),'KBinsDiscretizer requires at least two requested bins')
  call require(all(self%n_bins_<huge(1)-1),'KBinsDiscretizer requested bin count overflow')
  w=spread(1._dp,1,n)
  if(present(sample_weight)) then
   ! Baseline validates weights AFTER subsampling X; it does not resample weights.
   call require(size(sample_weight)==n,'KBinsDiscretizer weight count mismatch')
   call require(all(ieee_is_finite(sample_weight)),'invalid KBinsDiscretizer sample weights')
   w=sample_weight
  end if
  if(allocated(self%bin_edges_)) deallocate(self%bin_edges_)
  allocate(self%bin_edges_(p));self%n_iter_=spread(0,1,p);self%converged_=.true.
  do j=1,p
   k=self%n_bins_(j);low=minval(a(:,j));high=maxval(a(:,j))
   if(low==high) then
    self%n_bins_(j)=1
    self%bin_edges_(j)%values=[ieee_value(0._dp,ieee_negative_inf),ieee_value(0._dp,ieee_positive_inf)];cycle
   end if
   select case(self%strategy)
   case('uniform')
    edges=linspace(low,high,k)
   case('quantile')
    order=argsort(a(:,j));sorted=a(order,j);percent=linspace(0._dp,100._dp,k)
    if(allocated(edges)) deallocate(edges)
    allocate(edges(k+1))
    if(present(sample_weight)) then
     cdf=w(order)
     do i=2,n
      cdf(i)=sub(cdf(i-1),-cdf(i))
     end do
     do i=1,k+1
      target=mul(percent(i)/100,cdf(n))
      if(.not.ieee_is_nan(target)) then
       if(target==0) target=nearest(0._dp,1._dp)
      end if
      left=1;right=n+1
      do while(left<right)
       mid=left+(right-left)/2
       if(ieee_is_nan(target)) then
        less=.not.ieee_is_nan(cdf(mid))
       else
        less=.false.;if(.not.ieee_is_nan(cdf(mid))) less=cdf(mid)<target
       end if
       if(less) then
        left=mid+1
       else
        right=mid
       end if
      end do
      edges(i)=sorted(min(left,n))
     end do
    else
     do i=1,k+1
      q=percent(i)/100*real(n-1,dp);left=floor(q)+1;right=min(left+1,n);t=q-floor(q)
      delta=sub(sorted(right),sorted(left))
      if(t>=.5_dp) then
       edges(i)=sub(sorted(right),mul(1-t,delta))
      else
       edges(i)=sub(sorted(left),-mul(t,delta))
      end if
     end do
    end if
   case('kmeans')
    call require(k<=n,'kmeans bin count exceeds sample count')
    if(allocated(edges)) deallocate(edges)
    allocate(edges(k+1));call bin_kmeans(a(:,j),w,k,edges,self%n_iter_(j),converged)
    self%converged_=self%converged_.and.converged
   end select
   if(self%strategy/='uniform') then
    keep=spread(.true.,1,k+1)
    do i=2,k+1
     keep(i)=.false.;delta=sub(edges(i),edges(i-1))
     if(.not.ieee_is_nan(delta)) keep(i)=delta>1.e-8_dp
    end do
    edges=pack(edges,keep)
   end if
   self%n_bins_(j)=size(edges)-1;self%bin_edges_(j)%values=edges
  end do
  self%n_features_out_=p
  if(self%encode/='ordinal') then
   call require(all(self%n_bins_>0),'KBinsDiscretizer one-hot encoder has no categories')
   call require(sum(int(self%n_bins_,int64))<huge(1),'KBinsDiscretizer encoded width overflow')
   self%encoder_bins_=self%n_bins_;self%n_features_out_=sum(self%n_bins_)
  end if
 end subroutine
 subroutine fitted(self)
  class(kbins_discretizer),intent(in)::self
  call require(allocated(self%bin_edges_),'KBinsDiscretizer is not fitted')
 end subroutine
 subroutine encoder(self)
  class(kbins_discretizer),intent(in)::self
  call require(allocated(self%encoder_bins_),'KBinsDiscretizer fitted encoder is absent')
 end subroutine
 function ordinal(self,x) result(codes)
  class(kbins_discretizer),intent(in)::self
  real(dp),intent(in)::x(:,:)
  integer::codes(size(x,1),size(x,2)),i,j,left,right,mid,k
  call fitted(self);call check_x(x)
  call require(size(x,2)==self%n_features_in_,'KBinsDiscretizer feature count mismatch')
  do j=1,size(x,2)
   k=size(self%bin_edges_(j)%values)-2
   do i=1,size(x,1)
    left=1;right=k+1
    do while(left<right)
     mid=left+(right-left)/2
     if(le(self%bin_edges_(j)%values(mid+1),x(i,j))) then
      left=mid+1
     else
      right=mid
     end if
    end do
    codes(i,j)=left-1
   end do
  end do
 end function
 subroutine encode_check(self,codes)
  class(kbins_discretizer),intent(in)::self
  integer,intent(in)::codes(:,:)
  integer::j
  call encoder(self)
  call require(size(codes,2)==size(self%encoder_bins_),'KBinsDiscretizer cached encoder feature count mismatch')
  do j=1,size(codes,2)
   call require(all(codes(:,j)<self%encoder_bins_(j)),'KBinsDiscretizer cached encoder found unknown bin')
  end do
 end subroutine
 function bins_transform(self,x) result(z)
  class(kbins_discretizer),intent(in)::self
  real(dp),intent(in)::x(:,:)
  real(dp),allocatable::z(:,:)
  integer,allocatable::codes(:,:)
  integer::i,j,offset
  codes=ordinal(self,x)
  if(self%encode=='ordinal') then
   z=real(codes,dp);return
  end if
  call encode_check(self,codes)
  allocate(z(size(x,1),sum(self%encoder_bins_)));z=0;offset=0
  do j=1,size(x,2)
   do i=1,size(x,1)
    z(i,offset+codes(i,j)+1)=1
   end do
   offset=offset+self%encoder_bins_(j)
  end do
 end function
 function bins_csr(self,x) result(z)
  class(kbins_discretizer),intent(in)::self
  real(dp),intent(in)::x(:,:)
  type(csr_matrix)::z
  integer,allocatable::codes(:,:)
  integer::i,j,pos,offset
  call require(self%encode/='ordinal','KBinsDiscretizer CSR output requires onehot encoding')
  codes=ordinal(self,x);call encode_check(self,codes)
  call require(int(size(x,1),int64)*size(x,2)<huge(1),'KBinsDiscretizer sparse stored count overflow')
  z%n_rows=size(x,1);z%n_cols=sum(self%encoder_bins_)
  allocate(z%indptr(z%n_rows+1),z%data(size(codes)),z%indices(size(codes)));z%data=1;pos=1
  do i=1,z%n_rows
   z%indptr(i)=pos;offset=0
   do j=1,size(codes,2)
    z%indices(pos)=offset+codes(i,j)+1;pos=pos+1;offset=offset+self%encoder_bins_(j)
   end do
  end do
  z%indptr(z%n_rows+1)=pos;call z%validate()
 end function
 function centers_from_codes(self,codes) result(z)
  class(kbins_discretizer),intent(in)::self
  real(dp),intent(in)::codes(:,:)
  real(dp),allocatable::z(:,:)
  integer(int64)::b
  integer::i,j,k
  call check_x(codes)
  call require(size(codes,2)==size(self%n_bins_),'KBinsDiscretizer inverse feature count mismatch')
  allocate(z(size(codes,1),size(codes,2)))
  do j=1,size(codes,2)
   k=size(self%bin_edges_(j)%values)-1
   do i=1,size(codes,1)
    call require(abs(codes(i,j))<9223372036854775808._dp,'KBinsDiscretizer inverse bin out of range')
    b=int(codes(i,j),int64);if(b<0) b=b+k
    call require(b>=0.and.b<k,'KBinsDiscretizer inverse bin out of range')
    z(i,j)=mul(sub(self%bin_edges_(j)%values(int(b)+1),-self%bin_edges_(j)%values(int(b)+2)),.5_dp)
   end do
  end do
 end function
 integer function first_max(row) result(k)
  real(dp),intent(in)::row(:)
  integer::i
  k=1
  do i=1,size(row)
   if(ieee_is_nan(row(i))) then
    k=i;return
   end if
   if(row(i)>row(k)) k=i
  end do
 end function
 function bins_inverse(self,x) result(z)
  class(kbins_discretizer),intent(in)::self
  real(dp),intent(in)::x(:,:)
  real(dp),allocatable::z(:,:),codes(:,:)
  integer::i,j,k,offset
  call fitted(self)
  if(self%encode=='ordinal') then
   z=centers_from_codes(self,x);return
  end if
  call encoder(self);call check_x(x)
  call require(size(x,2)==sum(self%encoder_bins_),'KBinsDiscretizer encoded width mismatch')
  allocate(codes(size(x,1),size(self%encoder_bins_)));offset=0
  do j=1,size(self%encoder_bins_)
   k=self%encoder_bins_(j)
   do i=1,size(x,1)
    call require(.not.le(total(x(i,offset+1:offset+k)),0._dp).or. &
      .not.le(0._dp,total(x(i,offset+1:offset+k))),'KBinsDiscretizer inverse row has zero sum')
    codes(i,j)=real(first_max(x(i,offset+1:offset+k))-1,dp)
   end do
   offset=offset+k
  end do
  z=centers_from_codes(self,codes)
 end function
 function bins_inverse_csr(self,x) result(z)
  class(kbins_discretizer),intent(in)::self
  type(csr_matrix),intent(in)::x
  real(dp),allocatable::z(:,:),codes(:,:),row(:)
  real(dp)::sumrow
  integer::i,j,t,offset,k,col
  call fitted(self);call require(self%encode/='ordinal','KBinsDiscretizer ordinal inverse rejects CSR input')
  call encoder(self);call x%validate()
  call require(x%n_rows>0.and.x%n_cols>0,'KBinsDiscretizer inverse needs nonempty CSR input')
  call require(all(ieee_is_finite(x%data)),'KBinsDiscretizer inverse rejects nonfinite CSR data')
  call require(x%n_cols==sum(self%encoder_bins_),'KBinsDiscretizer encoded width mismatch')
  allocate(codes(x%n_rows,size(self%encoder_bins_)))
  do i=1,x%n_rows
   offset=0
   do j=1,size(self%encoder_bins_)
    k=self%encoder_bins_(j);row=spread(0._dp,1,k)
    do t=x%indptr(i),x%indptr(i+1)-1
     col=x%indices(t)-offset
     if(col<1.or.col>k) cycle
     row(col)=sub(row(col),-x%data(t))
    end do
    sumrow=0
    do t=1,k
     sumrow=sub(sumrow,-row(t))
    end do
    call require(.not.le(sumrow,0._dp).or..not.le(0._dp,sumrow),'KBinsDiscretizer inverse row has zero sum')
    codes(i,j)=real(first_max(row)-1,dp);offset=offset+k
   end do
  end do
  z=centers_from_codes(self,codes)
 end function
 function bins_names(self,input_features) result(names)
  class(kbins_discretizer),intent(in)::self
  type(feature_string),optional,intent(in)::input_features(:)
  type(feature_string),allocatable::names(:),input(:)
  integer::i,j,k
  character(32)::text
  call fitted(self)
  if(present(input_features)) then
   call require(size(input_features)==self%n_features_in_,'KBinsDiscretizer input name count mismatch')
   do i=1,size(input_features)
    call require(allocated(input_features(i)%text),'KBinsDiscretizer feature name is unallocated')
    if(allocated(self%feature_names_in_)) then
     call require(len(input_features(i)%text)==len(self%feature_names_in_(i)%text), &
      'KBinsDiscretizer input feature names differ')
     call require(input_features(i)%text==self%feature_names_in_(i)%text,'KBinsDiscretizer input feature names differ')
    end if
   end do
   input=input_features
  else if(allocated(self%feature_names_in_)) then
   input=self%feature_names_in_
  else
   allocate(input(self%n_features_in_))
   do i=1,size(input)
    write(text,'(i0)') i-1;input(i)%text='x'//trim(text)
   end do
  end if
  if(.not.allocated(self%encoder_bins_)) then
   names=input;return
  end if
  call require(size(input)==size(self%encoder_bins_),'KBinsDiscretizer cached encoder feature count mismatch')
  allocate(names(sum(self%encoder_bins_)));k=0
  do i=1,size(input)
   do j=0,self%encoder_bins_(i)-1
    write(text,'(i0)') j;k=k+1;names(k)%text=input(i)%text//'_'//trim(text)//'.0'
   end do
  end do
 end function
 function bins_fit_transform(self,x,n_bins,sample_weight,feature_names,rng) result(z)
  class(kbins_discretizer),intent(inout)::self
  real(dp),intent(in)::x(:,:)
  integer,optional,intent(in)::n_bins(:)
  real(dp),optional,intent(in)::sample_weight(:)
  type(feature_string),optional,intent(in)::feature_names(:)
  type(legacy_random_state),optional,intent(inout)::rng
  real(dp),allocatable::z(:,:)
  call self%fit(x,n_bins,sample_weight,feature_names,rng);z=self%transform(x)
 end function
 function bins_fit_csr(self,x,n_bins,sample_weight,feature_names,rng) result(z)
  class(kbins_discretizer),intent(inout)::self
  real(dp),intent(in)::x(:,:)
  integer,optional,intent(in)::n_bins(:)
  real(dp),optional,intent(in)::sample_weight(:)
  type(feature_string),optional,intent(in)::feature_names(:)
  type(legacy_random_state),optional,intent(inout)::rng
  type(csr_matrix)::z
  call self%fit(x,n_bins,sample_weight,feature_names,rng);z=self%transform_csr(x)
 end function
end module
