! RobustScaler/robust_scale follow sklearn 1.6.1 preprocessing/_data.py.
! Retained upstream BSD-3-Clause notices are in LICENSES/.
module sklearn_robust_scaling
 use sklearn_common,only:dp,require,argsort
 use sklearn_sparse,only:csr_matrix
 use sklearn_ieee_math,only:ieee_difference,ieee_product,ieee_quotient
 use,intrinsic::ieee_arithmetic,only:ieee_is_nan,ieee_is_finite,ieee_value,ieee_quiet_nan,ieee_positive_inf
 implicit none
 private
 public::robust_scaler,robust_scale
 type::robust_scaler
  logical::with_centering=.true.,with_scaling=.true.,unit_variance=.false.
  real(dp)::quantile_low=25,quantile_high=75
  real(dp),allocatable::center_(:),scale_(:)
  integer::n_features_in_=0
 contains
  procedure,private::fit_dense,fit_csr,transform_dense,transform_csr,inverse_dense,inverse_csr
  procedure,private::fit_transform_dense,fit_transform_csr
  generic::fit=>fit_dense,fit_csr
  generic::transform=>transform_dense,transform_csr
  generic::inverse_transform=>inverse_dense,inverse_csr
  generic::fit_transform=>fit_transform_dense,fit_transform_csr
 end type
 interface robust_scale
  module procedure scale_matrix,scale_vector,scale_csr
 end interface
contains
 subroutine dense_input(x)
  real(dp),intent(in)::x(:,:)
  call require(size(x,1)>0.and.size(x,2)>0,'RobustScaler input must be nonempty')
  call require(all(ieee_is_finite(x).or.ieee_is_nan(x)),'RobustScaler input contains infinity')
 end subroutine
 subroutine sparse_input(x)
  type(csr_matrix),intent(in)::x
  call x%validate()
  call require(x%n_rows>0.and.x%n_cols>0,'RobustScaler CSR input must be nonempty')
  call require(all(ieee_is_finite(x%data).or.ieee_is_nan(x%data)),'RobustScaler input contains infinity')
 end subroutine
 subroutine controls(self,p)
  class(robust_scaler),intent(inout)::self
  integer,intent(in)::p
  call require(ieee_is_finite(self%quantile_low).and.ieee_is_finite(self%quantile_high),'invalid quantile range')
  call require(self%quantile_low>=0.and.self%quantile_low<=self%quantile_high.and.self%quantile_high<=100, &
   'invalid quantile range')
  if(allocated(self%center_)) deallocate(self%center_)
  if(allocated(self%scale_)) deallocate(self%scale_)
  self%n_features_in_=p
  if(self%with_centering) allocate(self%center_(p))
  if(self%with_scaling) allocate(self%scale_(p))
 end subroutine
 real(dp) function log_tail(z) result(value)
  real(dp),intent(in)::z
  real(dp)::term,total,previous
  integer::k
  if(z<10) then
   value=log(.5_dp*erfc(z/sqrt(2._dp)));return
  end if
  ! Asymptotic Gaussian tail avoids erfc underflow for subnormal probabilities.
  term=1;total=1
  do k=1,200
   previous=term;term=-term*real(2*k-1,dp)/(z*z)
   if(abs(term)>=abs(previous)) exit
   total=total+term
   if(abs(term)<epsilon(1._dp)*abs(total)) exit
  end do
  value=-.5_dp*z*z-log(z)-.5_dp*log(2*acos(-1._dp))+log(total)
 end function
 real(dp) function normal_ppf(p) result(value)
  real(dp),intent(in)::p
  real(dp)::low,high,mid,target
  integer::i
  if(p<=0.or.p>=1) then
   value=ieee_value(0._dp,ieee_positive_inf);if(p<=0) value=-value
   return
  end if
  if(p==.5_dp) then
   value=0;return
  end if
  target=log(min(p,1-p));low=0;high=40
  do i=1,80
   mid=.5_dp*(low+high)
   if(log_tail(mid)>target) then
    low=mid
   else
    high=mid
   end if
  end do
  value=.5_dp*(low+high);if(p<.5_dp) value=-value
 end function
 subroutine finish_scale(self)
  class(robust_scaler),intent(inout)::self
  real(dp)::adjust
  integer::j
  if(.not.self%with_scaling) return
  adjust=1
  if(self%unit_variance) adjust=ieee_difference(normal_ppf(self%quantile_high/100),normal_ppf(self%quantile_low/100))
  do j=1,size(self%scale_)
   if(.not.ieee_is_nan(self%scale_(j))) then
    if(self%scale_(j)<10*epsilon(1._dp)) self%scale_(j)=1
   end if
   if(self%unit_variance) self%scale_(j)=ieee_quotient(self%scale_(j),adjust)
  end do
 end subroutine
 real(dp) function element(a,zeros,negative,k) result(value)
  real(dp),intent(in)::a(:)
  integer,intent(in)::zeros,negative,k
  if(k<=negative) then
   value=a(k)
  else if(k<=negative+zeros) then
   value=0
  else
   value=a(k-zeros)
  end if
 end function
 real(dp) function quantile(a,zeros,q) result(value)
  real(dp),intent(in)::a(:),q
  integer,intent(in)::zeros
  integer::n,left,right,negative
  real(dp)::position,t,lo,hi,difference
  n=size(a)+zeros;value=ieee_value(0._dp,ieee_quiet_nan)
  if(n==0) return
  negative=count(a<0)
  position=real(n-1,dp)*(q/100);left=floor(position)+1;right=min(left+1,n);t=position-floor(position)
  lo=element(a,zeros,negative,left);hi=element(a,zeros,negative,right);difference=ieee_difference(hi,lo)
  if(t>=.5_dp) then
   value=ieee_difference(hi,ieee_product(difference,1-t))
  else
   value=ieee_difference(lo,-ieee_product(difference,t))
  end if
 end function
 subroutine fit_dense(self,x)
  class(robust_scaler),intent(inout)::self
  real(dp),intent(in)::x(:,:)
  real(dp),allocatable::a(:)
  integer::j,n,left,right
  call dense_input(x);call controls(self,size(x,2))
  do j=1,size(x,2)
   if(.not.self%with_centering.and..not.self%with_scaling) exit
   a=pack(x(:,j),.not.ieee_is_nan(x(:,j)));n=size(a);a=a(argsort(a))
   if(self%with_centering) then
    self%center_(j)=ieee_value(0._dp,ieee_quiet_nan)
    if(n>0) then
     if(size(x,1)<600) then
      ! NumPy's masked 2-D median duplicates the middle value for odd counts.
      ! Preserve its overflow behavior, unlike the large-axis 1-D dispatch.
      left=(n+1)/2;right=n/2+1
      self%center_(j)=ieee_quotient(ieee_difference(a(left),-a(right)),2._dp)
     else if(modulo(n,2)==1) then
      self%center_(j)=a(n/2+1)
     else
      self%center_(j)=ieee_quotient(ieee_difference(a(n/2),-a(n/2+1)),2._dp)
     end if
    end if
   end if
   if(self%with_scaling) self%scale_(j)= &
    ieee_difference(quantile(a,0,self%quantile_high),quantile(a,0,self%quantile_low))
  end do
  call finish_scale(self)
 end subroutine
 subroutine fit_csr(self,x)
  class(robust_scaler),intent(inout)::self
  type(csr_matrix),intent(in)::x
  integer,allocatable::ptr(:),next(:)
  real(dp),allocatable::values(:),a(:)
  integer::j,k,stored,zeros
  call sparse_input(x);call require(.not.self%with_centering,'RobustScaler cannot center sparse matrices')
  if(self%with_scaling) call require(x%n_cols<huge(1),'RobustScaler sparse feature count overflow')
  call controls(self,x%n_cols)
  if(.not.self%with_scaling) return
  allocate(ptr(x%n_cols+1));ptr=0
  do k=1,size(x%data)
   j=x%indices(k);ptr(j+1)=ptr(j+1)+1
  end do
  ptr(1)=1
  do j=1,x%n_cols
   ptr(j+1)=ptr(j+1)+ptr(j)
  end do
  next=ptr;allocate(values(size(x%data)))
  do k=1,size(x%data)
   j=x%indices(k);values(next(j))=x%data(k);next(j)=next(j)+1
  end do
  do j=1,x%n_cols
   stored=ptr(j+1)-ptr(j)
   call require(stored<=x%n_rows,'RobustScaler source sparse column has more stored entries than rows')
   zeros=x%n_rows-stored;a=pack(values(ptr(j):ptr(j+1)-1),.not.ieee_is_nan(values(ptr(j):ptr(j+1)-1)))
   a=a(argsort(a))
   self%scale_(j)=ieee_difference(quantile(a,zeros,self%quantile_high),quantile(a,zeros,self%quantile_low))
  end do
  call finish_scale(self)
 end subroutine
 subroutine ready(self)
  class(robust_scaler),intent(in)::self
  call require(self%n_features_in_>0,'robust_scaler is not fitted')
 end subroutine
 function transform_dense(self,x) result(z)
  class(robust_scaler),intent(in)::self
  real(dp),intent(in)::x(:,:)
  real(dp)::z(size(x,1),size(x,2)),v
  integer::i,j
  call ready(self);call dense_input(x)
  call require(size(x,2)==self%n_features_in_,'RobustScaler feature count mismatch')
  if(self%with_centering) call require(allocated(self%center_),'RobustScaler fitted center is absent')
  if(self%with_scaling) call require(allocated(self%scale_),'RobustScaler fitted scale is absent')
  do j=1,size(x,2)
   do i=1,size(x,1)
    v=x(i,j)
    if(self%with_centering) v=ieee_difference(v,self%center_(j))
    if(self%with_scaling) v=ieee_quotient(v,self%scale_(j))
    z(i,j)=v
   end do
  end do
 end function
 function transform_csr(self,x) result(z)
  class(robust_scaler),intent(in)::self
  type(csr_matrix),intent(in)::x
  type(csr_matrix)::z
  integer::k
  call ready(self);call sparse_input(x)
  call require(x%n_cols==self%n_features_in_,'RobustScaler feature count mismatch');z=x
  if(.not.self%with_scaling) return
  call require(allocated(self%scale_),'RobustScaler fitted scale is absent')
  do k=1,size(z%data)
   z%data(k)=ieee_product(z%data(k),ieee_quotient(1._dp,self%scale_(z%indices(k))))
  end do
 end function
 function inverse_dense(self,x) result(z)
  class(robust_scaler),intent(in)::self
  real(dp),intent(in)::x(:,:)
  real(dp)::z(size(x,1),size(x,2)),v
  integer::i,j,k
  call ready(self);call dense_input(x)
  if(self%with_centering) call require(allocated(self%center_),'RobustScaler fitted center is absent')
  if(self%with_scaling) call require(allocated(self%scale_),'RobustScaler fitted scale is absent')
  if(self%with_centering.or.self%with_scaling) &
   call require(size(x,2)==self%n_features_in_.or.self%n_features_in_==1,'RobustScaler inverse broadcast mismatch')
  do j=1,size(x,2)
   k=min(j,self%n_features_in_)
   do i=1,size(x,1)
    v=x(i,j)
    if(self%with_scaling) v=ieee_product(v,self%scale_(k))
    if(self%with_centering) v=ieee_difference(v,-self%center_(k))
    z(i,j)=v
   end do
  end do
 end function
 function inverse_csr(self,x) result(z)
  class(robust_scaler),intent(in)::self
  type(csr_matrix),intent(in)::x
  type(csr_matrix)::z
  integer::k
  call ready(self);call sparse_input(x);z=x
  if(.not.self%with_scaling) return
  call require(allocated(self%scale_),'RobustScaler fitted scale is absent')
  call require(x%n_cols==size(self%scale_),'RobustScaler CSR inverse width mismatch')
  do k=1,size(z%data)
   z%data(k)=ieee_product(z%data(k),self%scale_(z%indices(k)))
  end do
 end function
 function fit_transform_dense(self,x) result(z)
  class(robust_scaler),intent(inout)::self
  real(dp),intent(in)::x(:,:)
  real(dp)::z(size(x,1),size(x,2))
  call self%fit(x);z=self%transform(x)
 end function
 function fit_transform_csr(self,x) result(z)
  class(robust_scaler),intent(inout)::self
  type(csr_matrix),intent(in)::x
  type(csr_matrix)::z
  call self%fit(x);z=self%transform(x)
 end function
 function scale_matrix(x,with_centering,with_scaling,quantile_low,quantile_high,unit_variance,axis) result(z)
  real(dp),intent(in)::x(:,:)
  logical,intent(in),optional::with_centering,with_scaling,unit_variance
  real(dp),intent(in),optional::quantile_low,quantile_high
  integer,intent(in),optional::axis
  real(dp)::z(size(x,1),size(x,2))
  type(robust_scaler)::model
  real(dp),allocatable::work(:,:)
  integer::along
  call configure(model,with_centering,with_scaling,quantile_low,quantile_high,unit_variance,axis,along)
  if(along==0) then
   z=model%fit_transform(x)
  else
   work=model%fit_transform(transpose(x));z=transpose(work)
  end if
 end function
 function scale_vector(x,with_centering,with_scaling,quantile_low,quantile_high,unit_variance,axis) result(z)
  real(dp),intent(in)::x(:)
  logical,intent(in),optional::with_centering,with_scaling,unit_variance
  real(dp),intent(in),optional::quantile_low,quantile_high
  integer,intent(in),optional::axis
  real(dp)::z(size(x))
  real(dp),allocatable::work(:,:)
  work=scale_matrix(reshape(x,[size(x),1]),with_centering,with_scaling,quantile_low,quantile_high,unit_variance,axis)
  z=work(:,1)
 end function
 subroutine configure(model,centering,scaling,low,high,unit,axis,along)
  type(robust_scaler),intent(out)::model
  logical,intent(in),optional::centering,scaling,unit
  real(dp),intent(in),optional::low,high
  integer,intent(in),optional::axis
  integer,intent(out)::along
  if(present(centering)) model%with_centering=centering
  if(present(scaling)) model%with_scaling=scaling
  if(present(low)) model%quantile_low=low
  if(present(high)) model%quantile_high=high
  if(present(unit)) model%unit_variance=unit
  along=0;if(present(axis)) along=axis
  call require(along==0.or.along==1,'robust_scale axis must be zero or one')
 end subroutine
 function scale_csr(x,with_centering,with_scaling,quantile_low,quantile_high,unit_variance,axis) result(z)
  type(csr_matrix),intent(in)::x
  logical,intent(in),optional::with_centering,with_scaling,unit_variance
  real(dp),intent(in),optional::quantile_low,quantile_high
  integer,intent(in),optional::axis
  type(csr_matrix)::z
  type(robust_scaler)::model
  real(dp),allocatable::a(:)
  integer::along,i,k,n,stored
  call configure(model,with_centering,with_scaling,quantile_low,quantile_high,unit_variance,axis,along)
  if(along==0) then
   z=model%fit_transform(x);return
  end if
  call sparse_input(x);call require(.not.model%with_centering,'RobustScaler cannot center sparse matrices')
  call controls(model,x%n_rows);z=x
  if(.not.model%with_scaling) return
  do i=1,x%n_rows
   stored=x%indptr(i+1)-x%indptr(i)
   call require(stored<=x%n_cols,'RobustScaler source sparse column has more stored entries than rows')
   a=pack(x%data(x%indptr(i):x%indptr(i+1)-1),.not.ieee_is_nan(x%data(x%indptr(i):x%indptr(i+1)-1)))
   a=a(argsort(a));n=x%n_cols-stored
   model%scale_(i)=ieee_difference(quantile(a,n,model%quantile_high),quantile(a,n,model%quantile_low))
  end do
  call finish_scale(model)
  do i=1,x%n_rows
   do k=x%indptr(i),x%indptr(i+1)-1
    z%data(k)=ieee_product(z%data(k),ieee_quotient(1._dp,model%scale_(i)))
   end do
  end do
 end function
end module
