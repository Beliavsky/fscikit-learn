! MinMaxScaler/minmax_scale follow sklearn 1.6.1 preprocessing/_data.py.
! Upstream BSD-3-Clause notices are retained under LICENSES/.
module sklearn_minmax_scaling
 use sklearn_common,only:dp,require
 use sklearn_ieee_math,only:ieee_difference,ieee_product,ieee_quotient
 use,intrinsic::iso_fortran_env,only:int64
 use,intrinsic::ieee_arithmetic,only:ieee_is_nan,ieee_is_finite,ieee_value,ieee_quiet_nan
 implicit none
 private
 public::minmax_scaler,minmax_scale
 type::minmax_scaler
  real(dp)::feature_min=0,feature_max=1
  logical::clip=.false.
  real(dp),allocatable::data_min_(:),data_max_(:),data_range_(:),scale_(:),min_(:)
  integer::n_features_in_=0
  integer(int64)::n_samples_seen_=0
 contains
  procedure::fit=>minmax_fit,partial_fit=>minmax_partial,fit_transform=>minmax_fit_transform
  procedure::transform=>minmax_transform,inverse_transform=>minmax_inverse
 end type
 interface minmax_scale
  module procedure scale_matrix,scale_vector
 end interface
contains
 subroutine valid_input(x)
  real(dp),intent(in)::x(:,:)
  call require(size(x,1)>0.and.size(x,2)>0,'MinMaxScaler input must be nonempty')
  call require(all(ieee_is_finite(x).or.ieee_is_nan(x)),'MinMaxScaler input contains infinity')
 end subroutine
 subroutine fitted(self)
  class(minmax_scaler),intent(in)::self
  call require(allocated(self%scale_),'minmax_scaler is not fitted')
 end subroutine
 subroutine minmax_fit(self,x)
  class(minmax_scaler),intent(inout)::self
  real(dp),intent(in)::x(:,:)
  if(allocated(self%scale_)) deallocate(self%scale_)
  if(allocated(self%min_)) deallocate(self%min_)
  if(allocated(self%data_min_)) deallocate(self%data_min_)
  if(allocated(self%data_max_)) deallocate(self%data_max_)
  if(allocated(self%data_range_)) deallocate(self%data_range_)
  self%n_features_in_=0;self%n_samples_seen_=0
  call self%partial_fit(x)
 end subroutine
 subroutine minmax_partial(self,x)
  class(minmax_scaler),intent(inout)::self
  real(dp),intent(in)::x(:,:)
  real(dp)::lo(size(x,2)),hi(size(x,2)),value,span,denom
  integer::i,j
  ! Source rejects equal/reversed endpoints but accepts NaN/infinite endpoints.
  if(.not.ieee_is_nan(self%feature_min).and..not.ieee_is_nan(self%feature_max)) &
   call require(self%feature_min<self%feature_max,'invalid feature range')
  call valid_input(x)
  if(self%n_samples_seen_>0) &
   call require(size(x,2)==self%n_features_in_,'MinMaxScaler feature count mismatch')
  lo=ieee_value(0._dp,ieee_quiet_nan);hi=lo
  do j=1,size(x,2)
   do i=1,size(x,1)
    value=x(i,j);if(ieee_is_nan(value)) cycle
    if(ieee_is_nan(lo(j))) then
     lo(j)=value;hi(j)=value
    else
     lo(j)=min(lo(j),value);hi(j)=max(hi(j),value)
    end if
   end do
   if(self%n_samples_seen_>0) then
    if(ieee_is_nan(lo(j)).or.ieee_is_nan(self%data_min_(j))) then
     lo(j)=ieee_value(0._dp,ieee_quiet_nan);hi(j)=lo(j)
    else
     lo(j)=min(lo(j),self%data_min_(j));hi(j)=max(hi(j),self%data_max_(j))
    end if
   end if
  end do
  call require(self%n_samples_seen_<=huge(0_int64)-int(size(x,1),int64),'MinMaxScaler sample count overflow')
  self%n_samples_seen_=self%n_samples_seen_+int(size(x,1),int64);self%n_features_in_=size(x,2)
  self%data_min_=lo;self%data_max_=hi
  self%data_range_=lo;self%scale_=lo;self%min_=lo
  span=ieee_difference(self%feature_max,self%feature_min)
  do j=1,size(lo)
   self%data_range_(j)=ieee_difference(hi(j),lo(j));denom=self%data_range_(j)
   if(.not.ieee_is_nan(denom)) then
    if(denom<10*epsilon(1._dp)) denom=1
   end if
   self%scale_(j)=ieee_quotient(span,denom)
   self%min_(j)=ieee_difference(self%feature_min,ieee_product(lo(j),self%scale_(j)))
  end do
 end subroutine
 function minmax_transform(self,x) result(z)
  class(minmax_scaler),intent(in)::self
  real(dp),intent(in)::x(:,:)
  real(dp)::z(size(x,1),size(x,2)),v
  integer::i,j
  call fitted(self);call valid_input(x)
  call require(size(x,2)==self%n_features_in_,'MinMaxScaler feature count mismatch')
  do j=1,size(x,2)
   do i=1,size(x,1)
    v=ieee_product(x(i,j),self%scale_(j));v=ieee_difference(v,-self%min_(j))
    if(self%clip) then
     if(ieee_is_nan(v).or.ieee_is_nan(self%feature_min).or.ieee_is_nan(self%feature_max)) then
      v=ieee_value(0._dp,ieee_quiet_nan)
     else
      ! np.clip uses minimum(maximum(v,lower),upper), even for reversed bounds.
      v=min(max(v,self%feature_min),self%feature_max)
     end if
    end if
    z(i,j)=v
   end do
  end do
 end function
 function minmax_inverse(self,x) result(z)
  class(minmax_scaler),intent(in)::self
  real(dp),intent(in)::x(:,:)
  real(dp)::z(size(x,1),size(x,2)),v
  integer::i,j,k
  call fitted(self);call valid_input(x)
  call require(size(x,2)==size(self%scale_).or.size(self%scale_)==1,'MinMaxScaler inverse broadcast mismatch')
  do j=1,size(x,2)
   k=min(j,size(self%scale_))
   do i=1,size(x,1)
    v=ieee_difference(x(i,j),self%min_(k));z(i,j)=ieee_quotient(v,self%scale_(k))
   end do
  end do
 end function
 function minmax_fit_transform(self,x) result(z)
  class(minmax_scaler),intent(inout)::self
  real(dp),intent(in)::x(:,:)
  real(dp)::z(size(x,1),size(x,2))
  call self%fit(x);z=self%transform(x)
 end function
 function scale_matrix(x,feature_min,feature_max,axis) result(z)
  real(dp),intent(in)::x(:,:)
  real(dp),intent(in),optional::feature_min,feature_max
  integer,intent(in),optional::axis
  real(dp)::z(size(x,1),size(x,2))
  type(minmax_scaler)::model
  real(dp),allocatable::work(:,:)
  integer::along
  along=0;if(present(axis)) along=axis
  call require(along==0.or.along==1,'minmax_scale axis must be zero or one')
  if(present(feature_min)) model%feature_min=feature_min
  if(present(feature_max)) model%feature_max=feature_max
  if(along==0) then
   z=model%fit_transform(x)
  else
   work=model%fit_transform(transpose(x));z=transpose(work)
  end if
 end function
 function scale_vector(x,feature_min,feature_max,axis) result(z)
  real(dp),intent(in)::x(:)
  real(dp),intent(in),optional::feature_min,feature_max
  integer,intent(in),optional::axis
  real(dp)::z(size(x))
  real(dp),allocatable::work(:,:)
  work=scale_matrix(reshape(x,[size(x),1]),feature_min,feature_max,axis);z=work(:,1)
 end function
end module
