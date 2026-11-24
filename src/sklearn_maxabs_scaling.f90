! MaxAbsScaler/maxabs_scale contracts follow sklearn 1.6.1 preprocessing/_data.py.
! See THIRD_PARTY_NOTICES.md for retained sklearn BSD-3-Clause attribution.
module sklearn_maxabs_scaling
 use sklearn_common,only:dp,require
 use sklearn_sparse,only:csr_matrix
 use sklearn_stateless_preprocessing,only:sorted_sparse
 use sklearn_ieee_math,only:ieee_difference,ieee_product,ieee_quotient
 use,intrinsic::iso_fortran_env,only:int64
 use,intrinsic::ieee_arithmetic,only:ieee_is_nan,ieee_is_finite,ieee_value,ieee_quiet_nan
 implicit none
 private
 public::maxabs_scaler,maxabs_scale
 type::maxabs_scaler
  real(dp),allocatable::scale_(:),max_abs_(:)
  integer::n_features_in_=0
  integer(int64)::n_samples_seen_=0
 contains
  procedure,private::fit_dense,fit_csr,partial_dense,partial_csr
  procedure,private::transform_dense,transform_csr,inverse_dense,inverse_csr,fit_transform_dense,fit_transform_csr
  generic::fit=>fit_dense,fit_csr
  generic::partial_fit=>partial_dense,partial_csr
  generic::transform=>transform_dense,transform_csr
  generic::inverse_transform=>inverse_dense,inverse_csr
  generic::fit_transform=>fit_transform_dense,fit_transform_csr
 end type
 interface maxabs_scale
  module procedure scale_dense,scale_vector,scale_csr
 end interface
contains
 subroutine check_dense(x)
  real(dp),intent(in)::x(:,:)
  call require(size(x,1)>0.and.size(x,2)>0,'MaxAbsScaler input must be nonempty')
  call require(all(ieee_is_finite(x).or.ieee_is_nan(x)),'MaxAbsScaler input contains infinity')
 end subroutine
 subroutine check_sparse(x)
  type(csr_matrix),intent(in)::x
  call x%validate()
  call require(x%n_rows>0.and.x%n_cols>0,'MaxAbsScaler CSR input must be nonempty')
  call require(all(ieee_is_finite(x%data).or.ieee_is_nan(x%data)),'MaxAbsScaler input contains infinity')
 end subroutine
 subroutine fitted(self)
  class(maxabs_scaler),intent(in)::self
  call require(allocated(self%scale_),'maxabs_scaler is not fitted')
 end subroutine
 subroutine reset(self)
  class(maxabs_scaler),intent(inout)::self
  if(allocated(self%scale_)) deallocate(self%scale_)
  if(allocated(self%max_abs_)) deallocate(self%max_abs_)
  self%n_samples_seen_=0;self%n_features_in_=0
 end subroutine
 function dense_max(x) result(values)
  real(dp),intent(in)::x(:,:)
  real(dp)::values(size(x,2)),v
  integer::i,j
  values=ieee_value(0._dp,ieee_quiet_nan)
  do j=1,size(x,2)
   do i=1,size(x,1)
    v=x(i,j);if(ieee_is_nan(v)) cycle
    if(ieee_is_nan(values(j))) values(j)=0
    values(j)=max(values(j),abs(v))
   end do
  end do
 end function
 function sparse_max(x,axis) result(values)
  type(csr_matrix),intent(in)::x
  integer,intent(in)::axis
  real(dp),allocatable::values(:)
  type(csr_matrix)::a
  integer,allocatable::stored(:)
  integer::n,extent,i,k,j
  a=sorted_sparse(x,.true.)
  n=x%n_cols;extent=x%n_rows
  if(axis==1) then
   n=x%n_rows;extent=x%n_cols
  end if
  allocate(values(n),stored(n));values=ieee_value(0._dp,ieee_quiet_nan);stored=0
  do i=1,x%n_rows
   do k=a%indptr(i),a%indptr(i+1)-1
    j=a%indices(k);if(axis==1) j=i
    stored(j)=stored(j)+1
    if(ieee_is_nan(a%data(k))) cycle
    if(ieee_is_nan(values(j))) values(j)=0
    values(j)=max(values(j),abs(a%data(k)))
   end do
  end do
  do j=1,n
   if(stored(j)<extent.and.ieee_is_nan(values(j))) values(j)=0
  end do
 end function
 subroutine update(self,values,n)
  class(maxabs_scaler),intent(inout)::self
  real(dp),intent(in)::values(:)
  integer,intent(in)::n
  integer::j
  if(self%n_samples_seen_==0) then
   self%n_features_in_=size(values);self%max_abs_=values
  else
   call require(size(values)==self%n_features_in_,'MaxAbsScaler feature count mismatch')
   do j=1,size(values)
    if(ieee_is_nan(self%max_abs_(j)).or.ieee_is_nan(values(j))) then
     self%max_abs_(j)=ieee_value(0._dp,ieee_quiet_nan)
    else
     self%max_abs_(j)=max(self%max_abs_(j),values(j))
    end if
   end do
  end if
  call require(self%n_samples_seen_<=huge(0_int64)-int(n,int64),'MaxAbsScaler sample count overflow')
  self%n_samples_seen_=self%n_samples_seen_+int(n,int64);self%scale_=self%max_abs_
  do j=1,size(values)
   if(ieee_is_nan(self%scale_(j))) cycle
   if(self%scale_(j)<10*epsilon(1._dp)) self%scale_(j)=1
  end do
 end subroutine
 subroutine partial_dense(self,x)
  class(maxabs_scaler),intent(inout)::self
  real(dp),intent(in)::x(:,:)
  call check_dense(x)
  if(self%n_samples_seen_>0) call require(size(x,2)==self%n_features_in_,'MaxAbsScaler feature count mismatch')
  call update(self,dense_max(x),size(x,1))
 end subroutine
 subroutine partial_csr(self,x)
  class(maxabs_scaler),intent(inout)::self
  type(csr_matrix),intent(in)::x
  call check_sparse(x)
  if(self%n_samples_seen_>0) call require(x%n_cols==self%n_features_in_,'MaxAbsScaler feature count mismatch')
  call update(self,sparse_max(x,0),x%n_rows)
 end subroutine
 subroutine fit_dense(self,x)
  class(maxabs_scaler),intent(inout)::self
  real(dp),intent(in)::x(:,:)
  call reset(self);call self%partial_fit(x)
 end subroutine
 subroutine fit_csr(self,x)
  class(maxabs_scaler),intent(inout)::self
  type(csr_matrix),intent(in)::x
  call reset(self);call self%partial_fit(x)
 end subroutine
 function transform_dense(self,x) result(z)
  class(maxabs_scaler),intent(in)::self
  real(dp),intent(in)::x(:,:)
  real(dp)::z(size(x,1),size(x,2))
  integer::i,j
  call fitted(self);call check_dense(x)
  call require(size(x,2)==self%n_features_in_,'MaxAbsScaler feature count mismatch')
  do j=1,size(x,2)
   do i=1,size(x,1)
    z(i,j)=ieee_quotient(x(i,j),self%scale_(j))
   end do
  end do
 end function
 function transform_csr(self,x) result(z)
  class(maxabs_scaler),intent(in)::self
  type(csr_matrix),intent(in)::x
  type(csr_matrix)::z
  integer::k,j
  call fitted(self);call check_sparse(x)
  call require(x%n_cols==self%n_features_in_,'MaxAbsScaler feature count mismatch')
  z=x
  do k=1,size(z%data)
   j=z%indices(k);z%data(k)=ieee_product(z%data(k),ieee_quotient(1._dp,self%scale_(j)))
  end do
 end function
 function inverse_dense(self,x) result(z)
  class(maxabs_scaler),intent(in)::self
  real(dp),intent(in)::x(:,:)
  real(dp),allocatable::z(:,:)
  integer::i,j,p,a,b
  call fitted(self);call check_dense(x)
  p=size(self%scale_)
  call require(size(x,2)==p.or.p==1,'MaxAbsScaler inverse broadcast mismatch')
  p=max(size(x,2),p);allocate(z(size(x,1),p))
  do j=1,p
   a=min(j,size(x,2));b=min(j,size(self%scale_))
   do i=1,size(x,1)
    z(i,j)=ieee_product(x(i,a),self%scale_(b))
   end do
  end do
 end function
 function inverse_csr(self,x) result(z)
  class(maxabs_scaler),intent(in)::self
  type(csr_matrix),intent(in)::x
  type(csr_matrix)::z
  integer::k
  call fitted(self);call check_sparse(x)
  call require(x%n_cols==size(self%scale_),'MaxAbsScaler CSR inverse width mismatch')
  z=x
  do k=1,size(z%data)
   z%data(k)=ieee_product(z%data(k),self%scale_(z%indices(k)))
  end do
 end function
 function fit_transform_dense(self,x) result(z)
  class(maxabs_scaler),intent(inout)::self
  real(dp),intent(in)::x(:,:)
  real(dp)::z(size(x,1),size(x,2))
  call self%fit(x);z=self%transform(x)
 end function
 function fit_transform_csr(self,x) result(z)
  class(maxabs_scaler),intent(inout)::self
  type(csr_matrix),intent(in)::x
  type(csr_matrix)::z
  call self%fit(x);z=self%transform(x)
 end function
 function scale_dense(x,axis) result(z)
  real(dp),intent(in)::x(:,:)
  integer,intent(in),optional::axis
  real(dp)::z(size(x,1),size(x,2))
  type(maxabs_scaler)::model
  real(dp),allocatable::work(:,:)
  integer::along
  along=0;if(present(axis)) along=axis
  call require(along==0.or.along==1,'maxabs_scale axis must be zero or one')
  if(along==0) then
   z=model%fit_transform(x)
  else
   work=model%fit_transform(transpose(x));z=transpose(work)
  end if
 end function
 function scale_vector(x,axis) result(z)
  real(dp),intent(in)::x(:)
  integer,intent(in),optional::axis
  real(dp)::z(size(x))
  real(dp),allocatable::work(:,:)
  work=scale_dense(reshape(x,[size(x),1]),axis);z=work(:,1)
 end function
 function scale_csr(x,axis) result(z)
  type(csr_matrix),intent(in)::x
  integer,intent(in),optional::axis
  type(csr_matrix)::z
  type(maxabs_scaler)::model
  real(dp),allocatable::scales(:)
  integer::along,i,k
  along=0;if(present(axis)) along=axis
  call require(along==0.or.along==1,'maxabs_scale axis must be zero or one')
  call check_sparse(x)
  if(along==0) then
   z=model%fit_transform(x)
  else
   ! X.T is CSC in source: its reduction canonicalizes shared stored rows.
   ! Work on an owned CSR copy, preserving the caller's arrays.
   z=sorted_sparse(x,.true.);scales=sparse_max(z,1)
   call check_sparse(z)
   do i=1,size(scales)
    if(ieee_is_nan(scales(i))) cycle
    if(scales(i)<10*epsilon(1._dp)) scales(i)=1
   end do
   do i=1,z%n_rows
    do k=z%indptr(i),z%indptr(i+1)-1
     z%data(k)=ieee_product(z%data(k),ieee_quotient(1._dp,scales(i)))
    end do
   end do
  end if
 end function
end module
