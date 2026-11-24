! StandardScaler follows sklearn 1.6.1 preprocessing/_data.py, extmath.py and
! sparsefuncs_fast.pyx. Retained BSD-3-Clause notices are under LICENSES/.
module sklearn_standard_scaling
 use sklearn_common,only:dp,require
 use sklearn_sparse,only:csr_matrix
 use sklearn_ieee_math,only:ieee_difference,ieee_product,ieee_quotient
 use,intrinsic::iso_fortran_env,only:int64
 use,intrinsic::ieee_arithmetic,only:ieee_is_nan,ieee_is_finite,ieee_value,ieee_quiet_nan
 implicit none
 private
 public::standard_scaler
 type::standard_scaler
  logical::with_mean=.true.,with_std=.true.
  real(dp),allocatable::mean_(:),var_(:),scale_(:)
  integer::n_features_in_=0
  ! Source scalar/vector and integer/real counts share a real64 vector here.
  real(dp),allocatable::n_samples_seen_(:)
  logical::sample_count_scalar_=.false.,sample_count_real_=.false.
 contains
  procedure,private::fit_dense,fit_csr,fit_dense_scalar,fit_csr_scalar
  procedure,private::partial_dense,partial_csr,partial_dense_scalar,partial_csr_scalar
  procedure,private::transform_dense,transform_csr,inverse_dense,inverse_csr
  procedure,private::fit_transform_dense,fit_transform_csr,fit_transform_dense_scalar,fit_transform_csr_scalar
  generic::fit=>fit_dense,fit_csr,fit_dense_scalar,fit_csr_scalar
  generic::partial_fit=>partial_dense,partial_csr,partial_dense_scalar,partial_csr_scalar
  generic::transform=>transform_dense,transform_csr
  generic::inverse_transform=>inverse_dense,inverse_csr
  generic::fit_transform=>fit_transform_dense,fit_transform_csr,fit_transform_dense_scalar,fit_transform_csr_scalar
 end type
contains
 real(dp) function add(a,b) result(v)
  real(dp),intent(in)::a,b
  v=ieee_difference(a,-b)
 end function
 real(dp) function square(a) result(v)
  real(dp),intent(in)::a
  v=ieee_product(a,a)
 end function
 recursive real(dp) function numpy_total(a) result(v)
  real(dp),intent(in)::a(:)
  real(dp)::lanes(8)
  integer::i,j,n,split,last
  n=size(a);v=-0._dp
  if(n<8) then
   do i=1,n
    v=add(v,a(i))
   end do
  else if(n<=128) then
   lanes=a(:8);last=n-modulo(n,8)
   do i=9,last,8
    do j=1,8
     lanes(j)=add(lanes(j),a(i+j-1))
    end do
   end do
   v=add(add(add(lanes(1),lanes(2)),add(lanes(3),lanes(4))), &
    add(add(lanes(5),lanes(6)),add(lanes(7),lanes(8))))
   do i=last+1,n
    v=add(v,a(i))
   end do
  else
   split=n/2;split=split-modulo(split,8);v=add(numpy_total(a(:split)),numpy_total(a(split+1:)))
  end if
 end function
 real(dp) function total(a,skip_nan,pairwise) result(v)
  real(dp),intent(in)::a(:)
  logical,intent(in)::skip_nan,pairwise
  real(dp)::values(size(a))
  integer::i
  values=a
  if(skip_nan) then
   where(ieee_is_nan(values)) values=0
  end if
  if(pairwise) then
   v=numpy_total(values)
  else
   v=0
   do i=1,size(a)
    v=add(v,values(i))
   end do
  end if
 end function
 subroutine dense_input(x)
  real(dp),intent(in)::x(:,:)
  call require(size(x,1)>0.and.size(x,2)>0,'StandardScaler input must be nonempty')
  call require(all(ieee_is_finite(x).or.ieee_is_nan(x)),'StandardScaler input contains infinity')
 end subroutine
 subroutine sparse_input(x)
  type(csr_matrix),intent(in)::x
  call x%validate()
  call require(x%n_rows>0.and.x%n_cols>0,'StandardScaler CSR input must be nonempty')
  call require(all(ieee_is_finite(x%data).or.ieee_is_nan(x%data)),'StandardScaler input contains infinity')
 end subroutine
 subroutine weight_input(w,n)
  real(dp),intent(in)::w(:)
  integer,intent(in)::n
  call require(size(w)==n,'StandardScaler sample_weight length mismatch')
  call require(all(ieee_is_finite(w)),'StandardScaler nonfinite sample_weight vector')
 end subroutine
 real(dp) function integer_cast(v) result(value)
  real(dp),intent(in)::v
  ! Match NumPy's astype(int64) result at nonfinite/out-of-range values.
  value=-real(huge(0_int64),dp)
  if(.not.ieee_is_finite(v)) return
  if(v>=real(huge(0_int64),dp).or.v< -real(huge(0_int64),dp)) return
  value=real(int(v,int64),dp)
 end function
 subroutine reset(self)
  class(standard_scaler),intent(inout)::self
  if(allocated(self%mean_)) deallocate(self%mean_)
  if(allocated(self%var_)) deallocate(self%var_)
  if(allocated(self%scale_)) deallocate(self%scale_)
  if(allocated(self%n_samples_seen_)) deallocate(self%n_samples_seen_)
  self%n_features_in_=0;self%sample_count_scalar_=.false.;self%sample_count_real_=.false.
 end subroutine
 subroutine prepare(self,p,weighted)
  class(standard_scaler),intent(inout)::self
  integer,intent(in)::p
  logical,intent(in)::weighted
  integer::j
  if(self%n_features_in_==0) then
   self%n_features_in_=p;allocate(self%n_samples_seen_(p));self%n_samples_seen_=0
   self%sample_count_real_=weighted
  else
   call require(p==self%n_features_in_,'StandardScaler feature count mismatch')
   if(self%sample_count_scalar_.or.size(self%n_samples_seen_)==1) then
    self%sample_count_real_=weighted
    if(.not.weighted) then
     do j=1,p
      self%n_samples_seen_(j)=integer_cast(self%n_samples_seen_(j))
     end do
    end if
   end if
  end if
 end subroutine
 subroutine finish(self)
  class(standard_scaler),intent(inout)::self
  real(dp)::bound,eps,n,m,v
  integer::j
  self%sample_count_scalar_=all(ieee_is_finite(self%n_samples_seen_))
  if(self%sample_count_scalar_) self%sample_count_scalar_=all(self%n_samples_seen_==self%n_samples_seen_(1))
  if(allocated(self%scale_)) deallocate(self%scale_)
  if(.not.self%with_std) return
  call require(allocated(self%var_).and.allocated(self%mean_),'StandardScaler fitted variance is absent')
  allocate(self%scale_(self%n_features_in_));eps=epsilon(1._dp)
  do j=1,self%n_features_in_
   n=self%n_samples_seen_(j);m=self%mean_(j);v=self%var_(j)
   bound=add(ieee_product(ieee_product(n,eps),v),square(ieee_product(ieee_product(n,m),eps)))
   self%scale_(j)=ieee_value(0._dp,ieee_quiet_nan)
   if(.not.ieee_is_nan(v)) then
    if(v>=0) self%scale_(j)=sqrt(v)
   end if
   if(.not.ieee_is_nan(v).and..not.ieee_is_nan(bound)) then
    if(v<=bound) self%scale_(j)=1
   end if
  end do
 end subroutine
 subroutine dense_update(self,x,w)
  class(standard_scaler),intent(inout)::self
  real(dp),intent(in)::x(:,:)
  real(dp),intent(in),optional::w(:)
  real(dp)::terms(size(x,1)),temp(size(x,1)),count_new,count_old,count_total,new_sum,last_sum,t,correction
  real(dp)::new_uvar,last_uvar,updated,ratio,new_mean,new_var
  logical::first,do_var,nan_mode,pairwise
  integer::i,j
  first=self%n_features_in_==0;call prepare(self,size(x,2),present(w));pairwise=size(x,2)==1
  if(.not.self%with_mean.and..not.self%with_std) then
   if(allocated(self%mean_)) deallocate(self%mean_)
   if(allocated(self%var_)) deallocate(self%var_)
   do j=1,size(x,2)
    self%n_samples_seen_(j)=add(self%n_samples_seen_(j),real(count(.not.ieee_is_nan(x(:,j))),dp))
   end do
   call finish(self);return
  end if
  if(first) then
   allocate(self%mean_(size(x,2)));self%mean_=0
   if(self%with_std) then
    allocate(self%var_(size(x,2)));self%var_=0
   end if
  end if
  call require(allocated(self%mean_),'StandardScaler fitted mean is absent')
  do_var=allocated(self%var_);nan_mode=any(ieee_is_nan(x))
  do j=1,size(x,2)
   count_old=self%n_samples_seen_(j);last_sum=ieee_product(self%mean_(j),count_old)
   if(present(w)) then
    do i=1,size(x,1)
     terms(i)=ieee_product(w(i),merge(0._dp,1._dp,ieee_is_nan(x(i,j))))
    end do
    count_new=total(terms,.false.,pairwise)
    do i=1,size(x,1)
     t=x(i,j);if(ieee_is_nan(t)) t=0
     terms(i)=ieee_product(w(i),t)
    end do
    new_sum=total(terms,.false.,.false.)
   else
    count_new=real(count(.not.ieee_is_nan(x(:,j))),dp);new_sum=total(x(:,j),nan_mode,pairwise)
   end if
   count_total=add(count_old,count_new);new_mean=ieee_quotient(add(last_sum,new_sum),count_total)
   if(do_var) then
    t=ieee_quotient(new_sum,count_new)
    do i=1,size(x,1)
     temp(i)=ieee_difference(x(i,j),t)
    end do
    if(present(w)) then
     do i=1,size(x,1)
      t=temp(i);if(ieee_is_nan(x(i,j))) t=0
      terms(i)=ieee_product(w(i),t)
     end do
     correction=total(terms,.false.,.false.)
     do i=1,size(x,1)
      t=square(temp(i));if(ieee_is_nan(x(i,j))) t=0
      terms(i)=ieee_product(w(i),t)
     end do
     new_uvar=total(terms,.false.,.false.)
    else
     correction=total(temp,nan_mode,pairwise)
     do i=1,size(x,1)
      terms(i)=square(temp(i))
     end do
     new_uvar=total(terms,nan_mode,pairwise)
    end if
    new_uvar=ieee_difference(new_uvar,ieee_quotient(square(correction),count_new))
    last_uvar=ieee_product(self%var_(j),count_old);ratio=ieee_quotient(count_old,count_new)
    t=square(ieee_difference(ieee_quotient(last_sum,ratio),new_sum))
    updated=add(add(last_uvar,new_uvar),ieee_product(ieee_quotient(ratio,count_total),t))
    if(count_old==0) updated=new_uvar
    new_var=ieee_quotient(updated,count_total);self%var_(j)=new_var
   end if
   self%mean_(j)=new_mean;self%n_samples_seen_(j)=count_total
  end do
  if(present(w)) self%sample_count_real_=.true.
  call finish(self)
 end subroutine
 subroutine sparse_batch(x,w,mean,var,n)
  type(csr_matrix),intent(in)::x
  real(dp),intent(in)::w(:)
  real(dp),intent(out)::mean(:),var(:),n(:)
  real(dp)::nz_weight(x%n_cols),correction(x%n_cols),diff,missing
  integer(int64)::counts(x%n_cols),nz_counts(x%n_cols)
  integer::i,k,j
  mean=0;var=0;n=numpy_total(w);nz_weight=0;correction=0;counts=x%n_rows;nz_counts=0
  do i=1,x%n_rows
   do k=x%indptr(i),x%indptr(i+1)-1
    j=x%indices(k)
    if(ieee_is_nan(x%data(k))) then
     n(j)=ieee_difference(n(j),w(i));counts(j)=counts(j)-1
    else
     mean(j)=add(mean(j),ieee_product(x%data(k),w(i)));nz_weight(j)=add(nz_weight(j),w(i))
     nz_counts(j)=nz_counts(j)+1
    end if
   end do
  end do
  do j=1,x%n_cols
   mean(j)=ieee_quotient(mean(j),n(j))
  end do
  do i=1,x%n_rows
   do k=x%indptr(i),x%indptr(i+1)-1
    if(ieee_is_nan(x%data(k))) cycle
    j=x%indices(k);diff=ieee_difference(x%data(k),mean(j))
    correction(j)=add(correction(j),ieee_product(diff,w(i)))
    var(j)=add(var(j),ieee_product(square(diff),w(i)))
   end do
  end do
  do j=1,x%n_cols
   missing=ieee_difference(n(j),nz_weight(j))
   if(counts(j)/=nz_counts(j)) correction(j)=ieee_difference(correction(j),ieee_product(missing,mean(j)))
   correction(j)=ieee_quotient(square(correction(j)),n(j))
   if(counts(j)/=nz_counts(j)) var(j)=add(var(j),ieee_product(missing,square(mean(j))))
   var(j)=ieee_quotient(ieee_difference(var(j),correction(j)),n(j))
  end do
 end subroutine
 subroutine csr_update(self,x,w,weighted)
  class(standard_scaler),intent(inout)::self
  type(csr_matrix),intent(in)::x
  real(dp),intent(in)::w(:)
  logical,intent(in)::weighted
  real(dp)::mean(x%n_cols),var(x%n_cols),n(x%n_cols),nan_weight(x%n_cols),sum_weights,t,ratio,total_n,a,b
  logical::first,all_empty
  integer::i,j,k
  first=self%n_features_in_==0;call prepare(self,x%n_cols,weighted)
  if(.not.self%with_std) then
   if(allocated(self%mean_)) deallocate(self%mean_)
   if(allocated(self%var_)) deallocate(self%var_)
   nan_weight=0;sum_weights=numpy_total(w)
   do i=1,x%n_rows
    do k=x%indptr(i),x%indptr(i+1)-1
     j=x%indices(k);t=ieee_product(w(i),merge(1._dp,0._dp,ieee_is_nan(x%data(k))))
     nan_weight(j)=add(nan_weight(j),t)
    end do
   end do
   do j=1,x%n_cols
    t=ieee_difference(sum_weights,nan_weight(j));if(.not.weighted) t=integer_cast(t)
    self%n_samples_seen_(j)=add(self%n_samples_seen_(j),t)
   end do
   call finish(self);return
  end if
  call sparse_batch(x,w,mean,var,n)
  if(first) then
   self%mean_=mean;self%var_=var;self%n_samples_seen_=n
  else
   call require(allocated(self%mean_).and.allocated(self%var_),'StandardScaler fitted sparse statistics are absent')
   all_empty=.true.
   do j=1,x%n_cols
    if(ieee_is_nan(self%n_samples_seen_(j))) cycle
    if(self%n_samples_seen_(j)>0) all_empty=.false.
   end do
   if(all_empty) then
    self%mean_=mean;self%var_=var;self%n_samples_seen_=n
   else
    do j=1,x%n_cols
     if(ieee_is_nan(n(j))) then
      ! Source new_n > 0 is false for NaN, preserving previous statistics.
      cycle
     end if
     if(n(j)<=0) cycle
     ratio=ieee_quotient(self%n_samples_seen_(j),n(j));total_n=add(self%n_samples_seen_(j),n(j))
     a=ieee_product(self%mean_(j),self%n_samples_seen_(j));b=ieee_product(mean(j),n(j))
     t=ieee_product(ieee_quotient(ratio,total_n),square(ieee_difference(ieee_quotient(a,ratio),b)))
     self%var_(j)=ieee_quotient(add(add(ieee_product(self%var_(j),self%n_samples_seen_(j)), &
      ieee_product(var(j),n(j))),t),total_n)
     self%mean_(j)=ieee_quotient(add(a,b),total_n);self%n_samples_seen_(j)=total_n
    end do
   end if
  end if
  self%sample_count_real_=.true.;call finish(self)
 end subroutine
 subroutine partial_dense(self,x,sample_weight)
  class(standard_scaler),intent(inout)::self
  real(dp),intent(in)::x(:,:)
  real(dp),intent(in),optional::sample_weight(:)
  call dense_input(x)
  if(present(sample_weight)) call weight_input(sample_weight,size(x,1))
  call dense_update(self,x,sample_weight)
 end subroutine
 subroutine partial_csr(self,x,sample_weight)
  class(standard_scaler),intent(inout)::self
  type(csr_matrix),intent(in)::x
  real(dp),intent(in),optional::sample_weight(:)
  real(dp)::w(x%n_rows)
  call sparse_input(x);call require(.not.self%with_mean,'StandardScaler cannot center sparse matrices')
  w=1
  if(present(sample_weight)) then
   call weight_input(sample_weight,x%n_rows);w=sample_weight
  end if
  call csr_update(self,x,w,present(sample_weight))
 end subroutine
 subroutine partial_dense_scalar(self,x,sample_weight)
  class(standard_scaler),intent(inout)::self
  real(dp),intent(in)::x(:,:),sample_weight
  real(dp)::w(size(x,1))
  call dense_input(x);w=sample_weight;call dense_update(self,x,w)
 end subroutine
 subroutine partial_csr_scalar(self,x,sample_weight)
  class(standard_scaler),intent(inout)::self
  type(csr_matrix),intent(in)::x
  real(dp),intent(in)::sample_weight
  real(dp)::w(x%n_rows)
  call sparse_input(x);call require(.not.self%with_mean,'StandardScaler cannot center sparse matrices')
  w=sample_weight
  ! Source sparse no-variance branch validates the expanded scalar again.
  if(.not.self%with_std) call weight_input(w,x%n_rows)
  call csr_update(self,x,w,.true.)
 end subroutine
 subroutine fit_dense(self,x,sample_weight)
  class(standard_scaler),intent(inout)::self
  real(dp),intent(in)::x(:,:)
  real(dp),intent(in),optional::sample_weight(:)
  call reset(self);call self%partial_fit(x,sample_weight)
 end subroutine
 subroutine fit_csr(self,x,sample_weight)
  class(standard_scaler),intent(inout)::self
  type(csr_matrix),intent(in)::x
  real(dp),intent(in),optional::sample_weight(:)
  call reset(self);call self%partial_fit(x,sample_weight)
 end subroutine
 subroutine fit_dense_scalar(self,x,sample_weight)
  class(standard_scaler),intent(inout)::self
  real(dp),intent(in)::x(:,:),sample_weight
  call reset(self);call self%partial_fit(x,sample_weight)
 end subroutine
 subroutine fit_csr_scalar(self,x,sample_weight)
  class(standard_scaler),intent(inout)::self
  type(csr_matrix),intent(in)::x
  real(dp),intent(in)::sample_weight
  call reset(self);call self%partial_fit(x,sample_weight)
 end subroutine
 subroutine ready(self)
  class(standard_scaler),intent(in)::self
  call require(self%n_features_in_>0,'standard_scaler is not fitted')
 end subroutine
 function transform_dense(self,x) result(z)
  class(standard_scaler),intent(in)::self
  real(dp),intent(in)::x(:,:)
  real(dp)::z(size(x,1),size(x,2)),v
  integer::i,j
  call ready(self);call dense_input(x)
  call require(size(x,2)==self%n_features_in_,'StandardScaler feature count mismatch')
  if(self%with_mean) call require(allocated(self%mean_),'StandardScaler fitted mean is absent')
  if(self%with_std) call require(allocated(self%scale_),'StandardScaler fitted scale is absent')
  do j=1,size(x,2)
   do i=1,size(x,1)
    v=x(i,j)
    if(self%with_mean) v=ieee_difference(v,self%mean_(j))
    if(self%with_std) v=ieee_quotient(v,self%scale_(j))
    z(i,j)=v
   end do
  end do
 end function
 function transform_csr(self,x) result(z)
  class(standard_scaler),intent(in)::self
  type(csr_matrix),intent(in)::x
  type(csr_matrix)::z
  integer::k
  call ready(self);call sparse_input(x)
  call require(.not.self%with_mean,'StandardScaler cannot center sparse matrices')
  call require(x%n_cols==self%n_features_in_,'StandardScaler feature count mismatch');z=x
  if(.not.allocated(self%scale_)) return
  do k=1,size(z%data)
   z%data(k)=ieee_product(z%data(k),ieee_quotient(1._dp,self%scale_(z%indices(k))))
  end do
 end function
 function inverse_dense(self,x) result(z)
  class(standard_scaler),intent(in)::self
  real(dp),intent(in)::x(:,:)
  real(dp)::z(size(x,1),size(x,2)),v
  integer::i,j,k
  call ready(self);call dense_input(x)
  if(self%with_mean) call require(allocated(self%mean_),'StandardScaler fitted mean is absent')
  if(self%with_std) call require(allocated(self%scale_),'StandardScaler fitted scale is absent')
  if(self%with_mean.or.self%with_std) &
   call require(size(x,2)==self%n_features_in_.or.self%n_features_in_==1,'StandardScaler inverse broadcast mismatch')
  do j=1,size(x,2)
   k=min(j,self%n_features_in_)
   do i=1,size(x,1)
    v=x(i,j)
    if(self%with_std) v=ieee_product(v,self%scale_(k))
    if(self%with_mean) v=add(v,self%mean_(k))
    z(i,j)=v
   end do
  end do
 end function
 function inverse_csr(self,x) result(z)
  class(standard_scaler),intent(in)::self
  type(csr_matrix),intent(in)::x
  type(csr_matrix)::z
  integer::k
  call ready(self);call sparse_input(x);call require(.not.self%with_mean,'StandardScaler cannot uncenter sparse matrices')
  z=x
  if(.not.allocated(self%scale_)) return
  call require(x%n_cols==size(self%scale_),'StandardScaler CSR inverse width mismatch')
  do k=1,size(z%data)
   z%data(k)=ieee_product(z%data(k),self%scale_(z%indices(k)))
  end do
 end function
 function fit_transform_dense(self,x,sample_weight) result(z)
  class(standard_scaler),intent(inout)::self
  real(dp),intent(in)::x(:,:)
  real(dp),intent(in),optional::sample_weight(:)
  real(dp)::z(size(x,1),size(x,2))
  call self%fit(x,sample_weight);z=self%transform(x)
 end function
 function fit_transform_csr(self,x,sample_weight) result(z)
  class(standard_scaler),intent(inout)::self
  type(csr_matrix),intent(in)::x
  real(dp),intent(in),optional::sample_weight(:)
  type(csr_matrix)::z
  call self%fit(x,sample_weight);z=self%transform(x)
 end function
 function fit_transform_dense_scalar(self,x,sample_weight) result(z)
  class(standard_scaler),intent(inout)::self
  real(dp),intent(in)::x(:,:),sample_weight
  real(dp)::z(size(x,1),size(x,2))
  call self%fit(x,sample_weight);z=self%transform(x)
 end function
 function fit_transform_csr_scalar(self,x,sample_weight) result(z)
  class(standard_scaler),intent(inout)::self
  type(csr_matrix),intent(in)::x
  real(dp),intent(in)::sample_weight
  type(csr_matrix)::z
  call self%fit(x,sample_weight);z=self%transform(x)
 end function
end module
