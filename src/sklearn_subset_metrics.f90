! sklearn 1.6.1 accuracy_score / zero_one_loss, numeric target contracts (BSD).
! SciPy 1.15.3 CSR row accumulation and NumPy 2.2.6 pairwise reductions (BSD).
! Masked-dot order follows OpenBLAS 0.3.29 SkylakeX ddot (2014 OpenBLAS Project,
! BSD-3-Clause, LICENSES/OpenBLAS-LICENSE); scalar Fortran, no SIMD/BLAS runtime.
! Copyright (c) 2014, The OpenBLAS Project. All rights reserved.
! Assumed-rank adapters, owned CSR and quiet IEEE integration are MIT.
module sklearn_subset_metrics
 use sklearn_common,only:dp,require
 use sklearn_sparse,only:csr_matrix,csr_from_dense
 use sklearn_ieee_math,only:sub=>ieee_difference,mul=>ieee_product,div=>ieee_quotient
 use,intrinsic::iso_fortran_env,only:int64
 use,intrinsic::iso_c_binding,only:c_loc,c_intptr_t
 use,intrinsic::ieee_arithmetic,only:ieee_is_finite,ieee_is_nan
 implicit none
 private
 public::accuracy_score,zero_one_loss
 ! Explicit typed ranks preserve predictor-result and empty-expression calls.
 interface accuracy_score
  module procedure accuracy_iv_iv
  module procedure accuracy_iv_im
  module procedure accuracy_iv_rv
  module procedure accuracy_iv_rm
  module procedure accuracy_iv_lv
  module procedure accuracy_iv_lm
  module procedure accuracy_iv_s
  module procedure accuracy_im_iv
  module procedure accuracy_im_im
  module procedure accuracy_im_rv
  module procedure accuracy_im_rm
  module procedure accuracy_im_lv
  module procedure accuracy_im_lm
  module procedure accuracy_im_s
  module procedure accuracy_rv_iv
  module procedure accuracy_rv_im
  module procedure accuracy_rv_rv
  module procedure accuracy_rv_rm
  module procedure accuracy_rv_lv
  module procedure accuracy_rv_lm
  module procedure accuracy_rv_s
  module procedure accuracy_rm_iv
  module procedure accuracy_rm_im
  module procedure accuracy_rm_rv
  module procedure accuracy_rm_rm
  module procedure accuracy_rm_lv
  module procedure accuracy_rm_lm
  module procedure accuracy_rm_s
  module procedure accuracy_lv_iv
  module procedure accuracy_lv_im
  module procedure accuracy_lv_rv
  module procedure accuracy_lv_rm
  module procedure accuracy_lv_lv
  module procedure accuracy_lv_lm
  module procedure accuracy_lv_s
  module procedure accuracy_lm_iv
  module procedure accuracy_lm_im
  module procedure accuracy_lm_rv
  module procedure accuracy_lm_rm
  module procedure accuracy_lm_lv
  module procedure accuracy_lm_lm
  module procedure accuracy_lm_s
  module procedure accuracy_s_iv
  module procedure accuracy_s_im
  module procedure accuracy_s_rv
  module procedure accuracy_s_rm
  module procedure accuracy_s_lv
  module procedure accuracy_s_lm
  module procedure accuracy_s_s
 end interface
 interface zero_one_loss
  module procedure zero_one_iv_iv
  module procedure zero_one_iv_im
  module procedure zero_one_iv_rv
  module procedure zero_one_iv_rm
  module procedure zero_one_iv_lv
  module procedure zero_one_iv_lm
  module procedure zero_one_iv_s
  module procedure zero_one_im_iv
  module procedure zero_one_im_im
  module procedure zero_one_im_rv
  module procedure zero_one_im_rm
  module procedure zero_one_im_lv
  module procedure zero_one_im_lm
  module procedure zero_one_im_s
  module procedure zero_one_rv_iv
  module procedure zero_one_rv_im
  module procedure zero_one_rv_rv
  module procedure zero_one_rv_rm
  module procedure zero_one_rv_lv
  module procedure zero_one_rv_lm
  module procedure zero_one_rv_s
  module procedure zero_one_rm_iv
  module procedure zero_one_rm_im
  module procedure zero_one_rm_rv
  module procedure zero_one_rm_rm
  module procedure zero_one_rm_lv
  module procedure zero_one_rm_lm
  module procedure zero_one_rm_s
  module procedure zero_one_lv_iv
  module procedure zero_one_lv_im
  module procedure zero_one_lv_rv
  module procedure zero_one_lv_rm
  module procedure zero_one_lv_lv
  module procedure zero_one_lv_lm
  module procedure zero_one_lv_s
  module procedure zero_one_lm_iv
  module procedure zero_one_lm_im
  module procedure zero_one_lm_rv
  module procedure zero_one_lm_rm
  module procedure zero_one_lm_lv
  module procedure zero_one_lm_lm
  module procedure zero_one_lm_s
  module procedure zero_one_s_iv
  module procedure zero_one_s_im
  module procedure zero_one_s_rv
  module procedure zero_one_s_rm
  module procedure zero_one_s_lv
  module procedure zero_one_s_lm
  module procedure zero_one_s_s
 end interface
 type::target_data
  logical::vector=.false.
  real(dp),allocatable::values(:)
  type(csr_matrix)::matrix
 end type
contains
 logical function integral(x) result(valid)
  real(dp),intent(in)::x
  valid=.false.
  if(.not.ieee_is_finite(x)) return
  if(x< -2._dp**63.or.x>=2._dp**63) return
  valid=x==real(int(x,int64),dp)
 end function
 subroutine numbers(x)
  real(dp),intent(in)::x(:)
  integer::i
  call require(all(ieee_is_finite(x)),'subset metric targets must be finite')
  do i=1,size(x)
   call require(integral(x(i)),'subset metrics require integral classification targets')
  end do
 end subroutine
 subroutine vector_target(t,x)
  type(target_data),intent(out)::t
  real(dp),intent(in)::x(:)
  call numbers(x);t%vector=.true.;t%values=x
 end subroutine
 subroutine matrix_target(t,x)
  type(target_data),intent(out)::t
  real(dp),intent(in)::x(:,:)
  real(dp)::first,second
  logical::found
  integer::i,j
  if(size(x,2)==1) then
   call require(size(x,1)>0,'subset metrics reject an empty column target')
   call vector_target(t,x(:,1));return
  end if
  call require(size(x,2)>=2,'subset metrics reject zero-width targets')
  call numbers(reshape(x,[size(x)]));first=0;second=0;found=.false.
  if(size(x)>0) first=x(1,1)
  do j=1,size(x,2)
   do i=1,size(x,1)
    if(x(i,j)==first) cycle
    if(.not.found) then
     second=x(i,j);found=.true.
    else
     call require(x(i,j)==second,'subset metrics reject multioutput targets')
    end if
   end do
  end do
  t%matrix=csr_from_dense(x)
 end subroutine
 subroutine sparse_target(t,x)
  type(target_data),intent(out)::t
  type(csr_matrix),intent(in)::x
  real(dp)::first,second
  logical::found
  integer::i
  call x%validate();call require(x%n_cols>=2,'subset metrics require CSR multilabel indicators')
  call numbers(x%data);first=0;second=0;found=.false.
  if(size(x%data)>0) first=x%data(1)
  do i=1,size(x%data)
   if(x%data(i)==first) cycle
   if(.not.found) then
    second=x%data(i);found=.true.
   else
    call require(x%data(i)==second,'subset metrics require CSR indicator values')
   end if
  end do
  if(found) call require(first==0.or.second==0,'subset metrics require CSR indicator values')
  t%matrix=x
 end subroutine
 function target_i(x) result(t)
  integer,intent(in)::x(..)
  type(target_data)::t
  select rank(x)
  rank(1)
   call vector_target(t,real(x,dp))
  rank(2)
   call matrix_target(t,real(x,dp))
  rank default
   call require(.false.,'subset metrics require vector, matrix or CSR targets')
  end select
 end function
 function target_r(x) result(t)
  real(dp),intent(in)::x(..)
  type(target_data)::t
  select rank(x)
  rank(1)
   call vector_target(t,x)
  rank(2)
   call matrix_target(t,x)
  rank default
   call require(.false.,'subset metrics require vector, matrix or CSR targets')
  end select
 end function
 function target_l(x) result(t)
  logical,intent(in)::x(..)
  type(target_data)::t
  select rank(x)
  rank(1)
   call vector_target(t,merge(1._dp,0._dp,x))
  rank(2)
   call matrix_target(t,merge(1._dp,0._dp,x))
  rank default
   call require(.false.,'subset metrics require vector, matrix or CSR targets')
  end select
 end function
 function target_c(x) result(t)
  type(csr_matrix),intent(in)::x
  type(target_data)::t
  call sparse_target(t,x)
 end function
 logical function canonical(x) result(valid)
  type(csr_matrix),intent(in)::x
  integer::i,k
  valid=.false.
  do i=1,x%n_rows
   do k=x%indptr(i)+1,x%indptr(i+1)-1
    if(x%indices(k)<=x%indices(k-1)) return
   end do
  end do
  valid=.true.
 end function
 subroutine canonical_matches(y,p,score)
  type(csr_matrix),intent(in)::y,p
  real(dp),intent(out)::score(:)
  integer::i,a,b,ae,be
  score=1
  do i=1,size(score)
   a=y%indptr(i);ae=y%indptr(i+1);b=p%indptr(i);be=p%indptr(i+1)
   do while(a<ae.and.b<be)
    if(y%indices(a)==p%indices(b)) then
     if(y%data(a)/=p%data(b)) score(i)=0
     a=a+1;b=b+1
    else if(y%indices(a)<p%indices(b)) then
     if(y%data(a)/=0) score(i)=0
     a=a+1
    else
     if(p%data(b)/=0) score(i)=0
     b=b+1
    end if
   end do
   if(a<ae) then
    if(any(y%data(a:ae-1)/=0)) score(i)=0
   end if
   if(b<be) then
    if(any(p%data(b:be-1)/=0)) score(i)=0
   end if
  end do
 end subroutine
 function matches(y,p) result(score)
  type(target_data),intent(in)::y,p
  real(dp),allocatable::score(:),left(:),right(:)
  integer,allocatable::next(:)
  integer::i,k,col,head,prior
  call require(y%vector.eqv.p%vector,'subset metrics reject mixed vector and multilabel targets')
  if(y%vector) then
   call require(size(y%values)==size(p%values),'invalid label sizes')
   score=merge(1._dp,0._dp,y%values==p%values);return
  end if
  call require(y%matrix%n_rows==p%matrix%n_rows,'invalid label sizes')
  call require(y%matrix%n_cols==p%matrix%n_cols,'subset metric indicator shapes differ')
  allocate(score(y%matrix%n_rows))
  if(canonical(y%matrix).and.canonical(p%matrix)) then
   call canonical_matches(y%matrix,p%matrix,score);return
  end if
  allocate(left(y%matrix%n_cols),right(y%matrix%n_cols),next(y%matrix%n_cols))
  score=1;left=0;right=0;next=-1
  do i=1,size(score)
   head=0
   do k=y%matrix%indptr(i),y%matrix%indptr(i+1)-1
    col=y%matrix%indices(k);left(col)=sub(left(col),-y%matrix%data(k))
    if(next(col)==-1) then
     next(col)=head;head=col
    end if
   end do
   do k=p%matrix%indptr(i),p%matrix%indptr(i+1)-1
    col=p%matrix%indices(k);right(col)=sub(right(col),-p%matrix%data(k))
    if(next(col)==-1) then
     next(col)=head;head=col
    end if
   end do
   do while(head>0)
    if(left(head)/=right(head)) score(i)=0
    left(head)=0;right(head)=0;prior=next(head);next(head)=-1;head=prior
   end do
  end do
 end function
 recursive real(dp) function pairwise(a) result(total)
  real(dp),intent(in)::a(:)
  real(dp)::r(8)
  integer::n,i,j,whole,split
  n=size(a);total=-0._dp
  if(n<8) then
   do i=1,n
    total=sub(total,-a(i))
   end do
  else if(n<=128) then
   r=a(:8);whole=n-mod(n,8)
   do i=9,whole,8
    do j=1,8
     r(j)=sub(r(j),-a(i+j-1))
    end do
   end do
   total=sub(sub(sub(r(1),-r(2)),-sub(r(3),-r(4))),-sub(sub(r(5),-r(6)),-sub(r(7),-r(8))))
   do i=whole+1,n
    total=sub(total,-a(i))
   end do
  else
   split=n/2;split=split-mod(split,8);total=sub(pairwise(a(:split)),-pairwise(a(split+1:)))
  end if
 end function
 real(dp) function summed(a) result(value)
  real(dp),intent(in)::a(:)
  integer::first,last
  value=0
  do first=1,size(a),8192
   last=min(size(a),first+8191);value=sub(value,-pairwise(a(first:last)))
  end do
 end function
 logical function positive_stride(weights) result(nonunit)
  real(dp),target,intent(in)::weights(:)
  integer(c_intptr_t)::first,second
  nonunit=.false.
  if(size(weights)<2.or.is_contiguous(weights)) return
  first=transfer(c_loc(weights(1)),first);second=transfer(c_loc(weights(2)),second)
  nonunit=second>first
 end function
 real(dp) function masked_dot(products,nonunit) result(value)
  real(dp),intent(in)::products(:)
  logical,intent(in)::nonunit
  real(dp)::wide(8,4),narrow(4,4),v(4),first,second
  integer::n,n16,n32,n4,i,j,k
  ! Scalar reconstruction of OpenBLAS 0.3.29 serial SkylakeX ddot lanes.
  n=size(products);n16=n-mod(n,16);n32=n16-mod(n16,32);wide=0;narrow=0;value=0
  ! NumPy's one-element dot fast path returns the product without adding zero.
  if(n==1) then
   value=products(1);return
  end if
  if(nonunit) then
   first=0;second=0;n4=n-mod(n,4)
   do i=1,n4,4
    first=sub(first,-sub(products(i),-products(i+2)))
    second=sub(second,-sub(products(i+1),-products(i+3)))
   end do
   do i=n4+1,n
    first=sub(first,-products(i))
   end do
   value=sub(first,-second);return
  end if
  if(n16>0) then
   do i=1,n32,32
    do j=1,4
     do k=1,8
      wide(k,j)=sub(wide(k,j),-products(i+(j-1)*8+k-1))
     end do
    end do
   end do
   do j=1,4
    do k=1,4
     narrow(k,j)=sub(wide(k,j),-wide(k+4,j))
    end do
   end do
   if(n16>n32) then
    do j=1,4
     do k=1,4
      narrow(k,j)=sub(narrow(k,j),-products(n32+(j-1)*4+k))
     end do
    end do
   end if
   do k=1,4
    v(k)=sub(sub(sub(narrow(k,1),-narrow(k,2)),-narrow(k,3)),-narrow(k,4))
   end do
   value=sub(sub(v(1),-v(3)),-sub(v(2),-v(4)))
  end if
  do i=n16+1,n
   value=sub(value,-products(i))
  end do
 end function
 real(dp) function evaluate(y,p,normalize,sample_weight,loss) result(value)
  type(target_data),intent(in)::y,p
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  logical,intent(in)::loss
  real(dp),allocatable::score(:),products(:)
  real(dp)::scale
  logical::norm
  integer::i
  score=matches(y,p);norm=.true.;if(present(normalize)) norm=normalize
  scale=real(size(score),dp)
  if(present(sample_weight)) then
   call require(size(sample_weight)==size(score),'subset metric sample weight length mismatch')
   allocate(products(size(score)))
   do i=1,size(score)
    products(i)=mul(score(i),sample_weight(i))
   end do
   scale=summed(sample_weight)
   if(norm) then
    if(.not.ieee_is_nan(scale)) call require(scale/=0,'subset metric weights sum to zero')
    value=div(summed(products),scale)
   else
    value=masked_dot(products,positive_stride(sample_weight))
   end if
  else
   value=summed(score)
   if(norm) value=div(value,scale)
  end if
  if(loss) then
   if(norm) then
    value=sub(1._dp,value)
   else
    value=sub(scale,value)
   end if
  end if
 end function
 real(dp) function accuracy_iv_iv(y,p,normalize,sample_weight) result(value)
  integer,intent(in)::y(:)
  integer,intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_i(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_iv_im(y,p,normalize,sample_weight) result(value)
  integer,intent(in)::y(:)
  integer,intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_i(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_iv_rv(y,p,normalize,sample_weight) result(value)
  integer,intent(in)::y(:)
  real(dp),intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_r(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_iv_rm(y,p,normalize,sample_weight) result(value)
  integer,intent(in)::y(:)
  real(dp),intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_r(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_iv_lv(y,p,normalize,sample_weight) result(value)
  integer,intent(in)::y(:)
  logical,intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_l(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_iv_lm(y,p,normalize,sample_weight) result(value)
  integer,intent(in)::y(:)
  logical,intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_l(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_iv_s(y,p,normalize,sample_weight) result(value)
  integer,intent(in)::y(:)
  type(csr_matrix),intent(in)::p
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_c(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_im_iv(y,p,normalize,sample_weight) result(value)
  integer,intent(in)::y(:,:)
  integer,intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_i(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_im_im(y,p,normalize,sample_weight) result(value)
  integer,intent(in)::y(:,:)
  integer,intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_i(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_im_rv(y,p,normalize,sample_weight) result(value)
  integer,intent(in)::y(:,:)
  real(dp),intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_r(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_im_rm(y,p,normalize,sample_weight) result(value)
  integer,intent(in)::y(:,:)
  real(dp),intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_r(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_im_lv(y,p,normalize,sample_weight) result(value)
  integer,intent(in)::y(:,:)
  logical,intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_l(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_im_lm(y,p,normalize,sample_weight) result(value)
  integer,intent(in)::y(:,:)
  logical,intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_l(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_im_s(y,p,normalize,sample_weight) result(value)
  integer,intent(in)::y(:,:)
  type(csr_matrix),intent(in)::p
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_c(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_rv_iv(y,p,normalize,sample_weight) result(value)
  real(dp),intent(in)::y(:)
  integer,intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_i(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_rv_im(y,p,normalize,sample_weight) result(value)
  real(dp),intent(in)::y(:)
  integer,intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_i(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_rv_rv(y,p,normalize,sample_weight) result(value)
  real(dp),intent(in)::y(:)
  real(dp),intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_r(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_rv_rm(y,p,normalize,sample_weight) result(value)
  real(dp),intent(in)::y(:)
  real(dp),intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_r(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_rv_lv(y,p,normalize,sample_weight) result(value)
  real(dp),intent(in)::y(:)
  logical,intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_l(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_rv_lm(y,p,normalize,sample_weight) result(value)
  real(dp),intent(in)::y(:)
  logical,intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_l(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_rv_s(y,p,normalize,sample_weight) result(value)
  real(dp),intent(in)::y(:)
  type(csr_matrix),intent(in)::p
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_c(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_rm_iv(y,p,normalize,sample_weight) result(value)
  real(dp),intent(in)::y(:,:)
  integer,intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_i(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_rm_im(y,p,normalize,sample_weight) result(value)
  real(dp),intent(in)::y(:,:)
  integer,intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_i(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_rm_rv(y,p,normalize,sample_weight) result(value)
  real(dp),intent(in)::y(:,:)
  real(dp),intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_r(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_rm_rm(y,p,normalize,sample_weight) result(value)
  real(dp),intent(in)::y(:,:)
  real(dp),intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_r(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_rm_lv(y,p,normalize,sample_weight) result(value)
  real(dp),intent(in)::y(:,:)
  logical,intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_l(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_rm_lm(y,p,normalize,sample_weight) result(value)
  real(dp),intent(in)::y(:,:)
  logical,intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_l(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_rm_s(y,p,normalize,sample_weight) result(value)
  real(dp),intent(in)::y(:,:)
  type(csr_matrix),intent(in)::p
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_c(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_lv_iv(y,p,normalize,sample_weight) result(value)
  logical,intent(in)::y(:)
  integer,intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_i(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_lv_im(y,p,normalize,sample_weight) result(value)
  logical,intent(in)::y(:)
  integer,intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_i(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_lv_rv(y,p,normalize,sample_weight) result(value)
  logical,intent(in)::y(:)
  real(dp),intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_r(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_lv_rm(y,p,normalize,sample_weight) result(value)
  logical,intent(in)::y(:)
  real(dp),intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_r(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_lv_lv(y,p,normalize,sample_weight) result(value)
  logical,intent(in)::y(:)
  logical,intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_l(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_lv_lm(y,p,normalize,sample_weight) result(value)
  logical,intent(in)::y(:)
  logical,intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_l(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_lv_s(y,p,normalize,sample_weight) result(value)
  logical,intent(in)::y(:)
  type(csr_matrix),intent(in)::p
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_c(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_lm_iv(y,p,normalize,sample_weight) result(value)
  logical,intent(in)::y(:,:)
  integer,intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_i(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_lm_im(y,p,normalize,sample_weight) result(value)
  logical,intent(in)::y(:,:)
  integer,intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_i(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_lm_rv(y,p,normalize,sample_weight) result(value)
  logical,intent(in)::y(:,:)
  real(dp),intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_r(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_lm_rm(y,p,normalize,sample_weight) result(value)
  logical,intent(in)::y(:,:)
  real(dp),intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_r(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_lm_lv(y,p,normalize,sample_weight) result(value)
  logical,intent(in)::y(:,:)
  logical,intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_l(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_lm_lm(y,p,normalize,sample_weight) result(value)
  logical,intent(in)::y(:,:)
  logical,intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_l(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_lm_s(y,p,normalize,sample_weight) result(value)
  logical,intent(in)::y(:,:)
  type(csr_matrix),intent(in)::p
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_c(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_s_iv(y,p,normalize,sample_weight) result(value)
  type(csr_matrix),intent(in)::y
  integer,intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_c(y),target_i(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_s_im(y,p,normalize,sample_weight) result(value)
  type(csr_matrix),intent(in)::y
  integer,intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_c(y),target_i(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_s_rv(y,p,normalize,sample_weight) result(value)
  type(csr_matrix),intent(in)::y
  real(dp),intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_c(y),target_r(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_s_rm(y,p,normalize,sample_weight) result(value)
  type(csr_matrix),intent(in)::y
  real(dp),intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_c(y),target_r(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_s_lv(y,p,normalize,sample_weight) result(value)
  type(csr_matrix),intent(in)::y
  logical,intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_c(y),target_l(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_s_lm(y,p,normalize,sample_weight) result(value)
  type(csr_matrix),intent(in)::y
  logical,intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_c(y),target_l(p),normalize,sample_weight,.false.)
 end function
 real(dp) function accuracy_s_s(y,p,normalize,sample_weight) result(value)
  type(csr_matrix),intent(in)::y
  type(csr_matrix),intent(in)::p
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_c(y),target_c(p),normalize,sample_weight,.false.)
 end function
 real(dp) function zero_one_iv_iv(y,p,normalize,sample_weight) result(value)
  integer,intent(in)::y(:)
  integer,intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_i(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_iv_im(y,p,normalize,sample_weight) result(value)
  integer,intent(in)::y(:)
  integer,intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_i(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_iv_rv(y,p,normalize,sample_weight) result(value)
  integer,intent(in)::y(:)
  real(dp),intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_r(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_iv_rm(y,p,normalize,sample_weight) result(value)
  integer,intent(in)::y(:)
  real(dp),intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_r(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_iv_lv(y,p,normalize,sample_weight) result(value)
  integer,intent(in)::y(:)
  logical,intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_l(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_iv_lm(y,p,normalize,sample_weight) result(value)
  integer,intent(in)::y(:)
  logical,intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_l(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_iv_s(y,p,normalize,sample_weight) result(value)
  integer,intent(in)::y(:)
  type(csr_matrix),intent(in)::p
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_c(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_im_iv(y,p,normalize,sample_weight) result(value)
  integer,intent(in)::y(:,:)
  integer,intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_i(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_im_im(y,p,normalize,sample_weight) result(value)
  integer,intent(in)::y(:,:)
  integer,intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_i(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_im_rv(y,p,normalize,sample_weight) result(value)
  integer,intent(in)::y(:,:)
  real(dp),intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_r(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_im_rm(y,p,normalize,sample_weight) result(value)
  integer,intent(in)::y(:,:)
  real(dp),intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_r(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_im_lv(y,p,normalize,sample_weight) result(value)
  integer,intent(in)::y(:,:)
  logical,intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_l(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_im_lm(y,p,normalize,sample_weight) result(value)
  integer,intent(in)::y(:,:)
  logical,intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_l(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_im_s(y,p,normalize,sample_weight) result(value)
  integer,intent(in)::y(:,:)
  type(csr_matrix),intent(in)::p
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_c(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_rv_iv(y,p,normalize,sample_weight) result(value)
  real(dp),intent(in)::y(:)
  integer,intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_i(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_rv_im(y,p,normalize,sample_weight) result(value)
  real(dp),intent(in)::y(:)
  integer,intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_i(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_rv_rv(y,p,normalize,sample_weight) result(value)
  real(dp),intent(in)::y(:)
  real(dp),intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_r(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_rv_rm(y,p,normalize,sample_weight) result(value)
  real(dp),intent(in)::y(:)
  real(dp),intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_r(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_rv_lv(y,p,normalize,sample_weight) result(value)
  real(dp),intent(in)::y(:)
  logical,intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_l(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_rv_lm(y,p,normalize,sample_weight) result(value)
  real(dp),intent(in)::y(:)
  logical,intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_l(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_rv_s(y,p,normalize,sample_weight) result(value)
  real(dp),intent(in)::y(:)
  type(csr_matrix),intent(in)::p
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_c(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_rm_iv(y,p,normalize,sample_weight) result(value)
  real(dp),intent(in)::y(:,:)
  integer,intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_i(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_rm_im(y,p,normalize,sample_weight) result(value)
  real(dp),intent(in)::y(:,:)
  integer,intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_i(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_rm_rv(y,p,normalize,sample_weight) result(value)
  real(dp),intent(in)::y(:,:)
  real(dp),intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_r(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_rm_rm(y,p,normalize,sample_weight) result(value)
  real(dp),intent(in)::y(:,:)
  real(dp),intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_r(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_rm_lv(y,p,normalize,sample_weight) result(value)
  real(dp),intent(in)::y(:,:)
  logical,intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_l(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_rm_lm(y,p,normalize,sample_weight) result(value)
  real(dp),intent(in)::y(:,:)
  logical,intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_l(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_rm_s(y,p,normalize,sample_weight) result(value)
  real(dp),intent(in)::y(:,:)
  type(csr_matrix),intent(in)::p
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_c(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_lv_iv(y,p,normalize,sample_weight) result(value)
  logical,intent(in)::y(:)
  integer,intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_i(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_lv_im(y,p,normalize,sample_weight) result(value)
  logical,intent(in)::y(:)
  integer,intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_i(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_lv_rv(y,p,normalize,sample_weight) result(value)
  logical,intent(in)::y(:)
  real(dp),intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_r(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_lv_rm(y,p,normalize,sample_weight) result(value)
  logical,intent(in)::y(:)
  real(dp),intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_r(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_lv_lv(y,p,normalize,sample_weight) result(value)
  logical,intent(in)::y(:)
  logical,intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_l(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_lv_lm(y,p,normalize,sample_weight) result(value)
  logical,intent(in)::y(:)
  logical,intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_l(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_lv_s(y,p,normalize,sample_weight) result(value)
  logical,intent(in)::y(:)
  type(csr_matrix),intent(in)::p
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_c(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_lm_iv(y,p,normalize,sample_weight) result(value)
  logical,intent(in)::y(:,:)
  integer,intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_i(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_lm_im(y,p,normalize,sample_weight) result(value)
  logical,intent(in)::y(:,:)
  integer,intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_i(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_lm_rv(y,p,normalize,sample_weight) result(value)
  logical,intent(in)::y(:,:)
  real(dp),intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_r(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_lm_rm(y,p,normalize,sample_weight) result(value)
  logical,intent(in)::y(:,:)
  real(dp),intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_r(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_lm_lv(y,p,normalize,sample_weight) result(value)
  logical,intent(in)::y(:,:)
  logical,intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_l(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_lm_lm(y,p,normalize,sample_weight) result(value)
  logical,intent(in)::y(:,:)
  logical,intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_l(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_lm_s(y,p,normalize,sample_weight) result(value)
  logical,intent(in)::y(:,:)
  type(csr_matrix),intent(in)::p
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_c(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_s_iv(y,p,normalize,sample_weight) result(value)
  type(csr_matrix),intent(in)::y
  integer,intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_c(y),target_i(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_s_im(y,p,normalize,sample_weight) result(value)
  type(csr_matrix),intent(in)::y
  integer,intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_c(y),target_i(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_s_rv(y,p,normalize,sample_weight) result(value)
  type(csr_matrix),intent(in)::y
  real(dp),intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_c(y),target_r(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_s_rm(y,p,normalize,sample_weight) result(value)
  type(csr_matrix),intent(in)::y
  real(dp),intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_c(y),target_r(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_s_lv(y,p,normalize,sample_weight) result(value)
  type(csr_matrix),intent(in)::y
  logical,intent(in)::p(:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_c(y),target_l(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_s_lm(y,p,normalize,sample_weight) result(value)
  type(csr_matrix),intent(in)::y
  logical,intent(in)::p(:,:)
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_c(y),target_l(p),normalize,sample_weight,.true.)
 end function
 real(dp) function zero_one_s_s(y,p,normalize,sample_weight) result(value)
  type(csr_matrix),intent(in)::y
  type(csr_matrix),intent(in)::p
  logical,optional,intent(in)::normalize
  real(dp),optional,target,intent(in)::sample_weight(:)
  value=evaluate(target_c(y),target_c(p),normalize,sample_weight,.true.)
 end function
end module
