! SMACOF: sklearn 1.6.1 manifold/_mds.py (BSD-3-Clause, retained notices).
! NumPy summation and RNG adapters retain their existing BSD notices.
module sklearn_smacof
 use sklearn_common,only:dp,require,check_x
 use sklearn_legacy_rng,only:legacy_random_state
 use sklearn_isotonic,only:isotonic_regression
 use sklearn_pairwise_chunks,only:pairwise_distances_chunked,pairwise_distance_iterator,distance_chunk
 use sklearn_kernel_dispatch,only:kernel_blas_dot
 use,intrinsic::ieee_arithmetic
 use,intrinsic::ieee_exceptions
 implicit none
 private
 public::smacof,smacof_result
 type::smacof_result
  real(dp),allocatable::embedding(:,:)
  real(dp)::stress=0
  integer::n_iter=0,n_init_effective=0
  logical::converged=.false.,normalized_stress=.false.
 end type
 interface smacof
  module procedure smacof_text,smacof_boolean
 end interface
contains
 recursive real(dp)function numpy_sum(a)result(v)
  real(dp),intent(in)::a(:)
  real(dp)::lane(8)
  integer::n,i,j,whole,split
  n=size(a);v=-0._dp
  if(n<8)then
   do i=1,n
    v=v+a(i)
   end do
  else if(n<=128)then
   lane=a(:8);whole=n-mod(n,8)
   do i=9,whole,8
    do j=1,8
     lane(j)=lane(j)+a(i+j-1)
    end do
   end do
   v=((lane(1)+lane(2))+(lane(3)+lane(4)))+((lane(5)+lane(6))+(lane(7)+lane(8)))
   do i=whole+1,n
    v=v+a(i)
   end do
  else
   split=n/2;split=split-mod(split,8);v=numpy_sum(a(:split))+numpy_sum(a(split+1:))
  end if
 end function
 real(dp)function matrix_sum(a)result(v)
  real(dp),intent(in)::a(:,:)
  v=numpy_sum(reshape(transpose(a),[size(a)]))
 end function
 function smacof_boolean(dissimilarities,normalized_stress,n_components,metric,init,n_init,max_iter,eps,random_state)result(fit)
  real(dp),intent(in)::dissimilarities(:,:)
  logical,intent(in)::normalized_stress
  integer,optional,intent(in)::n_components,n_init,max_iter,random_state
  logical,optional,intent(in)::metric
  real(dp),optional,intent(in)::init(:,:),eps
  type(smacof_result)::fit
  character(5)::normalized
  normalized='false';if(normalized_stress)normalized='true'
  fit=smacof_text(dissimilarities,n_components,metric,init,n_init,max_iter,eps,random_state,trim(normalized))
 end function
 function smacof_text(dissimilarities,n_components,metric,init,n_init,max_iter,eps,random_state, &
  normalized_stress)result(fit)
  real(dp),intent(in)::dissimilarities(:,:)
  integer,optional,intent(in)::n_components,n_init,max_iter,random_state
  logical,optional,intent(in)::metric
  real(dp),optional,intent(in)::init(:,:),eps
  character(*),optional,intent(in)::normalized_stress
  type(smacof_result)::fit
  type(legacy_random_state)::rng
  type(smacof_result)::candidate
  logical::modes(size(ieee_all)),flags(size(ieee_all)),metric_
  integer::n,k,runs,limit,seed,run,i,j
  real(dp)::tolerance
  character(:),allocatable::normalized
  call ieee_get_halting_mode(ieee_all,modes);call ieee_get_flag(ieee_all,flags)
  call ieee_set_halting_mode(ieee_all,.false.)
  call check_x(dissimilarities);n=size(dissimilarities,1)
  call require(size(dissimilarities,2)==n,'SMACOF dissimilarities must be square')
  call require(all(abs(dissimilarities-transpose(dissimilarities))<= &
   1.e-10_dp+1.e-5_dp*abs(transpose(dissimilarities))),'SMACOF dissimilarities must be symmetric')
  k=2;if(present(n_components))k=n_components
  runs=8;if(present(n_init))runs=n_init
  limit=300;if(present(max_iter))limit=max_iter
  seed=0;if(present(random_state))seed=random_state
  metric_=.true.;if(present(metric))metric_=metric
  tolerance=1.e-3_dp;if(present(eps))tolerance=eps
  normalized='auto';if(present(normalized_stress))normalized=normalized_stress
  call require(k>=1,'SMACOF n_components must be positive')
  call require(runs>=1,'SMACOF n_init must be positive')
  call require(limit>=1,'SMACOF max_iter must be positive')
  call require(seed>=0,'SMACOF random_state must be nonnegative')
  call require(.not.ieee_is_nan(tolerance).and.tolerance>=0,'SMACOF eps must be nonnegative')
  call require(any(normalized==[character(5)::'auto','true','false']),'invalid SMACOF normalized_stress')
  fit%normalized_stress=normalized=='true'.or.(normalized=='auto'.and..not.metric_)
  call require(.not.(fit%normalized_stress.and.metric_),'normalized stress requires nonmetric SMACOF')
  if(present(init))then
   call check_x(init);call require(size(init,1)==n,'SMACOF initial sample count differs')
   k=size(init,2);runs=1
  end if
  call rng%seed(seed)
  do run=1,runs
   allocate(candidate%embedding(n,k))
   if(present(init))then
    candidate%embedding=init
   else
    do i=1,n
     do j=1,k
      candidate%embedding(i,j)=rng%uniform()
     end do
    end do
   end if
   call single_run(dissimilarities,metric_,limit,tolerance,fit%normalized_stress,candidate)
   if(run==1)then
    fit%embedding=candidate%embedding;fit%stress=candidate%stress
    fit%n_iter=candidate%n_iter;fit%converged=candidate%converged
   else if(candidate%stress<fit%stress)then
    fit%embedding=candidate%embedding;fit%stress=candidate%stress
    fit%n_iter=candidate%n_iter;fit%converged=candidate%converged
   end if
   deallocate(candidate%embedding)
  end do
  fit%n_init_effective=runs
  call ieee_set_flag(ieee_all,flags);call ieee_set_halting_mode(ieee_all,modes)
 end function
 subroutine single_run(dissimilarities,metric,limit,tolerance,normalized,fit)
  real(dp),intent(in)::dissimilarities(:,:),tolerance
  logical,intent(in)::metric,normalized
  integer,intent(in)::limit
  type(smacof_result),intent(inout)::fit
  type(isotonic_regression)::monotone
  type(pairwise_distance_iterator)::stream
  type(distance_chunk)::chunk
  real(dp),allocatable::distance(:,:),disparity(:,:),ratio(:,:),b(:,:),next(:,:),similarity(:),observed(:),predicted(:)
  real(dp)::old_stress,scaled,stress,denominator,norm
  real(dp)::row_norm(size(fit%embedding,1))
  integer::n,k,i,j,it,count_,q
  logical::ok
  n=size(fit%embedding,1);k=size(fit%embedding,2)
  count_=0
  do i=1,n
   do j=i+1,n
    if(dissimilarities(i,j)/=0)count_=count_+1
   end do
  end do
  if(.not.metric)then
   call require(count_>0,'nonmetric SMACOF requires nonzero upper-triangle dissimilarities')
   allocate(similarity(count_),observed(count_));q=0
   do i=1,n
    do j=i+1,n
     if(dissimilarities(i,j)==0)cycle
     q=q+1;similarity(q)=dissimilarities(i,j)
    end do
   end do
  end if
  allocate(next(n,k));old_stress=0;fit%converged=.false.
  do it=1,limit
   stream=pairwise_distances_chunked(fit%embedding,working_memory=8._dp*real(n,dp)**2/1048576+1)
   ok=stream%next(chunk);call require(ok,'SMACOF distance calculation produced no block')
   distance=chunk%distance%values
   if(metric)then
    disparity=dissimilarities
   else
    q=0
    do i=1,n
     do j=i+1,n
      if(dissimilarities(i,j)==0)cycle
      q=q+1;observed(q)=distance(i,j)
     end do
    end do
    predicted=monotone%fit_transform(similarity,observed)
    disparity=distance;q=0
    do i=1,n
     do j=i+1,n
      if(dissimilarities(i,j)==0)cycle
      q=q+1;disparity(i,j)=predicted(q)
     end do
    end do
    denominator=matrix_sum(disparity**2)
    disparity=disparity*sqrt((real(n,dp)*(n-1)/2)/denominator)
   end if
   stress=matrix_sum((distance-disparity)**2)/2
   if(normalized)stress=sqrt(stress/(matrix_sum(disparity**2)/2))
   where(distance==0)distance=1.e-5_dp
   ratio=disparity/distance;b=-ratio
   do i=1,n
    b(i,i)=b(i,i)+numpy_sum(ratio(i,:))
   end do
   do j=1,k
    do i=1,n
     next(i,j)=(1._dp/n)*kernel_blas_dot(b(i,:),fit%embedding(:,j),n,k,i,j)
    end do
   end do
   fit%embedding=next
   do i=1,n
    row_norm(i)=sqrt(numpy_sum(fit%embedding(i,:)**2))
   end do
   norm=numpy_sum(row_norm);scaled=stress/norm
   if(it>1)then
    if(old_stress-scaled<tolerance)then
     fit%converged=.true.;exit
    end if
   end if
   old_stress=scaled
  end do
  fit%stress=stress;fit%n_iter=min(it,limit)
 end subroutine
end module
