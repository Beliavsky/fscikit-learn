program test_pairwise_reductions
 use sklearn,only:dp,pairwise_distances_argmin,pairwise_distances_argmin_min
 use sklearn_common,only:require
 use sklearn_sparse,only:csr_matrix,csr_from_dense
 use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan,ieee_is_nan
 implicit none
 real(dp),allocatable::x(:,:),y(:,:),d(:)
 integer,allocatable::i(:),j(:)
 type(csr_matrix)::sx,sy
 x=reshape([0._dp,1._dp,0._dp,1._dp],[2,2])
 y=reshape([1._dp,1._dp,0._dp,1._dp],[2,2])
 call pairwise_distances_argmin_min(x,y,i,d)
 call require(all(i==[1,2]).and.all(d==[1._dp,0._dp]),'nearest rows and Euclidean distances')
 j=pairwise_distances_argmin(x,y)
 call require(all(i==j),'argmin and argmin_min agree')
 call pairwise_distances_argmin_min(x,y,i,d,axis=0)
 call require(all(i==[1,2]).and.all(d==[1._dp,0._dp]),'axis zero swaps queries and candidates')
 sx=csr_from_dense(x);sy=csr_from_dense(y)
 call pairwise_distances_argmin_min(sx,sy,i,d,metric='manhattan')
 call require(all(i==[1,2]).and.all(d==[1._dp,0._dp]),'CSR nearest Manhattan')
 call pairwise_distances_argmin_min(sx,y,i,d,metric='minkowski',p=.5_dp)
 call require(all(i==[1,2]).and.all(d==[1._dp,0._dp]),'mixed CSR fractional Minkowski')
 call pairwise_distances_argmin_min(x,sy,i,d,metric='chebyshev')
 call require(all(i==[1,2]).and.all(d==[1._dp,0._dp]),'mixed CSR Chebyshev')
 call pairwise_distances_argmin_min(x,y,i,d,metric='cosine')
 call require(i(1)==1.and.d(1)==1.and.i(2)==2,'zero row cosine and first equal-distance candidate')
 call pairwise_distances_argmin_min(x,y,i,d,metric=asymmetric,axis=0)
 call require(all(i==[1,2]).and.all(d==[1.125_dp,.25_dp]),'callback axis zero argument orientation')
 call pairwise_distances_argmin_min(sx,sy,i,d,metric=asymmetric,axis=0)
 call require(all(i==[1,2]).and.all(d==[1.125_dp,.25_dp]),'CSR row callback orientation')
 call pairwise_distances_argmin_min(x,x,i,d,metric=asymmetric)
 call require(all(i==[1,2]).and.all(d==[0._dp,.25_dp]),'self callback diagonal')
 ! Asymmetric self callbacks are evaluated on the upper triangle and mirrored.
 call pairwise_distances_argmin_min(x,x,i,d,metric=decreasing)
 call require(all(i==[2,1]).and.all(d==[-2._dp,-2._dp]),'self callback upper-triangle symmetry')
 x=reshape([1e8_dp,1e8_dp],[1,2]);y=reshape([1e8_dp+1,1e8_dp],[1,2])
 call pairwise_distances_argmin_min(x,y,i,d)
 call require(i(1)==1.and.d(1)==0,'ordered Gram cancellation')
 call pairwise_distances_argmin_min(x,y,i,d,metric='l2')
 call require(d(1)==1,'coordinate l2 alias path')
 x=reshape([1e200_dp,2e200_dp],[1,2]);y=2*x
 call pairwise_distances_argmin_min(x,y,i,d)
 call require(d(1)==0,'optimized Gram NaN clips to zero')
 call pairwise_distances_argmin_min(x,y,i,d,metric='l2')
 call require(d(1)==sqrt(huge(1._dp)),'overflow retains reduced-distance heap sentinel')
 x=reshape([ieee_value(0._dp,ieee_quiet_nan),1._dp],[1,2]);y=reshape([2._dp,3._dp],[1,2])
 call pairwise_distances_argmin_min(x,y,i,d,metric='nan_euclidean',squared=.true.)
 call require(d(1)==8,'missing coordinates rescale squared distance')
 y=reshape([2._dp,ieee_value(0._dp,ieee_quiet_nan)],[1,2])
 call pairwise_distances_argmin_min(x,y,i,d,metric='nan_euclidean')
 call require(ieee_is_nan(d(1)).and.i(1)==1,'no overlapping coordinates returns NaN')
 print *,'Streaming nearest-pair reductions: PASS'
contains
 real(dp)function asymmetric(a,b)result(value)
  real(dp),intent(in)::a(:),b(:)
  value=sum(abs(a-b))+.125_dp*sum(a)
 end function
 real(dp)function decreasing(a,b)result(value)
  real(dp),intent(in)::a(:),b(:)
  value=-sum(abs(a-b))+sum(a)
 end function
end program
