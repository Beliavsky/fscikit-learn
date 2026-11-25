program test_parameter_distances
 use sklearn,only:dp,pairwise_distances_argmin,pairwise_distances_argmin_min
 use sklearn_common,only:require
 use sklearn_sparse,only:csr_matrix,csr_from_dense
 use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan,ieee_is_nan
 implicit none
 real(dp),allocatable::x(:,:),y(:,:),d(:),absent(:),absent_vi(:,:)
 integer,allocatable::i(:),j(:)
 type(csr_matrix)::sx,sy
 real(dp)::precision(2,2)
 x=reshape([0._dp,0._dp],[1,2]);y=reshape([2._dp,0._dp,0._dp,1._dp],[2,2])
 call pairwise_distances_argmin_min(x,y,i,d,metric='minkowski',w=[.1_dp,10._dp])
 call require(i(1)==1.and.abs(d(1)-sqrt(.4_dp))<1.e-15_dp,'weighted Minkowski chooses the weighted nearest row')
 j=pairwise_distances_argmin(x,y,metric='minkowski',w=[.1_dp,10._dp])
 call require(all(i==j),'weighted public APIs agree')
 sx=csr_from_dense(x);sy=csr_from_dense(y)
 ! A zero-stored CSR is rejected by the source optimized backend; use a stored row.
 x=reshape([1._dp,1._dp],[1,2]);sx=csr_from_dense(x)
 call pairwise_distances_argmin_min(sx,sy,i,d,metric='minkowski',p=3._dp,w=[1._dp,2._dp])
 call require(i(1)==2.and.d(1)==1,'CSR weighted cubic Minkowski')
 call pairwise_distances_argmin_min(sx,y,i,d,metric='seuclidean',v=[100._dp,1._dp])
 call require(i(1)==2.and.d(1)==.1_dp,'mixed standardized Euclidean')
 x=reshape([0._dp,0._dp],[1,2]);y=reshape([1._dp,2._dp],[1,2])
 precision(1,:)=[1._dp,2._dp];precision(2,:)=[3._dp,4._dp]
 call pairwise_distances_argmin_min(x,y,i,d,metric='mahalanobis',vi=precision)
 call require(abs(d(1)-sqrt(27._dp))<1.e-14_dp,'nonsymmetric Mahalanobis quadratic form')
 y=reshape([1._dp,2._dp,0._dp,0._dp],[2,2])
 call pairwise_distances_argmin_min(x,y,i,d,metric='seuclidean',v=[-1._dp,1._dp])
 call require(i(1)==2.and.d(1)==0,'negative variances select reduced distance before clipping')
 call pairwise_distances_argmin_min(x,y,i,d,metric='seuclidean',v=[ieee_value(0._dp,ieee_quiet_nan),1._dp])
 call require(i(1)==2.and.ieee_is_nan(d(1)),'NaN variance preserves source heap ordering')
 y=reshape([1._dp,0._dp],[1,2])
 call pairwise_distances_argmin_min(x,y,i,d,metric='seuclidean',v=[0._dp,1._dp])
 call require(d(1)==sqrt(huge(1._dp)),'zero variance retains the optimized heap sentinel')
 x=reshape([0._dp,0._dp],[1,2])
 call pairwise_distances_argmin_min(x,x,i,d,metric='correlation')
 call require(d(1)==0,'aliased one-row correlation uses the source fallback diagonal')
 call pairwise_distances_argmin_min(x,x,i,d,metric='dice')
 call require(d(1)==0,'aliased all-false Dice diagonal is zero')
 call pairwise_distances_argmin_min(x,x,i,d,metric='sokalsneath')
 call require(d(1)==0,'aliased all-false Sokal-Sneath diagonal is zero')
 x=reshape([0._dp,0._dp,0._dp,0._dp,0._dp,0._dp],[3,2])
 call pairwise_distances_argmin_min(x,x,i,d,metric='russellrao')
 call require(all(i==[1,2,3]).and.all(d==0),'aliased Russell-Rao chooses each forced diagonal')
 call pairwise_distances_argmin_min(x,x,i,d,metric='correlation')
 call require(all(i==[2,1,1]).and.all(ieee_is_nan(d)),'off-diagonal NaN overrides a forced zero diagonal')
 x=reshape([1._dp,0._dp,-1._dp,0._dp,0._dp,1._dp,0._dp,-1._dp],[4,2])
 call pairwise_distances_argmin_min(x,x,i,d,metric='mahalanobis',vi=absent_vi)
 call require(all(i==[1,2,3,4]).and.all(d==0),'automatic self covariance and absent optional precision')
 x=reshape([1._dp,1._dp],[1,2])
 call pairwise_distances_argmin_min(x,x,i,d,metric=asymmetric)
 call require(d(1)==.25_dp,'self callbacks retain their nonzero diagonal')
 call pairwise_distances_argmin_min(x,x,i,d,metric='p',w=absent)
 call require(d(1)==0,'Minkowski p alias and absent optional weights')
 print *,'Parameterized nearest-pair distances and aliased input semantics: PASS'
contains
 real(dp)function asymmetric(a,b)result(value)
  real(dp),intent(in)::a(:),b(:)
  value=sum(abs(a-b))+.125_dp*sum(a)
 end function
end program
