program test_extra_dissimilarities
 use sklearn,only:dp,pairwise_distances_argmin,pairwise_distances_argmin_min
 use sklearn_common,only:require
 use sklearn_sparse,only:csr_matrix,csr_from_dense
 use,intrinsic::ieee_arithmetic,only:ieee_is_nan
 implicit none
 real(dp)::x(1,3),y(1,3),want(12),z(2,2)
 real(dp),allocatable::d(:)
 integer,allocatable::i(:),j(:)
 character(16)::names(12)
 type(csr_matrix)::sx,sy
 integer::k
 names=[character(16)::'braycurtis','canberra','correlation','hamming','dice','jaccard', &
  'rogerstanimoto','russellrao','sokalmichener','sokalsneath','yule','matching']
 x(1,:)=[1._dp,0._dp,1._dp];y(1,:)=[1._dp,1._dp,0._dp]
 want=[.5_dp,2._dp,1.5_dp,2._dp/3,.5_dp,2._dp/3,.8_dp,2._dp/3,.8_dp,.8_dp,2._dp,2._dp/3]
 do k=1,size(names)
  call pairwise_distances_argmin_min(x,y,i,d,metric=trim(names(k)))
  j=pairwise_distances_argmin(x,y,metric=trim(names(k)))
  call require(all(i==1).and.all(j==i),'additional dissimilarity index and API agreement')
  call require(abs(d(1)-want(k))<2.e-14_dp,'closed-form additional dissimilarity')
 end do
 x(1,:)=[-2._dp,0._dp,3._dp];y(1,:)=[1._dp,0._dp,-1._dp]
 call pairwise_distances_argmin_min(x,y,i,d,metric='matching')
 call require(d(1)==0,'matching compares nonzero truth values')
 call pairwise_distances_argmin_min(x,y,i,d,metric='hamming')
 call require(abs(d(1)-2._dp/3)<1.e-15_dp,'Hamming compares numeric coordinates')
 call pairwise_distances_argmin_min(x,y,i,d,metric='braycurtis')
 call require(d(1)==1,'optimized Bray-Curtis denominator sums absolute operands')
 x=0;y=0
 call pairwise_distances_argmin_min(x,y,i,d,metric='dice')
 call require(ieee_is_nan(d(1)),'Dice all-false pair is NaN')
 call pairwise_distances_argmin_min(x,y,i,d,metric='sokalsneath')
 call require(ieee_is_nan(d(1)),'Sokal-Sneath all-false pair is NaN in cdist')
 call pairwise_distances_argmin_min(x,y,i,d,metric='jaccard')
 call require(d(1)==0,'Jaccard all-false pair is zero')
 call pairwise_distances_argmin_min(x,y,i,d,metric='russellrao')
 call require(d(1)==1,'Russell-Rao all-false pair is one')
 call pairwise_distances_argmin_min(x,y,i,d,metric='yule')
 call require(d(1)==0,'Yule degenerate product is zero')
 x=1;y=2
 call pairwise_distances_argmin_min(x,y,i,d,metric='correlation')
 call require(ieee_is_nan(d(1)),'correlation constant rows have undefined distance')
 z(1,:)=[0._dp,0._dp];z(2,:)=[1._dp,1._dp]
 sx=csr_from_dense(z);sy=csr_from_dense(2*z)
 call pairwise_distances_argmin_min(sx,sy,i,d,metric='canberra')
 call require(i(1)==1.and.d(1)==0,'CSR implicit zeros do not contribute to Canberra')
 call pairwise_distances_argmin_min(sx,2*z,i,d,metric='canberra')
 call require(i(1)==1.and.d(1)==2,'mixed Canberra retains dense stored-zero positions')
 call pairwise_distances_argmin_min(sx,sy,i,d,metric='braycurtis')
 call require(i(1)==1.and.d(1)==0,'CSR Bray-Curtis uses the documented dense numerical contract')
 call pairwise_distances_argmin_min(sx,sy,i,d,metric='matching')
 call require(all(i==[1,2]).and.all(d==0),'CSR matching uses Boolean coordinates')
 print *,'Additional numeric and Boolean nearest-pair dissimilarities: PASS'
end program
