program parameter_distances
 use sklearn,only:dp,pairwise_distances_argmin_min
 implicit none
 real(dp)::x(1,2),y(2,2),precision(2,2)
 integer,allocatable::indices(:)
 real(dp),allocatable::distances(:)
 x=0;y(1,:)=[2._dp,0._dp];y(2,:)=[0._dp,1._dp]
 call pairwise_distances_argmin_min(x,y,indices,distances,metric='minkowski',w=[.1_dp,10._dp])
 print *,'Weighted Minkowski: ',indices,distances
 call pairwise_distances_argmin_min(x,y,indices,distances,metric='seuclidean',v=[100._dp,1._dp])
 print *,'Standardized Euclidean: ',indices,distances
 precision=0;precision(1,1)=1;precision(2,2)=4
 call pairwise_distances_argmin_min(x,y,indices,distances,metric='mahalanobis',vi=precision)
 print *,'Mahalanobis: ',indices,distances
end program
