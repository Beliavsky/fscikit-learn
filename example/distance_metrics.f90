program distance_metrics
 use sklearn,only:dp,distance_metric,get_metric
 implicit none
 type(distance_metric)::metric
 real(dp)::x(2,2)
 real(dp),allocatable::distances(:,:)
 x(1,:)=[0._dp,0._dp];x(2,:)=[3._dp,4._dp]
 metric=get_metric('minkowski',p=2._dp,w=[1._dp,2._dp])
 distances=metric%pairwise(x)
 print *,'Weighted Minkowski distance:',distances(1,2)
 metric=get_metric('seuclidean',v=[1._dp,4._dp])
 distances=metric%pairwise(x)
 print *,'Standardized Euclidean distance:',distances(1,2)
end program
