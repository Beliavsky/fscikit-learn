program nearest_pairs
 use sklearn,only:dp,pairwise_distances_argmin,pairwise_distances_argmin_min
 implicit none
 real(dp)::x(2,2),y(2,2)
 integer,allocatable::indices(:)
 real(dp),allocatable::distances(:)
 x(1,:)=[0._dp,0._dp];x(2,:)=[1._dp,1._dp]
 y(1,:)=[1._dp,0._dp];y(2,:)=[1._dp,1._dp]
 call pairwise_distances_argmin_min(x,y,indices,distances)
 print *,'Nearest row indices (1-based): ',indices
 print *,'Nearest distances: ',distances
 indices=pairwise_distances_argmin(x,y,metric='manhattan',axis=0)
 print *,'Nearest X rows for Y (Manhattan): ',indices
end program
