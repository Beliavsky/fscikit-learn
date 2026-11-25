program extra_dissimilarities
 use sklearn,only:dp,pairwise_distances_argmin_min
 implicit none
 real(dp)::x(1,3),y(1,3)
 integer,allocatable::indices(:)
 real(dp),allocatable::distances(:)
 character(16)::names(4)
 integer::k
 x(1,:)=[1._dp,0._dp,1._dp];y(1,:)=[1._dp,1._dp,0._dp]
 names=[character(16)::'canberra','correlation','jaccard','dice']
 do k=1,size(names)
  call pairwise_distances_argmin_min(x,y,indices,distances,metric=trim(names(k)))
  print *,trim(names(k)),': ',indices,distances
 end do
end program
