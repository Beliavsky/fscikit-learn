program reference_neighbor_selection_driver
 use sklearn_common,only:dp
 use sklearn_neighbor_selection,only:argkmin_order,pairwise_knn_order
 implicit none
 real(dp),allocatable::values(:),selected(:)
 integer,allocatable::indices(:)
 integer::n,k,queries,kind,status
 do
  read(*,*,iostat=status)n,k,queries,kind
  if(status/=0)exit
  allocate(values(n),selected(k));read(*,*)values
  if(kind==0)then
   indices=argkmin_order(values,k,queries,selected)
  else
   indices=pairwise_knn_order(values,k)
   selected=values(indices)
  end if
  write(*,'(*(i0,1x))')indices
  write(*,'(*(es26.17e3,1x))')selected
  deallocate(values,selected)
 end do
end program
