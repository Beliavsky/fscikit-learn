program reference_cosine_reference_frames_driver
 use sklearn_common,only:dp
 use sklearn_pairwise_chunks,only:pairwise_distance_iterator,distance_chunk,pairwise_distances_chunked
 implicit none
 type(pairwise_distance_iterator)::state
 type(distance_chunk)::chunk
 real(dp),allocatable::x(:,:),y(:,:)
 real(dp)::physical,reference
 integer::n,m,p,own,i,status
 logical::available
 do
  read(*,*,iostat=status)n,m,p,own,physical,reference
  if(status/=0)exit
  allocate(x(n,p),y(m,p))
  do i=1,n;read(*,*)x(i,:);end do
  if(own==0)then
   do i=1,m;read(*,*)y(i,:);end do
   state=pairwise_distances_chunked(x,y,metric='cosine',working_memory=physical)
  else
   state=pairwise_distances_chunked(x,metric='cosine',working_memory=physical)
  end if
  call state%set_reference_memory(reference)
  do
   available=state%next(chunk);if(.not.available)exit
   write(*,'(*(i0,1x))')chunk%distance%start,chunk%distance%n_rows
   write(*,'(*(es26.17e3,1x))')transpose(chunk%distance%values)
  end do
  write(*,'(a)')'END'
  deallocate(x,y)
 end do
end program
