program pairwise_chunks_example
 use sklearn_common,only:dp
 use sklearn_metrics,only:pairwise_distances_chunked,pairwise_distance_iterator,distance_chunk
 implicit none
 type(pairwise_distance_iterator)::stream
 type(distance_chunk)::chunk
 real(dp)::x(3,2)
 x(:,1)=[0._dp,1._dp,2._dp];x(:,2)=[1._dp,2._dp,0._dp]
 stream=pairwise_distances_chunked(x,working_memory=0._dp)
 do while(stream%next(chunk))
  print *,'Zero-based start:',chunk%distance%start
  print *,'Distances:',chunk%distance%values
 end do
end program
