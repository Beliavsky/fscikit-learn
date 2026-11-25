program test_pairwise_chunks
 use sklearn_common,only:dp,require
 use sklearn_pairwise_chunks
 use sklearn_sparse,only:csr_matrix,csr_from_arrays
 use,intrinsic::ieee_exceptions
 implicit none
 type(pairwise_distance_iterator)::it
 type(distance_chunk)::chunk
 type(csr_matrix)::s
 real(dp)::x(3,1),expected(3,3)
 logical::flags(size(ieee_all)),after(size(ieee_all)),ok
 integer::i
 x(:,1)=[0._dp,3._dp,6._dp]
 expected(1,:)=[0._dp,3._dp,6._dp];expected(2,:)=[3._dp,0._dp,3._dp];expected(3,:)=[6._dp,3._dp,0._dp]
 it=pairwise_distances_chunked(x,metric='manhattan',working_memory=0._dp)
 call require(it%chunk_rows()==1.and.it%exceeds_working_memory(),'memory budget minimum-one-row contract')
 do i=1,3
  call ieee_get_flag(ieee_all,flags);ok=it%next(chunk);call ieee_get_flag(ieee_all,after)
  call require(ok.and.all(flags.eqv.after),'iterator preserves IEEE flags')
  call require(chunk%distance%start==i-1.and.all(chunk%distance%values(1,:)==expected(i,:)), &
   'vertical row chunks use zero-based starts')
 end do
 call require(.not.it%next(chunk),'iterator exhaustion')
 it=pairwise_distances_chunked(expected,metric='precomputed',working_memory=0._dp)
 call require(it%next(chunk),'precomputed block exists')
 call require(size(chunk%distance%values,1)==3.and.all(chunk%distance%values==expected),'precomputed ignores memory limit')
 s=csr_from_arrays(50000,1,[(i,i=1,50001)],[(1,i=1,50000)],[(1._dp,i=1,50000)])
 it=pairwise_distances_chunked(s,working_memory=0._dp)
 call require(it%next(chunk),'large sparse first chunk exists')
 call require(size(chunk%distance%values,1)==1.and.size(chunk%distance%values,2)==50000, &
  'streamed output avoids a 20 GB full distance allocation')
 call require(all(chunk%distance%values==0),'identical sparse rows have zero distance')
 print *,'Chunk boundaries, precomputed path, IEEE state and sparse 20 GB avoided allocation passed'
end program
