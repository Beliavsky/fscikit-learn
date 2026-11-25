module chunks_invalid_support
 use sklearn_common,only:dp
 use sklearn_pairwise_chunks
 implicit none
 type,extends(pairwise_chunk_reducer)::bad_reducer
  integer::mode=0
 contains
  procedure::apply=>bad_apply
 end type
contains
 subroutine bad_apply(self,block,reduced)
  class(bad_reducer),intent(inout)::self
  type(distance_block),intent(inout)::block
  type(distance_reduction),intent(out)::reduced
  select case(self%mode)
  case(1);allocate(reduced%real_vector(block%n_rows+1))
  case(2);allocate(reduced%tuple(1))
  case(3);allocate(reduced%tuple(1));allocate(reduced%tuple(1)%tuple(0))
  case(4);allocate(reduced%real_vector(block%n_rows),reduced%integer_vector(block%n_rows))
  end select
 end subroutine
end module
program reference_pairwise_chunks_invalid_driver
 use sklearn_common,only:dp
 use sklearn_sparse,only:csr_matrix,csr_from_dense
 use sklearn_pairwise_chunks
 use chunks_invalid_support
 use,intrinsic::ieee_arithmetic
 implicit none
 real(dp)::x(3,2),y(2,2),nan,inf
 type(csr_matrix)::s
 type(pairwise_distance_iterator)::it
 type(distance_chunk)::chunk
 type(bad_reducer)::reducer
 logical::ok
 character(32)::mode
 x=1;y=2;nan=ieee_value(0._dp,ieee_quiet_nan);inf=ieee_value(0._dp,ieee_positive_inf)
 call get_command_argument(1,mode)
 select case(trim(mode))
 case('memory_negative');it=pairwise_distances_chunked(x,working_memory=-1._dp)
 case('memory_nan');it=pairwise_distances_chunked(x,working_memory=nan)
 case('memory_inf');it=pairwise_distances_chunked(x,working_memory=inf)
 case('memory_precomputed_inf');it=pairwise_distances_chunked(x,x(:2,:),metric='precomputed',working_memory=inf)
 case('metric');it=pairwise_distances_chunked(x,metric='invalid')
 case('features');it=pairwise_distances_chunked(x,y(:,:1))
 case('empty');it=pairwise_distances_chunked(x(:0,:))
 case('precomputed_negative');x(1,1)=-1;it=pairwise_distances_chunked(x,y,metric='precomputed')
 case('precomputed_shape');it=pairwise_distances_chunked(x,metric='precomputed')
 case('sparse_metric');s=csr_from_dense(x);it=pairwise_distances_chunked(s,metric='dice')
 case('variance_missing');it=pairwise_distances_chunked(x,y,metric='seuclidean')
 case('precision_missing');it=pairwise_distances_chunked(x,y,metric='mahalanobis')
 case('variance_shape');it=pairwise_distances_chunked(x,metric='seuclidean',v=[1._dp])
 case('precision_auto_scalar');it=pairwise_distances_chunked(x(:,:1),metric='mahalanobis')
 case('weights_shape');it=pairwise_distances_chunked(x,metric='minkowski',w=[1._dp])
 case('weights_negative');it=pairwise_distances_chunked(x,metric='minkowski',w=[1._dp,-1._dp])
 case('reduction_rows');reducer%mode=1;it=pairwise_distances_chunked(x);ok=it%next(chunk,reducer)
 case('reduction_none_member');reducer%mode=2;it=pairwise_distances_chunked(x);ok=it%next(chunk,reducer)
 case('reduction_nested');reducer%mode=3;it=pairwise_distances_chunked(x);ok=it%next(chunk,reducer)
 case('reduction_multiple');reducer%mode=4;it=pairwise_distances_chunked(x);ok=it%next(chunk,reducer)
 case default;error stop 'unknown chunks input guard'
 end select
 error stop 'invalid chunked distance input accepted'
end program
