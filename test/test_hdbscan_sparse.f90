program test_hdbscan_sparse
  use sklearn_hdbscan, only: hdbscan
  use sklearn_sparse, only: csr_matrix,csr_from_dense,csr_from_arrays
  use sklearn_common, only: dp,require
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf,ieee_is_nan
  use, intrinsic :: ieee_exceptions, only: ieee_get_flag,ieee_set_flag,ieee_invalid
  implicit none
  type(hdbscan) :: model,dense,copy
  type(csr_matrix) :: graph,before
  real(dp) :: positions(12),distances(12,12)
  integer, allocatable :: labels(:),cuts(:)
  integer :: i,j,n,e,total
  logical :: invalid
  character(32) :: mode
  call get_command_argument(1,mode)
  if(mode=='memory') then
    n=12000; graph%n_rows=n; graph%n_cols=n
    total=5*n-6
    allocate(graph%indptr(n+1),graph%indices(total),graph%data(total)); e=1
    do i=1,n
      graph%indptr(i)=e
      do j=max(1,i-2),min(n,i+2)
        graph%indices(e)=j; graph%data(e)=abs(i-j)*(1+real(min(i,j),dp)*1.e-6_dp); e=e+1
      end do
    end do
    graph%indptr(n+1)=e
    model%metric='precomputed'; model%min_samples=2; model%min_cluster_size=5; model%allow_single_cluster=.true.
    labels=model%fit_predict(graph); cuts=model%dbscan_clustering(3._dp,5)
    call require(size(labels)==n.and.all(cuts==0),'large sparse HDBSCAN complete DBSCAN cut')
    call require(all(model%distances_>0).and..not.model%has_infinite_edges_,'large sparse HDBSCAN finite hierarchy')
    call require(model%cluster_sizes_(n-1)==n,'large sparse HDBSCAN root size')
    print *, 'dense_distance_bytes',int(n,int64)*n*8
    print *, 'Large sparse HDBSCAN memory case: PASS'
    stop
  end if
  positions=[-5._dp,-4.9_dp,-4.8_dp,-4.7_dp,-4.6_dp,5._dp,5.1_dp,5.2_dp,5.3_dp,5.4_dp,-20._dp,20._dp]
  do i=1,12
    distances(i,:)=abs(positions(i)-positions)
  end do
  graph=csr_from_dense(distances,store_zeros=.true.); before=graph
  model%metric='precomputed'; model%min_samples=3; model%min_cluster_size=3
  select case(mode)
  case('asymmetric'); graph%data(2)=graph%data(2)+.1_dp
  case('pattern'); graph=csr_from_arrays(3,3,[1,3,4,6],[1,2,2,1,3],[0._dp,1._dp,0._dp,1._dp,0._dp])
  case('few_neighbors','few_neighbors_max')
    graph=csr_from_arrays(3,3,[1,2,3,4],[1,2,3],[1._dp,1._dp,1._dp])
    if(mode=='few_neighbors_max') model%max_distance=10
  case('disconnected')
    graph=csr_from_arrays(4,4,[1,3,5,7,9],[1,2,1,2,3,4,3,4],[0._dp,1._dp,1._dp,0._dp,0._dp,1._dp,1._dp,0._dp])
    model%min_samples=2
  case('nan'); graph%data(1)=ieee_value(0._dp,ieee_quiet_nan)
  case('infinity'); graph%data(1)=ieee_value(0._dp,ieee_positive_inf)
  case('negative'); graph%data=-1
  case('zero_mst'); graph%data=0; model%min_samples=1
  case('single'); graph=csr_from_arrays(1,1,[1,2],[1],[0._dp])
  case('shape')
    graph=csr_from_arrays(2,3,[1,3,5],[1,2,1,2],[0._dp,1._dp,1._dp,0._dp]); model%min_samples=1
  case('min_samples'); model%min_samples=13
  case('centers'); model%store_centers='both'
  case('bad_pointer'); graph%indptr(1)=0
  case('bad_index'); graph%indices(1)=13
  end select
  labels=model%fit_predict(graph)
  if(len_trim(mode)>0) error stop 'Expected guard did not reject input'
  call require(all(graph%data==before%data).and.all(graph%indices==before%indices).and. &
    all(graph%indptr==before%indptr),'CSR HDBSCAN preserves caller storage')
  call require(model%n_clusters_==2.and.all(labels(:5)==labels(1)).and.all(labels(6:10)==labels(6)), &
    'CSR HDBSCAN separated clusters')
  call require(labels(1)>=0.and.labels(6)>=0.and.labels(1)/=labels(6),'CSR HDBSCAN distinct clusters')
  call require(all(labels(11:)==-1),'CSR HDBSCAN noise')
  dense=model; call dense%fit(distances)
  call require(maxval(abs(model%core_distances_-dense%core_distances_))<1.e-14_dp,'CSR explicit diagonal core distances')
  call require(maxval(abs(model%probabilities_-dense%probabilities_))<1.e-13_dp,'CSR/dense HDBSCAN membership strengths')
  copy=model; call model%fit(graph)
  call require(all(model%children_==copy%children_).and.all(model%labels_==copy%labels_),'CSR HDBSCAN deterministic refit')
  cuts=model%dbscan_clustering(.15_dp,3)
  call require(count(cuts>=0)==6,'CSR HDBSCAN DBSCAN cut')
  ! A copied CSR model can be refitted to dense features and back to distances.
  copy%metric='euclidean'; copy%store_centers='both'; call copy%fit(reshape(positions,[12,1]))
  copy%metric='precomputed'; copy%store_centers='none'; call copy%fit(graph)
  call require(.not.allocated(copy%centroids_).and..not.allocated(copy%medoids_),'CSR HDBSCAN clears dense centers')
  model%min_samples=12; model%alpha=1.e-310_dp; model%max_distance=10; model%allow_single_cluster=.true.
  call ieee_set_flag(ieee_invalid,.false.); call model%fit(graph); call ieee_get_flag(ieee_invalid,invalid)
  call require(.not.invalid.and.all(ieee_is_nan(model%core_distances_)), &
    'CSR reciprocal-overflow core NaNs and caller IEEE flags')
  call require(all(model%distances_==10),'CSR HDBSCAN max_distance fallback')
  print *, 'Sparse HDBSCAN tests: PASS'
end program
