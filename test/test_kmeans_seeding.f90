program test_kmeans_seeding
  use sklearn_cluster, only: kmeans_plusplus,kmeans
  use sklearn_common, only: dp,require
  use sklearn_sparse, only: csr_matrix,csr_from_dense
  implicit none
  real(dp) :: x(6,2),w(6)
  real(dp), allocatable :: centers(:,:),other(:,:)
  integer, allocatable :: ids(:),again(:)
  type(csr_matrix) :: sparse
  type(kmeans) :: model
  x(:,1)=[1._dp,1._dp,1._dp,10._dp,10._dp,10._dp]
  x(:,2)=[2._dp,4._dp,0._dp,2._dp,4._dp,0._dp]
  call kmeans_plusplus(x,2,centers,ids,random_state=0)
  call require(all(ids==[4,3]),'Public kmeans++ documented sklearn example')
  call require(all(centers==x(ids,:)),'Kmeans++ centers are selected input rows')
  model%n_clusters=2;model%n_init=1;call model%fit(x,init_centers=centers)
  call require(abs(model%inertia_-16)<1.e-12_dp,'Public seeds integrate with KMeans explicit initialization')
  sparse=csr_from_dense(x)
  call kmeans_plusplus(sparse,2,other,again,random_state=0)
  call require(all(ids==again).and.all(centers==other),'CSR initialization matches dense example')
  w=[0._dp,0._dp,0._dp,0._dp,0._dp,1._dp]
  call kmeans_plusplus(x,3,centers,ids,sample_weight=w,random_state=42,n_local_trials=1)
  call require(ids(1)==6,'First seed honors sample probabilities')
  call require(all(ids(2:)==1),'Zero potential retains source repeated-center behavior')
  call kmeans_plusplus(x,3,other,again,sample_weight=w,random_state=42,n_local_trials=1)
  call require(all(ids==again).and.all(centers==other),'Local RNG produces reproducible seeds')
  x(:,1)=[0._dp,1._dp,2._dp,3._dp,4._dp,5._dp];x(:,2)=0
  call kmeans_plusplus(x,3,centers,ids,sample_weight=-1._dp+[0._dp,0._dp,0._dp,0._dp,0._dp,0._dp],random_state=0)
  call require(all(ids==[4,6,6]),'All-negative source weights and batched cumulative search')
  print *, 'Public kmeans++ dense/CSR, weighted seeds, local trials and degenerate potential: PASS'
end program
