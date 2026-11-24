program weighted_kmeans_seeds
  use sklearn, only: dp,kmeans_plusplus,kmeans
  use sklearn_sparse, only: csr_matrix,csr_from_dense
  implicit none
  real(dp) :: x(6,2)
  real(dp), allocatable :: centers(:,:),sparse_centers(:,:)
  integer, allocatable :: ids(:),sparse_ids(:)
  type(csr_matrix) :: sparse
  type(kmeans) :: model
  x(:,1)=[1._dp,1._dp,1._dp,10._dp,10._dp,10._dp]
  x(:,2)=[2._dp,4._dp,0._dp,2._dp,4._dp,0._dp]
  call kmeans_plusplus(x,2,centers,ids,random_state=0)
  print *, 'Selected one-based rows:',ids
  sparse=csr_from_dense(x)
  call kmeans_plusplus(sparse,2,sparse_centers,sparse_ids,random_state=0)
  print *, 'Sparse seeds agree:',all(ids==sparse_ids).and.all(centers==sparse_centers)
  call kmeans_plusplus(x,2,centers,ids,sample_weight=[1._dp,1._dp,1._dp,3._dp,3._dp,3._dp], &
    random_state=7,n_local_trials=1)
  print *, 'Weighted one-based rows:',ids
  model%n_clusters=2;model%n_init=1;call model%fit(x,init_centers=centers)
  print *, 'KMeans inertia from these seeds:',model%inertia_
end program
