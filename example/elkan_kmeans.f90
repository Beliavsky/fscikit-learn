program elkan_kmeans
  use sklearn,only:dp,kmeans,k_means,kmeans_result
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  implicit none
  type(kmeans)::model
  type(kmeans_result)::result
  type(csr_matrix)::sparse
  real(dp)::x(4,2),weights(4)
  x(:,1)=[0._dp,2._dp,10._dp,12._dp];x(:,2)=[0._dp,0._dp,1._dp,1._dp]
  weights=[1._dp,3._dp,1._dp,1._dp];sparse=csr_from_dense(x)
  model%n_clusters=2;model%n_init=0;model%random_state=7;model%algorithm='elkan'
  call model%fit(sparse,sample_weight=weights)
  result=k_means(x,2,sample_weight=weights,random_state=7,algorithm='elkan')
  print *,'Sparse Elkan inertia:',model%inertia_
  print *,'Dense Elkan labels:',result%labels
end program
