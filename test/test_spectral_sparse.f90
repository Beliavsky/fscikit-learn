program test_spectral_sparse
  use sklearn_common,only:dp,require
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  use sklearn_spectral,only:spectral_embedding
  use sklearn_cluster,only:spectral_clustering
  use sklearn_metrics,only:pairwise_distances,adjusted_rand_score
  implicit none
  type(spectral_embedding)::dense,sparse
  type(spectral_clustering)::cluster
  type(csr_matrix)::s
  real(dp)::x(12,3)
  real(dp),allocatable::d(:,:),z(:,:),a(:,:)
  integer,allocatable::labels(:),other(:)
  integer::i
  character(20)::arg
  call get_command_argument(1,arg)
  do i=1,12
    x(i,:)=[real(i,dp),sin(real(i,dp)),cos(.3_dp*i)]
  end do
  dense%affinity='rbf';dense%gamma=.4_dp;sparse=dense;s=csr_from_dense(x)
  call dense%fit(x);call sparse%fit(s)
  call require(maxval(abs(dense%affinity_matrix_-sparse%affinity_matrix_))<1.e-12_dp,'spectral CSR RBF affinity')
  sparse%affinity='nearest_neighbors';z=sparse%fit_transform(s)
  call require(sparse%affinity=='rbf','SpectralEmbedding sparse nearest-neighbor affinity falls back to RBF')
  call require(maxval(abs(sparse%affinity_matrix_-dense%affinity_matrix_))<1.e-12_dp,'spectral fallback matrix')
  call require(all(s%to_dense()==x),'spectral input preservation')
  d=pairwise_distances(x,x);s=csr_from_dense(d,store_zeros=.true.)
  dense%affinity='precomputed_nearest_neighbors';dense%n_neighbors=4;sparse=dense
  call dense%fit(d);call sparse%fit(s)
  call require(maxval(abs(dense%affinity_matrix_-sparse%affinity_matrix_))==0,'spectral explicit-zero distance graphs')
  cluster%n_clusters=2;cluster%n_neighbors=4;cluster%assign_labels='cluster_qr'
  cluster%affinity='precomputed_nearest_neighbors';labels=cluster%fit_predict(d);other=cluster%fit_predict(s)
  call require(adjusted_rand_score(labels,other)==1,'spectral clustering CSR precomputed neighbors')
  if(arg=='--memory')then
    s=csr_matrix();s%n_rows=32;s%n_cols=1000000
    allocate(s%indptr(33),s%indices(64),s%data(64));s%indptr=[(2*i+1,i=0,32)]
    do i=1,32
      s%indices(2*i-1:2*i)=[1,1000000];s%data(2*i-1:2*i)=[sin(real(i,dp)),cos(.3_dp*i)]
    end do
    sparse%affinity='rbf';sparse%gamma=.4_dp;z=sparse%fit_transform(s)
    call require(all(shape(z)==[32,2]),'wide sparse spectral embedding shape')
    print *,'Spectral CSR 32 by 1000000 memory: PASS'
  end if
  print *,'Spectral CSR affinities, precomputed neighbors and fallback: PASS'
end program
