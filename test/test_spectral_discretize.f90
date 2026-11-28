program test_spectral_discretize
  use sklearn_common,only:dp,require
  use sklearn_spectral,only:spectral_discretize
  use sklearn_cluster,only:spectral_clustering
  use sklearn_sparse,only:csr_from_dense
  use sklearn_metrics,only:adjusted_rand_score
  implicit none
  real(dp)::x(12,3),before(12,3),graph(12,12)
  integer,allocatable::labels(:),other(:)
  integer::i,truth(12)
  type(spectral_clustering)::m
  x=0;graph=0
  do i=1,12
    truth(i)=(i-1)/4;x(i,truth(i)+1)=1
    graph(i,truth(i)*4+1:truth(i)*4+4)=1
  end do
  before=x;labels=spectral_discretize(x,random_state=21)
  call require(adjusted_rand_score(labels,truth)==1,'discretize one-hot partition')
  call require(all(x==before),'discretize preserves input')
  other=spectral_discretize(x,random_state=21)
  call require(all(other==labels),'discretize seeded repeatability')
  m%n_clusters=3;m%affinity='precomputed';m%assign_labels='discretize';m%random_state=21
  labels=m%fit_predict(graph)
  call require(adjusted_rand_score(labels,truth)==1,'spectral discretize dense precomputed partition')
  other=m%fit_predict(csr_from_dense(graph))
  call require(adjusted_rand_score(other,truth)==1,'spectral discretize CSR precomputed partition')
  print *,'Spectral discretization input preservation, seed and dense/CSR partitions: PASS'
end program
