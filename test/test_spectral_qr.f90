program test_spectral_qr
  use sklearn_common,only:dp,require
  use sklearn_spectral_qr,only:spectral_cluster_qr,spectral_qr_pivots
  use sklearn_cluster,only:spectral_clustering
  use sklearn_sparse,only:csr_from_dense
  use sklearn_metrics,only:adjusted_rand_score
  implicit none
  real(dp)::x(12,3),before(12,3),graph(12,12)
  integer::i,truth(12)
  integer,allocatable::labels(:),pivots(:),other(:)
  type(spectral_clustering)::m
  x=0;graph=0
  do i=1,12
    truth(i)=(i-1)/4;x(i,truth(i)+1)=1
    graph(i,truth(i)*4+1:truth(i)*4+4)=1
  end do
  before=x;labels=spectral_cluster_qr(x,pivots)
  call require(all(x==before),'spectral QR preserves embedding')
  call require(adjusted_rand_score(labels,truth)==1,'spectral QR one-hot partition')
  call require(size(pivots)==12.and.all(pivots>=1).and.all(pivots<=12),'spectral QR pivot range')
  call require(pivots(1)==1.and.pivots(2)==5.and.pivots(3)==9,'spectral QR known independent rows')
  other=spectral_qr_pivots(x)
  call require(all(other==pivots),'spectral QR exposed pivot sequence')
  x=0;labels=spectral_cluster_qr(x)
  call require(all(labels==0),'spectral QR accepts zero embeddings like Source')
  m%n_clusters=3;m%affinity='precomputed';m%assign_labels='cluster_qr'
  labels=m%fit_predict(graph);other=m%fit_predict(csr_from_dense(graph))
  call require(adjusted_rand_score(labels,truth)==1.and.adjusted_rand_score(other,truth)==1, &
    'spectral Householder QR dense/CSR estimator integration')
  print *,'Spectral Householder pivots, input preservation and partitions: PASS'
end program
