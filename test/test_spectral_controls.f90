program test_spectral_controls
  use sklearn_common,only:dp,require,unique_labels
  use sklearn_cluster,only:spectral_clustering
  use sklearn_sparse,only:csr_from_dense
  implicit none
  type(spectral_clustering)::m
  real(dp)::graph(12,12),x(12,3)
  integer::i
  graph=0
  do i=1,12
    graph(i,((i-1)/3)*3+1:((i-1)/3)*3+3)=1
    x(i,:)=[real(i,dp)/12,sin(real(i,dp))**2,cos(.3_dp*i)**2]
  end do
  m%n_clusters=2;m%n_components=4;m%affinity='precomputed';m%assign_labels='cluster_qr'
  call m%fit(graph)
  call require(size(m%embedding_,2)==4,'spectral n_components is independent of n_clusters')
  call require(size(unique_labels(m%labels_))==4,'spectral QR assigns by embedding dimension')
  call m%fit(csr_from_dense(graph))
  call require(size(unique_labels(m%labels_))==4,'spectral CSR component/cluster independence')
  m%n_components=1;call m%fit(graph)
  call require(all(m%labels_==0),'spectral QR one-component assignment')
  m%affinity='polynomial';m%degree=2.5_dp;m%coef0=.7_dp;m%gamma=.4_dp;m%n_components=3
  call m%fit(x)
  call require(maxval(abs(m%affinity_matrix_-(.4_dp*matmul(x,transpose(x))+.7_dp)**2.5_dp))<1.e-12_dp, &
    'spectral fractional polynomial controls')
  graph=1;graph(1,2)=-.25_dp;graph(2,1)=-.25_dp
  m%affinity='precomputed';m%n_components=2;call m%fit(graph)
  call require(m%affinity_matrix_(1,2)<0,'spectral signed affinities are retained')
  graph(1,2)=.1_dp;graph(2,1)=.9_dp;call m%fit(graph)
  call require(m%affinity_matrix_(1,2)==.1_dp.and.m%affinity_matrix_(2,1)==.9_dp, &
    'spectral stores original asymmetric precomputed affinity')
  m%n_components=20;call m%fit(graph)
  call require(size(m%embedding_,2)==12,'spectral dense eigensolver shape fallback')
  x(1,:)=0;m%affinity='linear';m%n_components=1;m%assign_labels='discretize'
  call m%fit(x)
  call require(all(m%labels_==0),'one-component spectral discretization assigns one label with isolated rows')
  print *,'Spectral kernels, component controls, signed/asymmetric affinities: PASS'
end program
