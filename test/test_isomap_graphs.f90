program test_isomap_graphs
  use sklearn_common, only: dp,require
  use sklearn_feature_extraction, only: feature_string
  use sklearn_manifold, only: isomap
  use sklearn_metrics, only: pairwise_distances
  use sklearn_sparse, only: csr_matrix,csr_from_dense
  implicit none
  type(isomap) :: knn,radius,pre,sparse
  type(csr_matrix) :: graph
  type(feature_string), allocatable :: names(:)
  real(dp) :: x(12,2)
  real(dp), allocatable :: d(:,:),z(:,:),other(:,:),before(:,:),q(:,:)
  integer :: i
  do i=1,12
    x(i,:)=[real(i,dp),.1_dp*sin(real(i,dp))]
  end do
  knn%n_neighbors=2;knn%path_method='D';z=knn%fit_transform(x)
  call require(knn%path_method_=='D','actual Dijkstra selection')
  d=pairwise_distances(x,x);pre%metric='precomputed';pre%n_neighbors=2;pre%path_method='FW'
  other=pre%fit_transform(d)
  call require(maxval(abs(knn%dist_matrix_-pre%dist_matrix_))<1.e-12_dp,'FW and Dijkstra agreement')
  call require(maxval(abs(z-other))<1.e-10_dp,'precomputed Isomap equivalence')
  graph=csr_from_dense(d,.true.);sparse=pre;other=sparse%fit_transform(graph)
  call require(maxval(abs(z-other))<1.e-10_dp,'CSR Isomap equivalence including explicit zeros')
  q=sparse%transform(graph);call require(maxval(abs(q-z))<1.e-10_dp,'CSR out-of-sample projection')
  radius%n_neighbors=0;radius%radius=1.1_dp;radius%path_method='D';other=radius%fit_transform(x)
  call require(radius%n_connected_components_==1,'radius connected graph')
  before=radius%dist_matrix_;radius%radius=0;other=radius%transform(x)
  call require(maxval(abs(other-radius%embedding_))<1.e-10_dp,'fitted radius controls govern transform')
  call radius%fit(x)
  call require(radius%n_connected_components_==12,'radius component completion records original components')
  call require(maxval(abs(radius%dist_matrix_-d))<1.e-12_dp,'zero-radius completion uses all component pairs')
  pre%n_components=20;call pre%fit(d)
  call require(pre%n_features_in_==12.and.pre%n_samples_fit_==12.and.pre%n_features_out_==12, &
    'fitted dimensions and oversized component truncation')
  names=knn%get_feature_names_out(['first ','second'])
  call require(size(names)==2.and.names(1)%text=='isomap0'.and.names(2)%text=='isomap1','Isomap output feature names')
  sparse%metric='euclidean';call sparse%fit(x);other=sparse%transform(x)
  call require(maxval(abs(other-z))<1.e-10_dp,'CSR to feature-data refit resets fitted state')
  print *, 'Isomap radius and precomputed graphs: PASS'
end program
