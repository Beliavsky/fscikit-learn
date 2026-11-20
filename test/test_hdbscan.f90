program test_hdbscan
  use sklearn, only: dp,hdbscan
  use sklearn_common, only: require
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_positive_inf,ieee_quiet_nan,ieee_is_nan
  implicit none
  type(hdbscan) :: model
  real(dp) :: x(12,1),before(12,1),dist(3,3),disconnected(4,4),tiny_graph(4,4),missing(5,2),inf
  integer, allocatable :: labels(:),old(:)
  integer :: i,j
  inf=ieee_value(0._dp,ieee_positive_inf)
  x(:,1)=[-5._dp,-4.9_dp,-4.8_dp,-4.7_dp,-4.6_dp,5._dp,5.1_dp,5.2_dp,5.3_dp,5.4_dp,-20._dp,20._dp]
  before=x; model%min_cluster_size=3; model%min_samples=3; model%store_centers='both'
  labels=model%fit_predict(x)
  call require(all(x==before),'HDBSCAN preserves feature input')
  call require(model%n_clusters_==2.and.all(labels(:5)==labels(1)).and.all(labels(6:10)==labels(6)), &
    'HDBSCAN separated density clusters')
  call require(labels(1)>=0.and.labels(6)>=0.and.labels(1)/=labels(6),'HDBSCAN distinct cluster labels')
  call require(all(labels(11:)==-1).and.all(model%probabilities_(11:)==0),'HDBSCAN noise strength')
  call require(maxval(abs(model%core_distances_(:5)-[.2_dp,.1_dp,.1_dp,.1_dp,.2_dp]))<1.e-12_dp, &
    'HDBSCAN min_samples includes self')
  call require(maxval(abs(model%probabilities_(:5)-[.5_dp,1._dp,1._dp,1._dp,.5_dp]))<1.e-12_dp, &
    'HDBSCAN analytic membership strengths')
  call require(abs(model%centroids_(labels(1)+1,1)+4.8_dp)<1.e-12_dp,'HDBSCAN probability-weighted centroid')
  call require(abs(model%medoids_(labels(6)+1,1)-5.2_dp)<1.e-12_dp,'HDBSCAN weighted medoid')
  old=labels; call model%fit(x)
  call require(all(model%labels_==old),'HDBSCAN deterministic repeated fit')
  labels=model%dbscan_clustering(.15_dp,3)
  call require(count(labels>=0)==6.and.all(labels(2:4)==labels(2)).and.all(labels(7:9)==labels(7)), &
    'HDBSCAN DBSCAN cut excludes border points')
  model%metric='precomputed'; model%store_centers='none'; model%min_samples=1; model%min_cluster_size=2
  model%allow_single_cluster=.true.
  dist=reshape([0._dp,1._dp,3._dp,1._dp,0._dp,2._dp,3._dp,2._dp,0._dp],[3,3])
  call model%fit(dist)
  call require(.not.allocated(model%centroids_).and..not.allocated(model%medoids_),'HDBSCAN refit clears stored centers')
  labels=model%dbscan_clustering(1._dp,2)
  call require(all(labels==-1),'HDBSCAN cut excludes edges equal to threshold')
  labels=model%dbscan_clustering(1.01_dp,2)
  call require(labels(1)>=0.and.labels(1)==labels(2).and.labels(3)==-1,'HDBSCAN cut merges shorter edges')
  model%alpha=2; call model%fit(dist)
  call require(maxval(abs(model%distances_-[.5_dp,1._dp]))<1.e-14_dp,'HDBSCAN brute alpha scales all distances')
  model=hdbscan(); model%metric='precomputed'; model%min_samples=1; model%min_cluster_size=2
  disconnected=inf
  do i=1,4
    disconnected(i,i)=0
  end do
  disconnected(1,2)=1; disconnected(2,1)=1; disconnected(3,4)=1; disconnected(4,3)=1
  call model%fit(disconnected)
  call require(model%has_infinite_edges_.and.model%n_clusters_==2,'HDBSCAN missing precomputed distances')
  call require(model%labels_(1)==model%labels_(2).and.model%labels_(3)==model%labels_(4), &
    'HDBSCAN disconnected finite components')
  call require(all(model%probabilities_==1),'HDBSCAN disconnected component strengths')
  tiny_graph=disconnected
  where(tiny_graph==inf) tiny_graph=3.e-310_dp
  where(tiny_graph==1) tiny_graph=1.e-310_dp
  call model%fit(tiny_graph)
  call require(model%n_clusters_==2.and.all(model%probabilities_==1),'HDBSCAN infinite lambda strengths')
  call require(any(ieee_is_nan(model%stabilities_)),'HDBSCAN zero-lifetime infinite stability handling')
  model=hdbscan(); model%min_cluster_size=3; model%min_samples=1; model%allow_single_cluster=.true.
  call model%fit(0*x+2)
  call require(all(model%labels_==0).and.all(model%probabilities_==1),'HDBSCAN duplicate single cluster')
  model%cluster_selection_method='leaf'; call model%fit(0*x+2)
  call require(all(model%labels_==-1),'HDBSCAN leaf selection has no cluster edges')
  model=hdbscan(); model%min_cluster_size=2; model%min_samples=1; model%allow_single_cluster=.true.
  model%store_centers='both'; missing=0
  missing(1,:)=[0._dp,0._dp]; missing(2,:)=inf; missing(3,:)=[1._dp,0._dp]
  missing(4,:)=ieee_value(0._dp,ieee_quiet_nan); missing(5,:)=[inf,-inf]
  call model%fit(missing)
  call require(model%labels_(2)==-2.and.all(model%labels_(4:)==-3),'HDBSCAN nonfinite feature labels')
  call require(model%probabilities_(2)==0.and.all(ieee_is_nan(model%probabilities_(4:))), &
    'HDBSCAN nonfinite membership strengths')
  call require(model%labels_(1)==0.and.model%labels_(3)==0,'HDBSCAN finite subset cluster')
  call require(all(shape(model%centroids_)==[1,2]).and.abs(model%centroids_(1,1)-.5_dp)<1.e-12_dp, &
    'HDBSCAN centers with nonfinite feature rows')
  call require(all(model%children_(2,:)==[2,6]).and.all(model%children_(3,:)==[4,7]), &
    'HDBSCAN two-finite-row hierarchy remapping')
  labels=model%dbscan_clustering(2._dp,2)
  call require(all(labels([1,3])==0).and.labels(2)==-2.and.all(labels(4:)==-3),'HDBSCAN remapped DBSCAN cut')
  print *, 'HDBSCAN tests passed'
end program
