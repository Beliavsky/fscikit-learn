program test_optics
  use sklearn
  use sklearn_common, only: require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  type(optics) :: model,precomputed
  real(dp) :: x(8,1),zero(4,2)
  real(dp), allocatable :: d(:,:)
  integer, allocatable :: labels(:),hierarchy(:,:)
  integer :: i
  x(:,1)=[0._dp,.1_dp,.2_dp,.3_dp,5._dp,5.1_dp,5.2_dp,5.3_dp]
  model%min_samples=2; model%min_cluster_size=3
  labels=model%fit_predict(x)
  call require(all(labels(:4)==labels(1)).and.all(labels(5:)==labels(5)), 'Xi separates dense groups')
  call require(labels(1)>=0.and.labels(5)>=0.and.labels(1)/=labels(5),'two Xi clusters')
  call require(all(shape(model%cluster_hierarchy_)==[3,2]),'nested Xi hierarchy includes parent')
  call require(model%predecessor_(1)==-1,'first point has no predecessor')
  call require(.not.ieee_is_finite(model%reachability_(1)),'root reachability is infinite')
  do i=1,8
    call require(count(model%ordering_==i)==1,'ordering is permutation')
  end do
  d=pairwise_distances(x,x)
  precomputed=model; precomputed%metric='precomputed'; call precomputed%fit(d)
  call require(all(precomputed%labels_==labels),'precomputed cluster agreement')
  model%cluster_method='dbscan'; model%eps=.25_dp; model%max_eps=.5_dp
  call model%fit(x)
  call require(all(model%labels_(:4)==0).and.all(model%labels_(5:)==1),'DBSCAN extraction')
  call require(size(model%cluster_hierarchy_)==0,'refit clears Xi hierarchy')
  labels=cluster_optics_dbscan(model%reachability_,model%core_distances_,model%ordering_,.01_dp)
  call require(all(labels==-1),'small radius is all noise')
  model%cluster_method='xi'; model%max_eps=0; call model%fit(x)
  call require(all(model%labels_==-1),'disconnected graph is all noise')
  call require(all(model%predecessor_==-1),'disconnected roots')
  zero=0; model%min_cluster_size=0; model%min_samples_fraction=.5_dp
  labels=model%fit_predict(zero)
  call require(all(labels==0),'duplicate plateau cluster')
  call require(model%min_samples_==2.and.model%min_cluster_size_==2,'fraction and default cluster size')
  call require(all(model%ordering_==[1,2,3,4]),'ties use original row order')
  call cluster_optics_xi(model%reachability_,model%predecessor_,model%ordering_,2,labels,hierarchy)
  call require(all(labels==model%labels_),'public Xi re-extraction')
  model%cluster_method='dbscan'; model%max_eps=huge(1._dp); model%eps=-1; call model%fit(zero)
  call require(all(model%labels_==-1),'unbounded DBSCAN eps follows sklearn infinity comparisons')
  print *, 'OPTICS tests passed'
end program
