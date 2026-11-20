program test_bisecting_kmeans
  use sklearn, only: dp,bisecting_kmeans
  use sklearn_common, only: require
  use sklearn_legacy_rng, only: legacy_random_state
  implicit none
  type(bisecting_kmeans) :: model
  type(legacy_random_state) :: rng
  real(dp) :: x(6,1),w(6),cloud(80,2),query(121,2)
  real(dp), allocatable :: distances(:,:),before(:,:)
  integer, allocatable :: labels(:),nearest(:)
  integer :: i,j,k
  x(:,1)=[-3._dp,-2._dp,-1._dp,1._dp,2._dp,3._dp]; w=[1._dp,2._dp,1._dp,1._dp,1._dp,2._dp]
  model%n_clusters=2; model%tol=0; model%n_init=3
  distances=model%fit_transform(x,w)
  call require(abs(minval(model%cluster_centers_)+2)<1.e-12_dp,'bisect weighted negative center')
  call require(abs(maxval(model%cluster_centers_)-2.25_dp)<1.e-12_dp,'bisect weighted positive center')
  call require(abs(model%inertia_-4.75_dp)<1.e-12_dp,'bisect analytic weighted inertia')
  call require(abs(model%score(x,w)+4.75_dp)<1.e-12_dp,'bisect nearest-center score')
  call require(all(shape(distances)==[6,2]),'bisect fit_transform shape')
  before=model%cluster_centers_; call model%fit(x,w)
  call require(maxval(abs(model%cluster_centers_-before))<1.e-14_dp,'bisect seeded repeated fit')
  model%n_clusters=1; call model%fit(x,w)
  call require(abs(model%cluster_centers_(1,1))<1.e-14_dp,'bisect source one-cluster unweighted root')
  model%n_clusters=3; model%algorithm='elkan'; model%init='k-means++'; labels=model%fit_predict(2+0*x)
  call require(all(labels==0).and.all(model%cluster_centers_==2),'bisect duplicate/empty leaves')
  call require(model%node_count_==5.and.count(model%children_(:,1)==0)==3,'bisect tree topology')
  call rng%seed(2026)
  do i=1,80
    cloud(i,:)=[rng%normal(),rng%normal()]
  end do
  k=0
  do i=1,11
    do j=1,11
      k=k+1; query(k,:)=[real(i-6,dp)*.6_dp,real(j-6,dp)*.6_dp]
    end do
  end do
  model=bisecting_kmeans(); model%n_clusters=5; call model%fit(cloud)
  labels=model%predict(query); distances=model%transform(query); nearest=minloc(distances,dim=2)-1
  call require(any(labels/=nearest),'bisect hierarchical prediction differs from global nearest center')
  call require(-model%score(cloud)<=model%inertia_+1.e-10_dp,'bisect global score versus hierarchical inertia')
  print *, 'BisectingKMeans tests passed'
end program
