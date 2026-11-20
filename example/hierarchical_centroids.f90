program hierarchical_centroids
  use sklearn, only: dp,bisecting_kmeans
  use sklearn_legacy_rng, only: legacy_random_state
  implicit none
  type(bisecting_kmeans) :: model
  type(legacy_random_state) :: rng
  real(dp) :: x(90,2),query(3,2)
  integer :: i
  call rng%seed(42)
  do i=1,90
    x(i,:)=[rng%normal(),rng%normal()]
    x(i,1)=x(i,1)+4*((i-1)/30)
  end do
  model%n_clusters=3; model%init='k-means++'; model%n_init=3; model%random_state=42
  call model%fit(x)
  query(:,1)=[0._dp,4._dp,8._dp]; query(:,2)=0
  print *, 'Centroid x coordinates:',model%cluster_centers_(:,1)
  print *, 'Hierarchical training inertia:',model%inertia_
  print *, 'Query cluster labels:',model%predict(query)
end program
