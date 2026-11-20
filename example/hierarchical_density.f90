program hierarchical_density
  use sklearn, only: dp,hdbscan
  use sklearn_legacy_rng, only: legacy_random_state
  implicit none
  type(hdbscan) :: model
  type(legacy_random_state) :: rng
  real(dp) :: x(90,2)
  integer, allocatable :: cut_labels(:)
  integer :: i
  call rng%seed(42)
  do i=1,90
    x(i,:)=[rng%normal(),rng%normal()]*.3_dp
    x(i,1)=x(i,1)+5*(mod(i-1,3)-1)
  end do
  model%min_cluster_size=8; model%min_samples=5; model%store_centers='both'
  call model%fit(x)
  cut_labels=model%dbscan_clustering(.7_dp,5)
  print *, 'Density clusters/noise: ',model%n_clusters_,count(model%labels_==-1)
  print *, 'Weighted centroid first coordinates: ',model%centroids_(:,1)
  print *, 'Membership strength range: ',minval(model%probabilities_),maxval(model%probabilities_)
  print *, 'DBSCAN-cut noise points: ',count(cut_labels==-1)
end program
