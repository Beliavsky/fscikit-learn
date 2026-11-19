program optics_demo
  use sklearn
  implicit none
  type(optics) :: model
  real(dp) :: x(8,1)
  integer, allocatable :: labels(:)
  x(:,1)=[0._dp,.1_dp,.2_dp,.3_dp,5._dp,5.1_dp,5.2_dp,5.3_dp]
  model%min_samples=2; model%min_cluster_size=3
  labels=model%fit_predict(x)
  print *, 'Xi cluster labels:',labels
  labels=cluster_optics_dbscan(model%reachability_,model%core_distances_,model%ordering_,.25_dp)
  print *, 'DBSCAN extraction:',labels
end program
