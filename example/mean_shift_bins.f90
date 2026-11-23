program mean_shift_bins
  use sklearn_cluster, only: mean_shift,estimate_bandwidth
  use sklearn_common, only: dp
  implicit none
  type(mean_shift) :: model
  real(dp) :: x(30,2),sampled_bandwidth
  integer :: i
  do i=1,30
    x(i,:)=[3._dp*real((i-1)/10,dp),0._dp]+.04_dp*[sin(real(i,dp)),cos(real(i,dp))]
  end do
  model%bandwidth=.4_dp;model%bin_seeding=.true.;model%min_bin_freq=3
  call model%fit(x)
  sampled_bandwidth=estimate_bandwidth(x,quantile=.3_dp,n_samples=20,random_state=42)
  print *, 'Clusters from frequent grid bins:',size(model%cluster_centers_,1)
  print *, 'Singleton-bin fallback:',model%binning_failed_
  print *, 'Seeded subsampled bandwidth:',sampled_bandwidth
end program
