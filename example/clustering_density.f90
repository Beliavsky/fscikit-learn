program clustering_density
  use sklearn
  implicit none
  type(mean_shift) :: modes
  type(kernel_density) :: density
  real(dp) :: x(60,2),queries(2,2)
  integer :: i,c
  do i=1,60
    c=(i-1)/20
    x(i,:)=[3._dp*(c-1),merge(3._dp,-3._dp,c==1)]+0.08_dp*[sin(real(i,dp)),cos(1.7_dp*i)]
  end do
  modes%bandwidth=0.6_dp; call modes%fit(x)
  print *, 'MeanShift clusters:',size(modes%cluster_centers_,1),' converged:',modes%converged_
  density%bandwidth=0.3_dp; call density%fit(x)
  queries(1,:)=[0._dp,3._dp]; queries(2,:)=[0._dp,0._dp]
  print *, 'Log densities at a cluster center and between clusters:',density%score_samples(queries)
end program
