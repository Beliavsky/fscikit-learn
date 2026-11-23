program density_seeded_sampling
  use sklearn, only: dp,kernel_density
  use sklearn_legacy_rng, only: legacy_random_state
  implicit none
  type(kernel_density) :: model
  type(legacy_random_state) :: rng
  real(dp) :: x(2,2)
  real(dp), allocatable :: draws(:,:),scores(:)
  x(1,:)=[0._dp,0._dp];x(2,:)=[4._dp,-2._dp]
  model%algorithm='ball_tree';model%bandwidth=.7_dp
  call model%fit(x,sample_weight=[0._dp,1._dp])
  draws=model%sample(3,random_state=42)
  print *, 'Gaussian sample dimensions:',shape(draws)
  print *, 'First Gaussian draw:',draws(1,:)
  model%kernel='tophat';call rng%seed(42)
  draws=model%sample(3,random_state=rng)
  print *, 'First tophat draw:',draws(1,:)
  model%kernel='cosine';scores=model%score_samples(x(2:2,:))
  print *, 'Cosine center log score:',scores(1)
end program
