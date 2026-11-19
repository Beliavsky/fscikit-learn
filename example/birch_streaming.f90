program birch_streaming
  use sklearn
  implicit none
  type(birch) :: model
  real(dp) :: x(8,1)
  integer, allocatable :: labels(:)
  x(:,1)=[0._dp,.1_dp,.2_dp,.3_dp,5._dp,5.1_dp,5.2_dp,5.3_dp]
  model%threshold=.2_dp; model%branching_factor=2; model%n_clusters=0
  call model%partial_fit(x(:4,:))
  call model%partial_fit(x(5:,:))
  labels=model%predict(x)
  print *, 'Subcluster labels:',labels
  model%n_clusters=2; call model%partial_fit()
  labels=model%predict(x)
  print *, 'Ward cluster labels:',labels
end program
