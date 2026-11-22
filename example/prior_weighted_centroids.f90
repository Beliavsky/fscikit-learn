program prior_weighted_centroids
  use sklearn, only: dp,nearest_centroid
  implicit none
  type(nearest_centroid) :: model
  real(dp) :: x(5,1),query(1,1)
  real(dp), allocatable :: probability(:,:)
  x(:,1)=[-2._dp,-1._dp,0._dp,2._dp,3._dp];query=.5_dp
  call model%fit(x,[-3,-3,-3,9,9])
  print '(a,i4)', 'Uniform prediction: ',model%predict(query)
  model%priors=[1._dp,19._dp];call model%fit(x,[-3,-3,-3,9,9])
  probability=model%predict_proba(query)
  print '(a,i4)', 'Custom-prior prediction: ',model%predict(query)
  print '(a,2f14.8)', 'Class probabilities: ',probability
end program
