program raw_bernoulli
  use sklearn,only:dp,bernoulli_nb
  implicit none
  type(bernoulli_nb)::model
  real(dp)::x(4,2)
  integer::y(4)
  x(:,1)=[.1_dp,.2_dp,.8_dp,.9_dp];x(:,2)=[.8_dp,.9_dp,.1_dp,.2_dp];y=[-7,-7,4,4]
  model%binarize=.false.
  call model%fit(x,y)
  print *,'Raw feature sums:',model%feature_count_
  print *,'Class probabilities:',model%predict_proba(x)
end program
