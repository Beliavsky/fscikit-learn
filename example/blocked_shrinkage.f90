program blocked_shrinkage
  use sklearn,only:dp,ledoit_wolf_shrinkage,ledoit_wolf
  implicit none
  real(dp)::x(4,2)
  type(ledoit_wolf)::model
  x(:,1)=[1._dp,-1._dp,0._dp,0._dp];x(:,2)=x(:,1)
  print *,'Shrinkage:',ledoit_wolf_shrinkage(x,assume_centered=.true.,block_size=1)
  model%block_size=1;model%assume_centered=.true.;call model%fit(x)
  print *,'Covariance:',model%covariance_
end program
