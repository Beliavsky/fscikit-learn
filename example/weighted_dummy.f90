program weighted_dummy
  use sklearn,only:dp,dummy_classifier,dummy_regressor
  implicit none
  type(dummy_classifier)::classifier
  type(dummy_regressor)::regressor
  real(dp)::x(4,0),y(4,2),w(4)
  integer::labels(4)
  y(:,1)=[1._dp,2._dp,3._dp,4._dp];y(:,2)=10*y(:,1)
  w=[0._dp,1._dp,3._dp,2._dp];labels=[-7,4,4,19]
  call classifier%fit(x,labels,w)
  print *,'Weighted class priors:',classifier%class_prior_
  regressor%strategy='median';call regressor%fit(x,y,w)
  print *,'Weighted output medians:',regressor%constants_
end program
