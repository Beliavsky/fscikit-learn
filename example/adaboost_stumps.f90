program adaboost_stumps
  use sklearn, only: dp,adaboost_classifier
  implicit none
  type(adaboost_classifier) :: model
  real(dp) :: x(6,1)
  x(:,1)=[-3._dp,-2._dp,-1._dp,1._dp,2._dp,3._dp]
  model%n_estimators=10
  call model%fit(x,[-4,-4,7,-4,7,7])
  print *, 'Boosted stump count:',model%n_estimators_
  print *, 'Training accuracy:',model%score(x,[-4,-4,7,-4,7,7])
end program
