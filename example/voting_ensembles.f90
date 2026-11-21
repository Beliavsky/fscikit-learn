program voting_ensembles
  use sklearn, only: dp,voting_classifier,voting_estimator,logistic_regression,gaussian_nb
  implicit none
  type(voting_classifier) :: model
  type(voting_estimator) :: bases(2)
  type(logistic_regression) :: lr
  type(gaussian_nb) :: nb
  real(dp) :: x(8,1),query(3,1)
  integer :: i
  x(:,1)=[(real(i,dp),i=1,8)]; query(:,1)=[1._dp,4._dp,8._dp]
  bases(1)%name='logistic'; allocate(bases(1)%estimator,source=lr)
  bases(2)%name='gaussian'; allocate(bases(2)%estimator,source=nb)
  model%voting='soft'; model%weights=[2._dp,1._dp]
  call model%fit(x,[-7,-7,-7,-7,19,19,19,19],bases)
  print *, 'Voting labels:',model%predict(query)
  print *, 'Voting probabilities:',model%predict_proba(query)
  print *, 'Per-estimator probabilities:',model%transform(query)
end program
