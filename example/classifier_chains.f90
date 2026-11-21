program classifier_chains
  use sklearn, only: dp,classifier_chain,logistic_regression
  implicit none
  type(classifier_chain) :: model
  type(logistic_regression) :: base
  real(dp) :: x(12,1),query(2,1)
  integer :: y(12,2),i
  x(:,1)=[(real(i-6,dp),i=1,12)]
  y(:,1)=[0,0,0,0,0,0,1,1,1,1,1,1]; y(:,2)=[0,0,0,1,1,1,1,1,1,0,0,0]
  query(:,1)=[-2._dp,2._dp]
  model%order=[2,1]; model%cv=3; model%chain_method='predict_proba'
  call model%fit(x,y,base)
  print *, 'Classifier chain order:',model%order_
  print *, 'Classifier chain predictions:',model%predict(query)
  print *, 'Classifier chain positive probabilities:',model%predict_proba(query)
end program
