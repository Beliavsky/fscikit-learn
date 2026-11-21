program self_training
  use sklearn, only: dp,self_training_classifier,kneighbors_classifier
  implicit none
  type(self_training_classifier) :: model
  type(kneighbors_classifier) :: estimator
  real(dp) :: x(8,1)
  integer :: y(8)=[0,0,-1,-1,-1,-1,1,1]
  x(:,1)=[-4._dp,-3._dp,-2._dp,-1._dp,1._dp,2._dp,3._dp,4._dp]
  estimator%n_neighbors=1
  model%criterion='k_best'; model%k_best=2; model%max_iter=-1
  call model%fit(x,y,estimator)
  print *, 'Transduced labels:',model%transduction_
  print *, 'Labeling iterations:',model%labeled_iter_
  print *, 'Stopped after',model%n_iter_,'iterations: ',trim(model%termination_condition_)
  print *, 'Query labels:',model%predict(reshape([-5._dp,5._dp],[2,1]))
end program
