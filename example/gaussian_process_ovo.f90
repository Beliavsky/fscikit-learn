program gaussian_process_ovo
  use sklearn, only: dp,gaussian_process_classifier,rbf_covariance,white_covariance,operator(+)
  implicit none
  type(gaussian_process_classifier) :: model
  real(dp) :: x(15,1),query(3,1)
  integer :: i,labels(15)
  x(:,1)=[(real(i,dp)/5-1.6_dp,i=1,15)]
  labels=[-4,-4,-4,-4,9,9,9,9,9,17,17,17,17,17,17]
  model%multi_class='one_vs_one'
  model%kernel=rbf_covariance(.7_dp)+white_covariance(.04_dp)
  call model%fit(x,labels)
  query(:,1)=[-1.2_dp,-.2_dp,1._dp]
  print '(a,3i5)', 'Predicted classes: ',model%predict(query)
  print '(a,3i5)', 'Pair training sizes: ',model%n_training_samples_
  print '(a,f14.7)', 'Mean pair likelihood: ',model%log_marginal_likelihood()
end program
