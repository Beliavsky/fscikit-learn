program example_threshold_tuning
  use sklearn, only: dp,logistic_regression,fixed_threshold_classifier,tuned_threshold_classifier_cv
  implicit none
  type(logistic_regression) :: base
  type(fixed_threshold_classifier) :: fixed
  type(tuned_threshold_classifier_cv) :: tuned
  real(dp) :: x(120,2)
  integer :: y(120),i
  do i=1,120
    x(i,:)=[sin(real(i,dp)),cos(.37_dp*real(i,dp))]
    y(i)=merge(7,-3,x(i,1)+.3_dp*x(i,2)+.4_dp*sin(.13_dp*i)>.55_dp)
  end do
  base%tol=1.e-10_dp; base%max_iter=1000
  call fixed%fit(x(:90,:),y(:90),base)
  tuned%cv=3; tuned%n_thresholds=51; tuned%store_cv_results=.true.
  call tuned%fit(x(:90,:),y(:90),base)
  print '(a,f10.6)', 'Tuned probability cutoff: ',tuned%best_threshold_
  print '(a,f10.6)', 'CV balanced accuracy: ',tuned%best_score_
  print '(a,f10.6)', 'Default held-out accuracy: ',fixed%score(x(91:,:),y(91:))
  print '(a,f10.6)', 'Tuned held-out accuracy: ',tuned%score(x(91:,:),y(91:))
end program
