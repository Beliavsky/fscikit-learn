program out_of_fold_predictions
  use sklearn, only: dp,ridge,cross_validate,cross_val_predict,cross_validation_options,cross_validation_result,r2_score
  implicit none
  type(ridge) :: model
  type(cross_validation_options) :: options
  type(cross_validation_result) :: results
  real(dp) :: x(40,3),y(40)
  real(dp), allocatable :: predictions(:,:)
  integer :: i
  do i=1,40
    x(i,:)=[sin(real(i,dp)),cos(real(2*i,dp)),sin(real(3*i,dp))]
  end do
  y=2*x(:,1)-x(:,2)+.2_dp*sin(real([(i,i=1,40)]*5,dp))
  model%alpha=.1_dp;options%cv=4;options%return_estimator=.true.
  results=cross_validate(model,x,y,options)
  predictions=cross_val_predict(model,x,y,options)
  write(*,'(a,i0)') 'Folds: ',results%n_splits
  write(*,'(a,i0)') 'Out-of-fold predictions: ',size(predictions,1)
  write(*,'(a,f10.7)') 'Out-of-fold R2: ',r2_score(y,predictions(:,1))
end program
