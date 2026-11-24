program regression_loss_scores
  use sklearn, only: dp,quantile_regressor,mean_pinball_loss,d2_pinball_score,d2_absolute_error_score, &
    mean_tweedie_deviance,d2_tweedie_score,root_mean_squared_log_error
  implicit none
  type(quantile_regressor) :: model
  real(dp) :: x(40,1),y(40),multi(40,2),multiple_prediction(40,2),value
  real(dp), allocatable :: predicted(:),raw(:)
  integer :: i
  do i=1,40
    x(i,1)=real(i,dp)/10
    y(i)=2+2*x(i,1)+(.2_dp+x(i,1)/5)*sin(real(i*7,dp))
  end do
  model%alpha=.01_dp;model%quantile=.9_dp;call model%fit(x,y);predicted=model%predict(x)
  print *, 'Quantile model converged:',model%converged_
  print *, 'Pinball loss:',mean_pinball_loss(y,predicted,alpha=.9_dp)
  print *, 'D2 pinball:',d2_pinball_score(y,predicted,alpha=.9_dp)
  print *, 'D2 absolute error:',d2_absolute_error_score(y,predicted)
  print *, 'Poisson deviance:',mean_tweedie_deviance(y,predicted,power=1._dp)
  print *, 'D2 Poisson:',d2_tweedie_score(y,predicted,power=1._dp)
  print *, 'RMSLE:',root_mean_squared_log_error(y,predicted)
  multi(:,1)=y;multi(:,2)=2*y;multiple_prediction(:,1)=predicted;multiple_prediction(:,2)=2*predicted
  value=mean_pinball_loss(multi,multiple_prediction,alpha=.9_dp,output_weights=[1._dp,3._dp],raw_values=raw)
  print *, 'Per-output pinball shape:',size(raw)
  print *, 'Weighted output aggregation agrees:',abs(value-(raw(1)+3*raw(2))/4)<1.e-14_dp
end program
