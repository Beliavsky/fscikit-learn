program test_regression_metrics
  use sklearn_common, only: dp,require
  use sklearn_metrics, only: mean_pinball_loss,mean_tweedie_deviance,d2_tweedie_score,d2_pinball_score, &
    d2_absolute_error_score,root_mean_squared_log_error
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan,ieee_is_finite
  implicit none
  real(dp) :: y(3),p(3),multi(3,2),pred(3,2),v,w(3)
  real(dp), allocatable :: raw(:)
  y=[1._dp,2._dp,3._dp];p=[1._dp,3._dp,3._dp]
  call require(abs(mean_pinball_loss(y,p)-1._dp/6)<1.e-14_dp,'pinball median loss')
  call require(abs(d2_pinball_score(y,p)-.5_dp)<1.e-14_dp,'D2 pinball example')
  call require(d2_absolute_error_score(y,p)==d2_pinball_score(y,p),'D2 absolute median identity')
  call require(mean_tweedie_deviance(y,p)==1._dp/3,'Tweedie normal MSE')
  call require(abs(d2_tweedie_score(y,p)-.5_dp)<1.e-14_dp,'D2 normal R2')
  call require(mean_tweedie_deviance(y,y,power=1._dp)==0,'perfect Poisson deviance')
  call require(mean_tweedie_deviance(y,y,power=2._dp)==0,'perfect Gamma deviance')
  w=[0._dp,2._dp,1._dp]
  call require(abs(mean_pinball_loss(y,p,w)-1._dp/3)<1.e-14_dp,'weighted pinball')
  multi(:,1)=y;multi(:,2)=2*y;pred(:,1)=p;pred(:,2)=2*p
  v=mean_pinball_loss(multi,pred,output_weights=[1._dp,3._dp],raw_values=raw)
  call require(size(raw)==2.and.abs(v-7._dp/24)<1.e-14_dp,'multioutput weighted pinball and raw values')
  v=root_mean_squared_log_error(multi,pred,raw_values=raw)
  call require(abs(v-sum(raw)/2)<1.e-14_dp,'RMSLE averages roots per output')
  call require(root_mean_squared_log_error([-.5_dp],[0._dp])>0,'RMSLE permits targets above minus one')
  call require(d2_pinball_score([2._dp,2._dp],[2._dp,2._dp])==1,'constant perfect pinball score')
  call require(d2_pinball_score([2._dp,2._dp],[1._dp,3._dp])==0,'constant imperfect pinball score')
  call require(ieee_is_nan(d2_absolute_error_score(y(:1),p(:1))),'singleton D2 NaN')
  v=mean_pinball_loss(y,y,alpha=0._dp);call require(v==0,'alpha endpoint')
  print *, 'Weighted pinball/Tweedie deviance, D2 scores and multioutput RMSLE: PASS'
end program
