program test_cv_clone
  use sklearn_common,only:dp,require
  use sklearn_linear_model,only:poisson_regressor,huber_regressor
  use sklearn_model_selection,only:cross_val_score
  implicit none
  type(poisson_regressor)::cold,warm
  type(huber_regressor)::hc,hw
  real(dp)::x(35,3),y(35),other(35),a(4),b(4),weights(35)
  real(dp),allocatable::before(:)
  integer::i
  do i=1,35
    x(i,:)=[sin(real(i,dp)),cos(real(i,dp)*.7_dp),real(i-17,dp)/35]
  end do
  y=exp(.3_dp+.4_dp*x(:,1)-.2_dp*x(:,2));other=exp(1.1_dp-.8_dp*x(:,1)+.7_dp*x(:,2))
  cold%max_iter=1;warm=cold;warm%warm_start=.true.;call warm%fit(x,other);before=warm%coef_
  a=cross_val_score(cold,x,y,cv=4);b=cross_val_score(warm,x,y,cv=4)
  call require(maxval(abs(a-b))<1.e-12_dp,'GLM CV ignores fitted caller and warm starts')
  call require(all(warm%coef_==before).and..not.allocated(cold%coef_),'GLM CV caller state')
  weights=1.7_dp;a=cross_val_score(warm,x,y,cv=4,sample_weight=weights)
  b=cross_val_score(warm,x,y,cv=4,sample_weight=1.7_dp)
  call require(maxval(abs(a-b))<1.e-12_dp,'scalar/vector GLM CV fitting weights')
  y=.7_dp+.4_dp*x(:,1)-.2_dp*x(:,2);other=2-.8_dp*x(:,1)+.7_dp*x(:,2)
  hc%max_iter=1;hw=hc;hw%warm_start=.true.;call hw%fit(x,other);before=hw%coef_
  a=cross_val_score(hc,x,y,cv=4);b=cross_val_score(hw,x,y,cv=4)
  call require(maxval(abs(a-b))<1.e-12_dp,'Huber CV ignores fitted caller and warm starts')
  call require(all(hw%coef_==before).and..not.allocated(hc%coef_),'Huber CV caller state')
  print *,'Fresh CV folds, warm-start independence and fitting weights: PASS'
end program
