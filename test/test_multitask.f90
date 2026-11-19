program test_multitask
  use sklearn, only: dp,multi_task_elastic_net,multi_task_lasso
  use sklearn_common, only: require
  implicit none
  type(multi_task_lasso) :: lasso
  type(multi_task_elastic_net) :: net,cold
  real(dp) :: x(2,2),y(2,2),zero(2,2)
  real(dp), allocatable :: prediction(:,:),first(:,:)
  x=0; x(1,1)=1; y=0; y(1,:)=[3._dp,4._dp]; zero=0
  lasso%fit_intercept=.false.; lasso%alpha=.5_dp; lasso%tol=1.e-12_dp
  call lasso%fit(x,y)
  call require(maxval(abs(lasso%coef_(:,1)-[2.4_dp,3.2_dp]))<1.e-13_dp,'group soft threshold')
  call require(all(lasso%coef_(:,2)==0).and.lasso%converged_,'zero feature and dual convergence')
  net%fit_intercept=.false.; net%alpha=.5_dp; net%tol=1.e-12_dp
  call net%fit(x,y)
  call require(maxval(abs(net%coef_(:,1)-[1.8_dp,2.4_dp]))<1.e-13_dp,'mixed group and ridge penalty')
  net%warm_start=.true.; net%alpha=.2_dp; call net%fit(x,y)
  cold%fit_intercept=.false.; cold%alpha=.2_dp; cold%tol=1.e-12_dp; call cold%fit(x,y)
  call require(maxval(abs(net%coef_-cold%coef_))<1.e-13_dp,'warm and cold solutions agree')
  first=net%coef_; net%warm_start=.false.; call net%fit(x,y)
  call require(maxval(abs(first-net%coef_))<1.e-13_dp,'cold refit resets coefficients')
  lasso%alpha=3; call lasso%fit(x,y)
  call require(all(lasso%coef_==0),'group threshold removes whole feature')
  net%fit_intercept=.true.; call net%fit(zero,y); prediction=net%predict(zero)
  call require(all(net%coef_==0).and.maxval(abs(prediction(1,:)-[1.5_dp,2._dp]))<1.e-13_dp, &
    'constant design predicts target means')
  call net%fit(x,zero)
  call require(net%converged_.and.net%dual_gap_==0.and.net%score(x,zero)==1,'zero targets converge and score')
  print *, 'Multi-task regression tests passed'
end program
