program test_linear_svm
  use sklearn, only: dp,linear_svc,linear_svr
  use sklearn_common, only: require
  implicit none
  type(linear_svc) :: classifier,weighted
  type(linear_svr) :: reg,repeat
  real(dp) :: pair(2,1),zero(2,1),x(4,1),duplicated(3,1),three(3,1)
  real(dp), allocatable :: first(:)
  reg%loss='squared_epsilon_insensitive';reg%dual='false'
  repeat%loss='squared_epsilon_insensitive';repeat%dual='false'
  pair(:,1)=[-1._dp,1._dp]; zero=0
  classifier%fit_intercept=.false.; classifier%tol=1.e-12_dp
  call classifier%fit(pair,[-3,9])
  call require(abs(classifier%coef_(1,1)-.8_dp)<1.e-13_dp,'analytic squared-hinge optimum')
  call require(abs(classifier%objective_(1)-.4_dp)<1.e-13_dp.and.classifier%converged_,'classification objective and convergence')
  reg%fit_intercept=.false.; reg%tol=1.e-12_dp; reg%epsilon=.5_dp
  call reg%fit(pair,[-2._dp,2._dp])
  call require(abs(reg%coef_(1)-1.2_dp)<1.e-13_dp.and.abs(reg%objective_-.9_dp)<1.e-13_dp, &
    'analytic squared epsilon-insensitive optimum')
  reg%epsilon=0; call reg%fit(pair,[-2._dp,2._dp])
  call require(abs(reg%coef_(1)-1.6_dp)<1.e-13_dp,'zero epsilon gives penalized least squares')
  weighted%tol=1.e-12_dp; weighted%intercept_scaling=2
  call weighted%fit(zero,[-3,9],sample_weight=[1._dp,3._dp])
  call require(abs(weighted%intercept_(1)-16._dp/33)<1.e-13_dp,'scaled intercept is regularized')
  first=weighted%intercept_; call weighted%fit(zero,[-3,9],class_weight=[1._dp,3._dp])
  call require(maxval(abs(first-weighted%intercept_))<1.e-13_dp,'binary class and sample weights agree')
  reg%fit_intercept=.true.; reg%intercept_scaling=2
  call reg%fit(zero,[2._dp,2._dp])
  call require(abs(reg%intercept_-32._dp/17)<1.e-13_dp,'regression intercept penalty uses synthetic feature')
  reg%intercept_scaling=1; call reg%fit(pair,[-2._dp,2._dp],sample_weight=[2._dp,1._dp])
  duplicated(:,1)=[-1._dp,-1._dp,1._dp]; repeat%tol=1.e-12_dp
  call repeat%fit(duplicated,[-2._dp,-2._dp,2._dp])
  call require(maxval(abs(reg%coef_-repeat%coef_))<1.e-13_dp.and.abs(reg%intercept_-repeat%intercept_)<1.e-13_dp, &
    'integer sample weights equal repeated observations')
  x(:,1)=[1._dp,2._dp,3._dp,4._dp]; classifier%fit_intercept=.true.; classifier%c=5; classifier%max_iter=1
  call classifier%fit(x,[-3,-3,9,9])
  call require(.not.classifier%converged_.and.classifier%n_iter_==1,'iteration cap reports nonconvergence')
  three=0; classifier%fit_intercept=.false.; call classifier%fit(three,[-4,7,21])
  call require(all(classifier%predict(three)==-4).and.classifier%converged_,'multiclass zero ties choose first label')
  reg%fit_intercept=.false.; call reg%fit(zero,[0._dp,0._dp])
  call require(reg%n_iter_==0.and.reg%converged_.and.reg%objective_==0,'stationary initial solution needs no update')
  print *, 'Linear SVC and SVR tests passed'
end program
