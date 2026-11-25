program test_anova
  use sklearn_common,only:dp,require
  use sklearn_univariate_selection,only:f_oneway,anova_sample,anova_result
  use sklearn_sparse,only:csr_matrix,csr_from_dense,csr_from_arrays
  use,intrinsic::ieee_arithmetic
  use,intrinsic::ieee_exceptions
  implicit none
  type(anova_result)::test,copy
  type(anova_sample)::samples(2)
  type(csr_matrix)::a,b
  real(dp)::x(3,1),y(3,1)
  logical::modes(size(ieee_all)),flags(size(ieee_all)),now(size(ieee_all)),expected(size(ieee_all)),after_modes(size(ieee_all))
  x(:,1)=[1._dp,2._dp,3._dp];y(:,1)=[2._dp,3._dp,4._dp]
  test=f_oneway(x,y)
  call require(abs(test%statistic(1)-1.5_dp)<1.e-14_dp,'analytic ANOVA statistic')
  call require(abs(test%pvalue(1)-.2878641347266908_dp)<1.e-13_dp,'analytic F survival probability')
  samples=[anova_sample(x),anova_sample(y)];x=0
  copy=f_oneway(samples)
  call require(all(copy%statistic==test%statistic),'sample constructors own data')
  a=csr_from_dense(y-1);b=csr_from_dense(y)
  copy=f_oneway(a,b)
  call require(all(copy%statistic==test%statistic),'dense and CSR groups agree')
  a=csr_from_arrays(2,1,[1,3,3],[1,1],[1._dp,2._dp])
  b=csr_from_arrays(2,1,[1,2,3],[1,1],[3._dp,4._dp])
  test=f_oneway(a,b)
  call require(abs(test%statistic(1)-8._dp)<1.e-14_dp,'raw duplicate squares follow safe_sqr')
  call ieee_get_flag(ieee_all,flags);call ieee_get_halting_mode(ieee_all,modes)
  call ieee_set_flag(ieee_underflow,.true.)
  call ieee_get_flag(ieee_all,expected)
  x=1;y=2;test=f_oneway(x,y)
  call require(.not.ieee_is_finite(test%statistic(1)).and.test%statistic(1)>0,'separate constants yield infinity')
  call require(test%pvalue(1)==0,'infinite F has zero pvalue')
  call ieee_get_flag(ieee_all,now)
  call ieee_get_halting_mode(ieee_all,after_modes)
  call require(all(now.eqv.expected).and.all(modes.eqv.after_modes),'caller IEEE flags and modes retained')
  test=f_oneway(x,x)
  call require(ieee_is_nan(test%statistic(1)).and.ieee_is_nan(test%pvalue(1)),'constant ANOVA remains undefined')
  call ieee_set_flag(ieee_all,flags);call ieee_set_halting_mode(ieee_all,modes)
  print *,'ANOVA dense, sparse, duplicate, ownership and IEEE cases passed'
end program
