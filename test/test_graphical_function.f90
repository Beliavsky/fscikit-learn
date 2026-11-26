program test_graphical_function
 use sklearn_covariance,only:graphical_lasso,graphical_lasso_result
 use sklearn_common,only:dp,require
 use,intrinsic::ieee_exceptions
 implicit none
 type(graphical_lasso_result)::fit
 type(graphical_lasso)::model
 real(dp)::cov(2,2),expected(2,2)
 logical::modes(size(ieee_all)),flags(size(ieee_all)),after_modes(size(ieee_all)),after_flags(size(ieee_all))
 cov=reshape([2._dp,.8_dp,.8_dp,1._dp],[2,2])
 call ieee_get_halting_mode(ieee_all,modes);call ieee_get_flag(ieee_all,flags)
 fit=graphical_lasso(cov,.3_dp,tol=1.e-12_dp,enet_tol=1.e-12_dp)
 call ieee_get_halting_mode(ieee_all,after_modes);call ieee_get_flag(ieee_all,after_flags)
 call require(all(modes.eqv.after_modes).and.all(flags.eqv.after_flags),'functional graphical lasso preserves IEEE state')
 expected=reshape([1._dp,-.5_dp,-.5_dp,2._dp],[2,2])/1.75_dp
 call require(maxval(abs(fit%precision-expected))<1.e-11_dp,'functional graphical lasso analytic precision')
 call require(abs(fit%covariance(1,2)-.5_dp)<1.e-12_dp,'functional covariance threshold')
 call require(fit%converged.and.size(fit%costs,1)==fit%n_iter,'owned costs and stopping metadata')
 fit%covariance(1,1)=99
 call require(cov(1,1)==2,'functional output owns storage independently of input')
 fit=graphical_lasso(cov,.3_dp,max_iter=0)
 call require(fit%n_iter==1.and.size(fit%costs,1)==0,'Source zero-iteration metadata')
 fit=graphical_lasso(cov,0._dp)
 call require(fit%n_iter==0.and.size(fit%costs,1)==1,'unregularized early return')
 call require(maxval(abs(matmul(cov,fit%precision)-reshape([1._dp,0._dp,0._dp,1._dp],[2,2])))<1.e-12_dp, &
  'unregularized inverse')
 model%covariance='precomputed';call model%fit(cov)
 call require(allocated(model%precision_),'derived type constructor and fit remain available')
 print *,'Functional graphical lasso analytic, zero-iteration and constructor checks passed'
end program
