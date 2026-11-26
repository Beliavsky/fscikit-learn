program test_qda_svd
 use sklearn_common,only:dp,require
 use sklearn_discriminant_analysis,only:quadratic_discriminant_analysis,linear_discriminant_analysis
 use,intrinsic::ieee_arithmetic
 use,intrinsic::ieee_exceptions
 implicit none
 type(quadratic_discriminant_analysis)::model
 type(linear_discriminant_analysis)::linear
 real(dp)::x(12,2),zero(6,2),wide(4,6),identity(2,2),query(1,2),direction(2)
 real(dp),allocatable::scores(:,:),prob(:,:),before(:,:)
 integer::y(12),i,c
 logical::modes(size(ieee_all)),flags(size(ieee_all)),after_modes(size(ieee_all)),after_flags(size(ieee_all))
 do i=1,12
  y(i)=merge(-3,7,i<=6);x(i,:)=[sin(real(i,dp)),cos(2._dp*i)]+merge(0._dp,2._dp,i<=6)
 end do
 model%reg_param=.2_dp
 call ieee_set_flag(ieee_all,.false.);call ieee_get_flag(ieee_all,flags);call ieee_get_halting_mode(ieee_all,modes)
 call model%fit(x,y)
 call ieee_get_flag(ieee_all,after_flags);call ieee_get_halting_mode(ieee_all,after_modes)
 call require(all(flags.eqv.after_flags).and.all(modes.eqv.after_modes),'QDA fit preserves IEEE state')
 call require(model%n_features_in_==2.and.size(model%subspaces_)==2,'QDA factor metadata')
 do c=1,2
  identity=matmul(transpose(model%subspaces_(c)%rotations),model%subspaces_(c)%rotations)
  identity(1,1)=identity(1,1)-1;identity(2,2)=identity(2,2)-1
  call require(maxval(abs(identity))<1.e-12_dp,'QDA rotations orthonormal')
 end do
 before=model%decision_function(x);model%tol=1.e9_dp;call model%fit(x,y)
 call require(all([(model%subspaces_(c)%rank==0,c=1,2)]),'tol controls rank diagnostics')
 call require(maxval(abs(before-model%decision_function(x)))==0,'tol does not truncate scoring axes')
 model%priors=[.25_dp];call model%fit(x,y);prob=model%predict_proba(x)
 call require(size(model%priors_)==1.and.maxval(abs(sum(prob,dim=2)-1))<1.e-12_dp,'QDA broadcasts scalar prior')
 model%priors=[-1._dp,1._dp];call model%fit(x,y);scores=model%predict_log_proba(x)
 call require(all(ieee_is_nan(scores)).and.all(model%predict(x)==-3),'negative prior follows Source NaN prediction')
 model%priors=[0._dp,0._dp];call model%fit(x,y)
 call require(all(ieee_is_nan(model%predict_proba(x))),'zero prior posterior is undefined')
 deallocate(model%priors);model%reg_param=0;zero=0
 call model%fit(zero,[-3,-3,-3,7,7,7])
 call ieee_set_flag(ieee_all,.false.);scores=model%predict_log_proba(zero)
 call ieee_get_flag(ieee_all,after_flags);call ieee_get_halting_mode(ieee_all,after_modes)
 call require(all(ieee_is_nan(scores)).and.all(model%predict(zero)==-3),'singular unregularized QDA remains fitted')
 call require(all(flags.eqv.after_flags).and.all(modes.eqv.after_modes),'QDA posterior preserves IEEE state')
 do i=1,6
  wide(:,i)=[-1._dp,1._dp,2._dp,4._dp]*i
 end do
 call model%fit(wide,[-3,-3,7,7])
 call require(size(model%subspaces_(1)%scalings)==2.and.size(model%subspaces_(1)%rotations,1)==6, &
  'unregularized wide QDA retains thin SVD axes')
 linear%shrinkage=.2_dp;call linear%fit(x,y)
 call require(linear%n_features_in_==2,'LDA inherited feature metadata')
 call require(maxval(abs(exp(linear%predict_log_proba(x))-linear%predict_proba(x)))<1.e-14_dp,'LDA inherited log probabilities')
 linear%priors=[1._dp,0._dp];call linear%fit(x,y);scores=linear%predict_log_proba(x)
 call require(all(ieee_is_finite(scores)).and.maxval(abs(scores(:,2)-log(tiny(1._dp))))<1.e-12_dp, &
  'LDA log probabilities clip exact zero to the smallest normal float')
 deallocate(linear%priors);call linear%fit(x,y)
 direction=linear%coef_(2,:)-linear%coef_(1,:)
 query(1,:)=direction*(50-(linear%intercept_(2)-linear%intercept_(1)))/sum(direction**2)
 prob=linear%predict_proba(query)
 call require(prob(1,1)==0.and.prob(1,2)==1,'binary LDA uses sigmoid/complement rounding')
 print *,'QDA SVD, singular fits, priors, rank diagnostics, IEEE state and LDA regression passed'
end program
