program test_svc_weighted
  use sklearn_svm,only:svc
  use sklearn_common,only:dp,require
  use sklearn_base_dispatch,only:base_weighted,train_classifier
  use sklearn_search_adapters,only:set_search_params
  use sklearn_parameter_space,only:parameter_set,parameter_item,param
  use sklearn_multiclass,only:one_vs_rest_classifier
  use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan
  implicit none
  type(svc)::model
  type(parameter_set)::parameters
  type(one_vs_rest_classifier)::ovr
  real(dp)::pair(2,1),three(3,1),zero(4,1),query(1,1)
  real(dp),allocatable::scores(:,:),coef(:,:)
  integer,allocatable::labels(:),before(:),after(:)
  integer::n
  character(32)::mode
  pair(:,1)=[-1._dp,1._dp];three(:,1)=[-4._dp,-1._dp,1._dp];zero=0;query=0
  model%kernel='linear';model%tol=1.e-9_dp;call get_command_argument(1,mode)
  select case(trim(mode))
  case('weight_count');call model%fit(pair,[-3,9],[1._dp])
  case('weight_negative');call model%fit(pair,[-3,9],[-1._dp,1._dp])
  case('weight_nan');call model%fit(pair,[-3,9],[ieee_value(0._dp,ieee_quiet_nan),1._dp])
  case('weight_zero');call model%fit(pair,[-3,9],[0._dp,0._dp])
  case('class_count');call model%fit(pair,[-3,9],class_weight=[1._dp])
  case('class_zero');call model%fit(pair,[-3,9],class_weight=[1._dp,0._dp])
  case('class_nan');call model%fit(pair,[-3,9],class_weight=[1._dp,ieee_value(0._dp,ieee_quiet_nan)])
  case('class_conflict');model%class_weight_balanced=.true.;call model%fit(pair,[-3,9],class_weight=[1._dp,1._dp])
  case('c');model%c=0;call model%fit(pair,[-3,9])
  case('tol');model%tol=0;call model%fit(pair,[-3,9])
  case('gamma');model%gamma=-2;call model%fit(pair,[-3,9])
  case('gamma_rule');model%gamma_rule='bad';call model%fit(pair,[-3,9])
  case('degree');model%degree=-1;call model%fit(pair,[-3,9])
  case('kernel');model%kernel='bad';call model%fit(pair,[-3,9])
  case('decision');model%decision_function_shape='bad';call model%fit(pair,[-3,9])
  case('unfitted');labels=model%predict(pair)
  case('query_width');call model%fit(pair,[-3,9]);scores=model%decision_function(reshape([1._dp,2._dp],[1,2]))
  case('query_nan');call model%fit(pair,[-3,9]);query=ieee_value(0._dp,ieee_quiet_nan);labels=model%predict(query)
  case('tie_ovo');call model%fit(pair,[-3,9]);model%break_ties=.true.;model%decision_function_shape='ovo';labels=model%predict(pair)
  end select
  call require(len_trim(mode)==0,'weighted SVC rejection unexpectedly accepted input')
  call random_seed(size=n);allocate(before(n),after(n));call random_seed(get=before)
  call require(model%shrinking,"SVC defaults to LIBSVM shrinking")
  model%shrinking=.false.
  call model%fit(pair,[-3,9],[1._dp,.1_dp])
  call require(abs(model%coef_(1,1)-.2_dp)<1.e-12_dp.and.abs(model%intercept_(1)+.8_dp)<1.e-12_dp,'weighted two-point SVC optimum')
  coef=model%coef_;model%shrinking=.true.;call model%fit(pair,[-3,9],class_weight=[1._dp,.1_dp])
  call require(all(coef==model%coef_),'class and sample SVC capacities agree')
  call model%fit(three,[-3,-3,9],[0._dp,1._dp,1._dp])
  call require(all(model%support_==[1,2]).and.all(model%support_original_==[2,3]),'compressed and original support indices')
  call require(all(model%n_support_==[1,1]).and.model%n_features_in_==1,'SVC fitted support metadata')
  call require(abs(model%coef_(1,1)-1._dp)<1.e-12_dp.and.model%fit_status_==0,'unweighted positive-capacity solution')
  labels=model%predict(query);call require(labels(1)==9,'ordinary binary zero margin predicts upper class')
  model%max_iter=1;call model%fit(pair,[-3,9]);call require(model%fit_status_==1.and.model%n_iter_==1,'pair-update cap status')
  model%max_iter=10000;call model%fit(zero,[-4,7,21,33]);scores=model%decision_function(query)
  call require(all(shape(scores)==[1,4]).and.all(scores(1,:)==[3._dp,2._dp,1._dp,0._dp]),'OVR zero-margin vote convention')
  labels=model%predict(query);call require(labels(1)==33,'ordinary OVO tie votes choose upper pair classes')
  model%break_ties=.true.;labels=model%predict(query);call require(labels(1)==-4,'break_ties uses OVR confidence scores')
  model%break_ties=.false.;model%decision_function_shape='ovo';scores=model%decision_function(query)
  call require(all(shape(scores)==[1,6]).and.all(scores==0),'OVO decision shape and signs')
  model%kernel='rbf';call model%fit(pair,[-3,9]);call require(.not.allocated(model%coef_),'nonlinear refit clears linear coefficients')
  model%gamma_rule='auto';model%gamma=-1;call model%fit(pair,[-3,9]);call require(model%gamma_==1,'auto gamma')
  model%kernel='linear';call model%fit(pair,[-3,9]);call require(model%score(pair,[-3,9],[.1_dp,2._dp])==1,'weighted SVC score')
  call require(base_weighted(model),'weighted SVC ensemble dispatch')
  call train_classifier(model,pair,[-3,9],[1._dp,.1_dp])
  call require(abs(model%coef_(1,1)-.2_dp)<1.e-12_dp,'weighted SVC dispatch forwards costs')
  call ovr%fit(pair,[-3,9],model,sample_weight=[1._dp,.1_dp])
  select type(head=>ovr%estimators_(1)%estimator_)
  type is(svc)
    call require(abs(head%coef_(1,1)-.2_dp)<1.e-12_dp,'weighted multiclass SVC fit forwarding')
  class default
    call require(.false.,'OVR preserves SVC base type')
  end select
  parameters%items=[parameter_item('shrinking',param(.false.)),parameter_item('decision_function_shape',param('ovr')),parameter_item('break_ties',param(.true.)), &
    parameter_item('gamma_rule',param('scale'))]
  call set_search_params(model,parameters)
  call require(.not.model%shrinking.and.model%break_ties.and.model%decision_function_shape=='ovr'.and.model%gamma_rule=='scale','SVC search controls')
  call random_seed(get=after);call require(all(before==after),'SVC preserves intrinsic RNG')
  print *,'Weighted SVC capacities, support metadata, decision shapes and ties: PASS'
end program
