program reference_gb_invalid_driver
 use sklearn_common,only:dp,require
 use sklearn_ensemble,only:gradient_boosting_regressor,gradient_boosting_classifier
 use,intrinsic::ieee_arithmetic
 use,intrinsic::iso_fortran_env,only:real32
 implicit none
 character(40)::kind,name
 call get_command_argument(1,kind);call get_command_argument(2,name)
 if(kind=='regression')then
  call regression()
 else
  call classification()
 end if
 call require(.false.,'bad gradient input accepted')
contains
 subroutine regression()
  type(gradient_boosting_regressor)::model
  real(dp)::x(8,3),w(8),bad,inf
  real(dp)::y(8)
  real(dp),allocatable::pred(:)
  integer::i
  do i=1,8
   x(i,:)=[real(i,dp),real(mod(i,3),dp),real(i*i,dp)];y(i)=real(i,dp)
  end do
  w=1;bad=ieee_value(0._dp,ieee_quiet_nan);inf=ieee_value(0._dp,ieee_positive_inf)
  if(name=='unfit')then
   pred=model%predict(x);return
  end if
  model%n_estimators=2;call model%fit(x,y)
  write(*,'(a)')'BASE_FIT_PASSED'
  select case(trim(name))
  case('count');model%n_estimators=0;call model%fit(x,y)
  case('warm_count');model%warm_start=.true.;model%n_estimators=1;call model%fit(x,y)
  case('warm_width');model%warm_start=.true.;call model%fit(x(:,:2),y)
  case('subsample_zero');model%subsample=0;call model%fit(x,y)
  case('subsample_negative');model%subsample=-.1_dp;call model%fit(x,y)
  case('subsample_large');model%subsample=1.01_dp;call model%fit(x,y)
  case('subsample_nan');model%subsample=bad;call model%fit(x,y)
  case('subsample_inf');model%subsample=inf;call model%fit(x,y)
  case('depth');model%max_depth=0;call model%fit(x,y)
  case('leaf');model%min_samples_leaf=0;call model%fit(x,y)
  case('split');model%min_samples_split=1;call model%fit(x,y)
  case('seed');model%random_state=-1;call model%fit(x,y)
  case('rate_negative');model%learning_rate=-.1_dp;call model%fit(x,y)
  case('rate_nan');model%learning_rate=bad;call model%fit(x,y)
  case('rate_inf');model%learning_rate=inf;call model%fit(x,y)
  case('decrease');model%min_impurity_decrease=-.1_dp;call model%fit(x,y)
  case('criterion');model%criterion='invalid';call model%fit(x,y)
  case('loss');model%loss='invalid';call model%fit(x,y)
  case('feature_integer');call model%set_max_features(-1);call model%fit(x,y)
  case('feature_zero');call model%set_max_features(0._dp);call model%fit(x,y)
  case('feature_large');call model%set_max_features(1.01_dp);call model%fit(x,y)
  case('feature_nan');call model%set_max_features(bad);call model%fit(x,y)
  case('feature_inf');call model%set_max_features(inf);call model%fit(x,y)
  case('feature_name');call model%set_max_features('invalid');call model%fit(x,y)
  case('input_nan');x(1,1)=bad;call model%fit(x,y)
  case('input_inf');x(1,1)=inf;call model%fit(x,y)
  case('float32_overflow');x(1,1)=2*real(huge(0._real32),dp);call model%fit(x,y)
  case('query_nan');x(1,1)=bad;pred=model%predict(x)
  case('query_inf');x(1,1)=inf;pred=model%predict(x)
  case('query_width');pred=model%predict(x(:,:2))
  case('target_size');call model%fit(x,y(:7))
  case('alpha_zero');model%alpha=0;call model%fit(x,y)
  case('alpha_one');model%alpha=1;call model%fit(x,y)
  case('alpha_nan');model%alpha=bad;call model%fit(x,y)
  case('alpha_inf');model%alpha=inf;call model%fit(x,y)
  case('budget');model%max_leaf_nodes=1;call model%fit(x,y)
  case('ccp');model%ccp_alpha=-.1_dp;call model%fit(x,y)
  case('weight_fraction');model%min_weight_fraction_leaf=.6_dp;call model%fit(x,y)
  case('weights_shape');call model%fit(x,y,[1._dp])
  case('weights_zero');w=0;call model%fit(x,y,w)
  case('weights_nan');w(1)=bad;call model%fit(x,y,w)
  case('weights_inf');w(1)=inf;call model%fit(x,y,w)
  case('target_nan');y(1)=bad;call model%fit(x,y)
  case('target_inf');y(1)=inf;call model%fit(x,y)
  end select
 end subroutine regression
 subroutine classification()
  type(gradient_boosting_classifier)::model
  real(dp)::x(8,3),w(8),bad,inf
  integer::y(8)
  integer,allocatable::pred(:)
  integer::i
  do i=1,8
   x(i,:)=[real(i,dp),real(mod(i,3),dp),real(i*i,dp)];y(i)=mod(i,2)
  end do
  w=1;bad=ieee_value(0._dp,ieee_quiet_nan);inf=ieee_value(0._dp,ieee_positive_inf)
  if(name=='unfit')then
   pred=model%predict(x);return
  end if
  model%n_estimators=2;call model%fit(x,y)
  write(*,'(a)')'BASE_FIT_PASSED'
  select case(trim(name))
  case('count');model%n_estimators=0;call model%fit(x,y)
  case('warm_count');model%warm_start=.true.;model%n_estimators=1;call model%fit(x,y)
  case('warm_width');model%warm_start=.true.;call model%fit(x(:,:2),y)
  case('warm_classes');model%warm_start=.true.;model%n_estimators=3;y=[(mod(i,3),i=1,8)];call model%fit(x,y)
  case('subsample_zero');model%subsample=0;call model%fit(x,y)
  case('subsample_negative');model%subsample=-.1_dp;call model%fit(x,y)
  case('subsample_large');model%subsample=1.01_dp;call model%fit(x,y)
  case('subsample_nan');model%subsample=bad;call model%fit(x,y)
  case('subsample_inf');model%subsample=inf;call model%fit(x,y)
  case('depth');model%max_depth=0;call model%fit(x,y)
  case('leaf');model%min_samples_leaf=0;call model%fit(x,y)
  case('split');model%min_samples_split=1;call model%fit(x,y)
  case('seed');model%random_state=-1;call model%fit(x,y)
  case('rate_negative');model%learning_rate=-.1_dp;call model%fit(x,y)
  case('rate_nan');model%learning_rate=bad;call model%fit(x,y)
  case('rate_inf');model%learning_rate=inf;call model%fit(x,y)
  case('decrease');model%min_impurity_decrease=-.1_dp;call model%fit(x,y)
  case('criterion');model%criterion='invalid';call model%fit(x,y)
  case('loss');model%loss='invalid';call model%fit(x,y)
  case('feature_integer');call model%set_max_features(-1);call model%fit(x,y)
  case('feature_zero');call model%set_max_features(0._dp);call model%fit(x,y)
  case('feature_large');call model%set_max_features(1.01_dp);call model%fit(x,y)
  case('feature_nan');call model%set_max_features(bad);call model%fit(x,y)
  case('feature_inf');call model%set_max_features(inf);call model%fit(x,y)
  case('feature_name');call model%set_max_features('invalid');call model%fit(x,y)
  case('input_nan');x(1,1)=bad;call model%fit(x,y)
  case('input_inf');x(1,1)=inf;call model%fit(x,y)
  case('float32_overflow');x(1,1)=2*real(huge(0._real32),dp);call model%fit(x,y)
  case('query_nan');x(1,1)=bad;pred=model%predict(x)
  case('query_inf');x(1,1)=inf;pred=model%predict(x)
  case('query_width');pred=model%predict(x(:,:2))
  case('target_size');call model%fit(x,y(:7))
  case('one_class');y=0;call model%fit(x,y)
  case('exponential_multiclass');model%loss='exponential';y(1)=99;call model%fit(x,y)
  case('budget');model%max_leaf_nodes=1;call model%fit(x,y)
  case('ccp');model%ccp_alpha=-.1_dp;call model%fit(x,y)
  case('weight_fraction');model%min_weight_fraction_leaf=.6_dp;call model%fit(x,y)
  case('weights_shape');call model%fit(x,y,[1._dp])
  case('weights_zero');w=0;call model%fit(x,y,w)
  case('weights_nan');w(1)=bad;call model%fit(x,y,w)
  case('weights_inf');w(1)=inf;call model%fit(x,y,w)
  case('weights_one_class');where(y==1)w=0;call model%fit(x,y,w)
  case('weights_cancel');where(y==1)w=-1;call model%fit(x,y,w)
  end select
 end subroutine classification
end program
