program reference_gb_early_invalid_driver
 use sklearn_common,only:dp
 use sklearn_ensemble,only:gradient_boosting_regressor,gradient_boosting_classifier
 use sklearn_search_adapters,only:set_search_params,fit_search_native
 use sklearn_parameter_space,only:parameter_set,param
 use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan,ieee_positive_inf
 implicit none
 class(*),allocatable::model
 type(parameter_set)::params
 real(dp),allocatable::x(:,:),y(:),w(:)
 integer,allocatable::labels(:)
 character(32)::kind,case
 integer::i,n
 call get_command_argument(1,kind);call get_command_argument(2,case)
 if(trim(kind)=='regression')then
  allocate(gradient_boosting_regressor::model)
 else
  allocate(gradient_boosting_classifier::model)
 end if
 allocate(params%items(4))
 params%items(1)%name='n_iter_no_change';params%items(1)%value=param(2)
 params%items(2)%name='validation_fraction';params%items(2)%value=param(.2_dp)
 params%items(3)%name='tol';params%items(3)%value=param(1.e-4_dp)
 params%items(4)%name='n_estimators';params%items(4)%value=param(6)
 n=20
 select case(trim(case))
 case('patience_negative');params%items(1)%value=param(-1)
 case('disabled');params%items(1)%value=param(0)
 case('fraction_zero');params%items(2)%value=param(0._dp)
 case('fraction_one');params%items(2)%value=param(1._dp)
 case('fraction_negative');params%items(2)%value=param(-.1_dp)
 case('fraction_nan');params%items(2)%value=param(ieee_value(0._dp,ieee_quiet_nan))
 case('fraction_inf');params%items(2)%value=param(ieee_value(0._dp,ieee_positive_inf))
 case('tol_negative');params%items(3)%value=param(-.1_dp)
 case('tol_nan');params%items(3)%value=param(ieee_value(0._dp,ieee_quiet_nan))
 case('tol_inf');params%items(3)%value=param(ieee_value(0._dp,ieee_positive_inf))
 case('disabled_fraction');params%items(1)%value=param(0);params%items(2)%value=param(1._dp)
 case('disabled_tol');params%items(1)%value=param(0);params%items(3)%value=param(-1._dp)
 case('empty_train');params%items(2)%value=param(.99_dp)
 case('missing_classes');n=100;params%items(2)%value=param(.9_dp)
 case('small_test');params%items(2)%value=param(.01_dp)
 end select
 allocate(x(n,2),y(n),w(n),labels(n))
 do i=1,n;x(i,:)=[real(i,dp),real(mod(i,3),dp)];y(i)=real(i*i,dp);labels(i)=mod(i,2);end do
 w=1
 select case(trim(case))
 case('singleton_classes');labels=[(i,i=1,n)]
 case('one_class');labels=0
 case('missing_classes');labels=2;labels(:2)=0;labels(3:4)=1
 case('weights_zero');w=0
 case('weights_nan');w(1)=ieee_value(0._dp,ieee_quiet_nan)
 case('weights_inf');w(1)=ieee_value(0._dp,ieee_positive_inf)
 end select
 call set_search_params(model,params)
 if(trim(kind)=='regression')then
  call fit_search_native(model,x,yr=y,sample_weight=w)
 else
  call fit_search_native(model,x,yi=labels,sample_weight=w)
 end if
 print *,'FIT ACCEPTED'
end program
