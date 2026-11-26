program reference_feature_invalid_driver
 use sklearn_common,only:dp,require
 use sklearn_tree,only:decision_tree_regressor,extra_tree_regressor,decision_tree_classifier,extra_tree_classifier
 use sklearn_ensemble,only:random_forest_regressor,extra_trees_regressor,random_forest_classifier,extra_trees_classifier
 use,intrinsic::ieee_arithmetic
 implicit none
 class(*),allocatable::base
 character(64)::argument,name
 integer::family,mode,i
 real(dp)::x(9,3),yr(9,2),bad,inf
 integer::yi(9,2)
 call get_command_argument(1,argument);read(argument,*)family
 call get_command_argument(2,argument);read(argument,*)mode
 call get_command_argument(3,name)
 do i=1,9
  x(i,:)=[real(i,dp),real(mod(i,3),dp),real(mod(i,5),dp)]
  yr(i,:)=[real(i,dp),real(10-i,dp)];yi(i,:)=[mod(i,2),mod(i/2,2)]
 end do
 bad=ieee_value(0._dp,ieee_quiet_nan);inf=ieee_value(0._dp,ieee_positive_inf)
 select case(family*2+mode)
 case(0);allocate(random_forest_regressor::base)
 case(1);allocate(extra_trees_regressor::base)
 case(2);allocate(random_forest_classifier::base)
 case(3);allocate(extra_trees_classifier::base)
 case(4);allocate(decision_tree_regressor::base)
 case(5);allocate(extra_tree_regressor::base)
 case(6);allocate(decision_tree_classifier::base)
 case(7);allocate(extra_tree_classifier::base)
 end select
 select type(base)
 class is(random_forest_regressor)
  base%n_estimators=3;base%warm_start=.true.;call base%set_bootstrap(.true.)
  call base%fit(x,yr)
  select case(trim(name))
  case('zero');call base%set_max_features(0._dp)
  case('negative');call base%set_max_features(-.1_dp)
  case('large');call base%set_max_features(1.01_dp)
  case('nan');call base%set_max_features(bad)
  case('inf');call base%set_max_features(inf)
  case('negative_inf');call base%set_max_features(-inf)
  case('integer');call base%set_max_features(-1)
  case('name');call base%set_max_features('invalid')
  case('long_name');call base%set_max_features('sqrt            invalid suffix')
  end select
  call base%fit(x,yr)
 class is(random_forest_classifier)
  base%n_estimators=3;base%warm_start=.true.;call base%set_bootstrap(.true.)
  call base%fit(x,yi)
  select case(trim(name))
  case('zero');call base%set_max_features(0._dp)
  case('negative');call base%set_max_features(-.1_dp)
  case('large');call base%set_max_features(1.01_dp)
  case('nan');call base%set_max_features(bad)
  case('inf');call base%set_max_features(inf)
  case('negative_inf');call base%set_max_features(-inf)
  case('integer');call base%set_max_features(-1)
  case('name');call base%set_max_features('invalid')
  case('long_name');call base%set_max_features('sqrt            invalid suffix')
  end select
  call base%fit(x,yi)
 class is(decision_tree_regressor)
  call base%fit(x,yr)
  select case(trim(name))
  case('zero');call base%set_max_features(0._dp)
  case('negative');call base%set_max_features(-.1_dp)
  case('large');call base%set_max_features(1.01_dp)
  case('nan');call base%set_max_features(bad)
  case('inf');call base%set_max_features(inf)
  case('negative_inf');call base%set_max_features(-inf)
  case('integer');call base%set_max_features(-1)
  case('name');call base%set_max_features('invalid')
  case('long_name');call base%set_max_features('sqrt            invalid suffix')
  end select
  call base%fit(x,yr)
 class is(decision_tree_classifier)
  call base%fit(x,yi)
  select case(trim(name))
  case('zero');call base%set_max_features(0._dp)
  case('negative');call base%set_max_features(-.1_dp)
  case('large');call base%set_max_features(1.01_dp)
  case('nan');call base%set_max_features(bad)
  case('inf');call base%set_max_features(inf)
  case('negative_inf');call base%set_max_features(-inf)
  case('integer');call base%set_max_features(-1)
  case('name');call base%set_max_features('invalid')
  case('long_name');call base%set_max_features('sqrt            invalid suffix')
  end select
  call base%fit(x,yi)
 end select
 call require(.false.,'invalid feature control accepted')
end program
