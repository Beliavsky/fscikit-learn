program reference_forest_warm_invalid_driver
 use sklearn_common,only:dp,require
 use sklearn_ensemble,only:random_forest_classifier,extra_trees_classifier,random_forest_regressor,extra_trees_regressor
 use,intrinsic::ieee_arithmetic
 implicit none
 character(40)::kind,name,argument
 integer::extra
 call get_command_argument(1,kind);call get_command_argument(2,argument);read(argument,*)extra
 call get_command_argument(3,name)
 if(kind=='classification')then
  call classification()
 else
  call regression()
 end if
 call require(.false.,'invalid warm forest input was accepted')
contains
 subroutine classification()
  class(random_forest_classifier),allocatable::model
  real(dp)::x(8,2),w(8),bad,inf
  integer::y(8,2)
  integer,allocatable::pred(:,:)
  integer::i
  if(extra==0)then
   allocate(random_forest_classifier::model)
  else
   allocate(extra_trees_classifier::model)
  end if
  do i=1,8
   x(i,:)=[real(i,dp),real(mod(i,3),dp)]
   y(i,:)=[mod(i,2),mod(i/2,2)]
  end do
  w=1;bad=ieee_value(0._dp,ieee_quiet_nan);inf=ieee_value(0._dp,ieee_positive_inf)
  model%warm_start=.true.;model%n_estimators=3;model%random_state=7;call model%fit(x,y)
  select case(trim(name))
  case('shrink');model%n_estimators=2;call model%fit(x,y)
  case('zero');model%n_estimators=0;call model%fit(x,y)
  case('criterion');model%criterion='invalid';call model%fit(x,y)
  case('depth');model%max_depth=-1;call model%fit(x,y)
  case('leaf');model%min_samples_leaf=0;call model%fit(x,y)
  case('split');model%min_samples_split=1;call model%fit(x,y)
  case('budget');model%max_leaf_nodes=1;call model%fit(x,y)
  case('features');model%max_features=-1;call model%fit(x,y)
  case('decrease');model%min_impurity_decrease=-.1_dp;call model%fit(x,y)
  case('alpha');model%ccp_alpha=-.1_dp;call model%fit(x,y)
  case('fraction');model%min_weight_fraction_leaf=.6_dp;call model%fit(x,y)
  case('max_samples_no_bootstrap');model%bootstrap=.false.;model%max_samples=2;call model%fit(x,y)
  case('samples_zero');call model%set_bootstrap(.true.);call model%set_max_samples(0._dp);call model%fit(x,y)
  case('samples_negative');call model%set_bootstrap(.true.);call model%set_max_samples(-.1_dp);call model%fit(x,y)
  case('samples_large');call model%set_bootstrap(.true.);call model%set_max_samples(1.01_dp);call model%fit(x,y)
  case('samples_nan');call model%set_bootstrap(.true.);call model%set_max_samples(bad);call model%fit(x,y)
  case('samples_inf');call model%set_bootstrap(.true.);call model%set_max_samples(inf);call model%fit(x,y)
  case('samples_no_bootstrap');call model%set_bootstrap(.false.);call model%set_max_samples(.5_dp);call model%fit(x,y)
  case('oob_no_bootstrap');model%bootstrap=.false.;model%oob_score=.true.;call model%fit(x,y)
  case('weights_length');call model%fit(x,y,[1._dp])
  case('weights_nan');w(1)=bad;call model%fit(x,y,w)
  case('input_inf');x(1,1)=inf;call model%fit(x,y)
  case('feature_count');call model%fit(reshape([(real(i,dp),i=1,24)],[8,3]),y)
  case('output_count');call model%fit(x,y(:,1))
  case('class_width_shrink');y=0;call model%fit(x,y);pred=model%predict_multi(x)
  case default;call require(.false.,'unknown warm forest guard')
  end select
 end subroutine classification
 subroutine regression()
  class(random_forest_regressor),allocatable::model
  real(dp)::x(8,2),w(8),bad,inf
  real(dp)::y(8,2)
  real(dp),allocatable::pred(:,:)
  integer::i
  if(extra==0)then
   allocate(random_forest_regressor::model)
  else
   allocate(extra_trees_regressor::model)
  end if
  do i=1,8
   x(i,:)=[real(i,dp),real(mod(i,3),dp)]
   y(i,:)=[real(i,dp),real(9-i,dp)]
  end do
  w=1;bad=ieee_value(0._dp,ieee_quiet_nan);inf=ieee_value(0._dp,ieee_positive_inf)
  model%warm_start=.true.;model%n_estimators=3;model%random_state=7;call model%fit(x,y)
  select case(trim(name))
  case('shrink');model%n_estimators=2;call model%fit(x,y)
  case('zero');model%n_estimators=0;call model%fit(x,y)
  case('criterion');model%criterion='invalid';call model%fit(x,y)
  case('depth');model%max_depth=-1;call model%fit(x,y)
  case('leaf');model%min_samples_leaf=0;call model%fit(x,y)
  case('split');model%min_samples_split=1;call model%fit(x,y)
  case('budget');model%max_leaf_nodes=1;call model%fit(x,y)
  case('features');model%max_features=-1;call model%fit(x,y)
  case('decrease');model%min_impurity_decrease=-.1_dp;call model%fit(x,y)
  case('alpha');model%ccp_alpha=-.1_dp;call model%fit(x,y)
  case('fraction');model%min_weight_fraction_leaf=.6_dp;call model%fit(x,y)
  case('max_samples_no_bootstrap');model%bootstrap=.false.;model%max_samples=2;call model%fit(x,y)
  case('samples_zero');call model%set_bootstrap(.true.);call model%set_max_samples(0._dp);call model%fit(x,y)
  case('samples_negative');call model%set_bootstrap(.true.);call model%set_max_samples(-.1_dp);call model%fit(x,y)
  case('samples_large');call model%set_bootstrap(.true.);call model%set_max_samples(1.01_dp);call model%fit(x,y)
  case('samples_nan');call model%set_bootstrap(.true.);call model%set_max_samples(bad);call model%fit(x,y)
  case('samples_inf');call model%set_bootstrap(.true.);call model%set_max_samples(inf);call model%fit(x,y)
  case('samples_no_bootstrap');call model%set_bootstrap(.false.);call model%set_max_samples(.5_dp);call model%fit(x,y)
  case('oob_no_bootstrap');model%bootstrap=.false.;model%oob_score=.true.;call model%fit(x,y)
  case('weights_length');call model%fit(x,y,[1._dp])
  case('weights_nan');w(1)=bad;call model%fit(x,y,w)
  case('input_inf');x(1,1)=inf;call model%fit(x,y)
  case('feature_count');call model%fit(reshape([(real(i,dp),i=1,24)],[8,3]),y)
  case('output_count');call model%fit(x,y(:,1))
  case('poisson_negative');model%criterion='poisson';y(1,1)=-1;call model%fit(x,y)
  case('poisson_zero');model%criterion='poisson';y=0;call model%fit(x,y)
  case default;call require(.false.,'unknown warm forest guard')
  end select
 end subroutine regression
end program
