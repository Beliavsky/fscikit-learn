program reference_joint_forest_invalid_driver
 use sklearn_ensemble,only:random_forest_regressor,extra_trees_regressor
 use sklearn_common,only:dp,require
 use,intrinsic::ieee_arithmetic
 implicit none
 type(random_forest_regressor)::model
 type(extra_trees_regressor)::extra
 real(dp)::x(4,2),y(4,2),bad,inf,score
 real(dp),allocatable::prediction(:)
 integer,allocatable::samples(:,:)
 character(40)::name
 x=1;y=1;bad=ieee_value(0._dp,ieee_quiet_nan);inf=ieee_value(0._dp,ieee_positive_inf)
 model%n_estimators=2;extra%n_estimators=2
 call get_command_argument(1,name)
 select case(trim(name))
 case('empty_outputs');call model%fit(x,y(:,:0))
 case('extra_empty_outputs');call extra%fit(x,y(:,:0))
 case('target_rows');call model%fit(x,y(:3,:))
 case('target_nan');y(1,2)=bad;call model%fit(x,y)
 case('target_inf');y(1,2)=inf;call model%fit(x,y)
 case('weights_length');call model%fit(x,y,[1._dp])
 case('weights_nan');call model%fit(x,y,[bad,1._dp,1._dp,1._dp])
 case('weights_inf');call model%fit(x,y,[inf,1._dp,1._dp,1._dp])
 case('poisson_negative');model%criterion='poisson';y(1,2)=-1;call model%fit(x,y)
 case('poisson_zero');model%criterion='poisson';y=0;call model%fit(x,y)
 case('score_outputs');call model%fit(x,y);score=model%score(x,y(:,:1))
 case('oob_no_bootstrap');model%bootstrap=.false.;model%oob_score=.true.;call model%fit(x,y)
 case('extra_oob_default');extra%oob_score=.true.;call extra%fit(x,y)
 case('max_samples_no_bootstrap');model%bootstrap=.false.;model%max_samples=2;call model%fit(x,y)
 case('max_samples_range');model%max_samples=5;call model%fit(x,y)
 case('scalar_predict');call model%fit(x,y);prediction=model%predict(x)
 case('samples_unfitted');samples=model%estimators_samples()
 case default;call require(.false.,'unknown joint forest invalid case')
 end select
 call require(.false.,'invalid joint forest input was accepted')
end program
