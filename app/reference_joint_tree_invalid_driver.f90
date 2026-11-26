program reference_joint_tree_invalid_driver
 use sklearn_tree,only:decision_tree_regressor,extra_tree_regressor
 use sklearn_common,only:dp,require
 use,intrinsic::ieee_arithmetic
 implicit none
 type(decision_tree_regressor)::model
 type(extra_tree_regressor)::extra
 real(dp)::x(4,2),y(4,2),bad,inf,score
 real(dp),allocatable::prediction(:)
 character(40)::name
 x=1;y=1;bad=ieee_value(0._dp,ieee_quiet_nan);inf=ieee_value(0._dp,ieee_positive_inf)
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
 case('scalar_predict');call model%fit(x,y);prediction=model%predict(x)
 case default;call require(.false.,'unknown joint tree invalid case')
 end select
 call require(.false.,'invalid joint tree input was accepted')
end program
