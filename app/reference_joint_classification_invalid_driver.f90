program reference_joint_classification_invalid_driver
 use sklearn_tree,only:decision_tree_classifier,extra_tree_classifier,tree_probability
 use sklearn_common,only:dp,require
 use,intrinsic::ieee_arithmetic
 implicit none
 type(decision_tree_classifier)::model
 type(extra_tree_classifier)::extra
 real(dp)::x(4,2),bad,inf,score
 integer::y(4,2),indicator(4,2)
 integer,allocatable::pred(:,:),scalar(:)
 real(dp),allocatable::prob(:,:)
 character(40)::name
 x=1;y(:,1)=[-7,-7,11,11];y(:,2)=[0,1,2,0]
 indicator(:,1)=[0,0,1,1];indicator(:,2)=[0,1,0,1]
 bad=ieee_value(0._dp,ieee_quiet_nan);inf=ieee_value(0._dp,ieee_positive_inf)
 call get_command_argument(1,name)
 select case(trim(name))
 case('empty_outputs');call model%fit(x,y(:,:0))
 case('extra_empty_outputs');call extra%fit(x,y(:,:0))
 case('target_rows');call model%fit(x,y(:3,:))
 case('input_inf');x(1,1)=inf;call model%fit(x,y)
 case('weights_length');call model%fit(x,y,[1._dp])
 case('weights_nan');call model%fit(x,y,[bad,1._dp,1._dp,1._dp])
 case('weights_inf');call model%fit(x,y,[inf,1._dp,1._dp,1._dp])
 case('criterion');model%criterion='unknown';call model%fit(x,y)
 case('score_multiclass');call model%fit(x,y);score=model%score(x,y)
 case('score_zero_weights')
  call model%fit(x,indicator,[0._dp,0._dp,0._dp,0._dp])
  score=model%score(x,indicator,[0._dp,0._dp,0._dp,0._dp])
 case('score_outputs');call model%fit(x,y);score=model%score(x,y(:,:1))
 case('query_features');call model%fit(x,y);pred=model%predict_multi(x(:,:1))
 case('query_inf');call model%fit(x,y);x(1,1)=inf;pred=model%predict_multi(x)
 case('scalar_predict');call model%fit(x,y);scalar=model%predict(x)
 case('scalar_proba');call model%fit(x,y);prob=model%predict_proba(x)
 case default;call require(.false.,'unknown joint classification invalid case')
 end select
 call require(.false.,'invalid joint classification input was accepted')
end program
