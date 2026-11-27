program reference_gbc_integration_driver
 use sklearn_common,only:dp,require
 use sklearn_ensemble,only:gradient_boosting_classifier,bagging_classifier,voting_classifier,voting_estimator,stacking_classifier
 use sklearn_pipeline,only:pipeline
 use sklearn_preprocessing,only:standard_scaler
 use sklearn_search,only:grid_search_cv,param
 use sklearn_model_selection_features,only:select_from_model
 use sklearn_classic_gradient_boosting,only:reset_boosting_fit
 implicit none
 type(gradient_boosting_classifier)::base,other,final
 type(grid_search_cv)::search
 type(select_from_model)::selector,prefit
 type(pipeline)::pipe
 type(standard_scaler)::scale
 type(voting_classifier)::vote
 type(voting_estimator)::voters(2)
 type(bagging_classifier)::bag
 type(stacking_classifier)::stack
 real(dp),allocatable::x(:,:),query(:,:),w(:)
 integer,allocatable::y(:)
 integer::n,p,seed,weighted,status,i,j
 character(16)::loss
 character(32)::argument
 logical::stochastic,warm_template,early_stopping
 stochastic=.false.;warm_template=.false.;early_stopping=.false.
 do i=1,command_argument_count()
  call get_command_argument(i,argument)
  if(trim(argument)=='--subsample')stochastic=.true.
  if(trim(argument)=='--warm-start')warm_template=.true.
  if(trim(argument)=='--early-stopping')early_stopping=.true.
 end do
 do
  read(*,*,iostat=status)n,p,seed,loss,weighted
  if(status/=0)exit
  allocate(x(n,p),y(n),w(n))
  do i=1,n;read(*,*)x(i,:);end do
  read(*,*)y;read(*,*)w;query=x+.137_dp
  base=gradient_boosting_classifier();base%n_estimators=3;base%max_depth=2;base%random_state=seed
  if(early_stopping)then
   base%n_iter_no_change=1;base%validation_fraction=.3_dp;base%tol=1.e6_dp
  end if
  base%warm_start=warm_template
  if(stochastic)base%subsample=merge(.65_dp,.8_dp,seed==42)
  base%loss=loss;call base%set_max_features(.7_dp)
  if(weighted==1)then
   call base%fit(x,y,w)
  else
   call base%fit(x,y)
  end if
  search=grid_search_cv();search%cv=3;allocate(search%estimator,source=base)
  allocate(search%param_grid(1));allocate(search%param_grid(1)%axes(4+merge(1,0,stochastic)+merge(1,0,warm_template)))
  search%param_grid(1)%axes(1)%name='ccp_alpha'
  search%param_grid(1)%axes(1)%values=[param(0._dp),param(.01_dp)]
  search%param_grid(1)%axes(2)%name='criterion'
  search%param_grid(1)%axes(2)%values=[param('friedman_mse'),param('squared_error')]
  search%param_grid(1)%axes(3)%name='max_features';search%param_grid(1)%axes(3)%values=[param(.5_dp),param('none')]
  search%param_grid(1)%axes(4)%name='max_leaf_nodes';search%param_grid(1)%axes(4)%values=[param(0),param(3)]
  if(stochastic)then
   search%param_grid(1)%axes(5)%name='subsample';search%param_grid(1)%axes(5)%values=[param(.5_dp),param(.8_dp)]
  end if
  if(warm_template)then
   i=5+merge(1,0,stochastic)
   search%param_grid(1)%axes(i)%name='warm_start';search%param_grid(1)%axes(i)%values=[param(.true.)]
  end if
  selector=select_from_model();prefit=select_from_model();prefit%prefit=.true.
  pipe=pipeline();allocate(pipe%steps(2));pipe%steps(1)%name='scale';allocate(pipe%steps(1)%model,source=scale)
  pipe%steps(2)%name='boost';allocate(pipe%steps(2)%model,source=base)
  call reset_boosting_fit(pipe%steps(2)%model)
  other=base;other%loss='log_loss';other%n_estimators=4;other%random_state=seed+1
  voters(1)%name='first';voters(2)%name='second'
  if(allocated(voters(1)%estimator))deallocate(voters(1)%estimator,voters(2)%estimator)
  allocate(voters(1)%estimator,source=base);allocate(voters(2)%estimator,source=other)
  vote=voting_classifier();vote%weights=[1._dp,2._dp];vote%voting='soft'
  bag=bagging_classifier();bag%n_estimators=4;bag%random_state=seed;bag%oob_score=.true.
  bag%max_samples_fraction=.75_dp;bag%max_features_fraction=.8_dp
  stack=stacking_classifier();stack%cv=3
  if(seed==42)stack%stack_method='decision_function'
  final=gradient_boosting_classifier();final%n_estimators=3;final%max_depth=1;final%random_state=19
  if(weighted==1)then
   call search%fit(x,y,w);call selector%fit(x,y,base,w);call pipe%fit(x,y,w)
   call vote%fit(x,y,voters,w);call bag%fit(x,y,estimator=base,sample_weight=w)
   call stack%fit(x,y,voters,final_estimator=final,sample_weight=w)
  else
   call search%fit(x,y);call selector%fit(x,y,base);call pipe%fit(x,y)
   call vote%fit(x,y,voters);call bag%fit(x,y,estimator=base)
   call stack%fit(x,y,voters,final_estimator=final)
  end if
  call prefit%fit(x,estimator=base)
  write(*,'(*(es26.17e3,1x))')search%cv_results_%mean_test_score
  write(*,'(*(es26.17e3,1x))')search%cv_results_%std_test_score
  write(*,'(*(es26.17e3,1x))')transpose(search%cv_results_%split_test_score(:,:,1))
  write(*,'(*(i0,1x))')search%cv_results_%rank_test_score
  write(*,'(*(i0,1x))')search%best_index_
  write(*,'(*(es26.17e3,1x))')search%best_score_
  write(*,'(*(es26.17e3,1x))')real(search%predict_class(query),dp)
  write(*,'(*(es26.17e3,1x))')selector%feature_importances()
  write(*,'(*(i0,1x))')merge(1,0,selector%get_support())
  write(*,'(*(es26.17e3,1x))')prefit%feature_importances()
  write(*,'(*(i0,1x))')merge(1,0,prefit%get_support())
  write(*,'(*(es26.17e3,1x))')real(base%predict(query),dp)
  write(*,'(*(es26.17e3,1x))')real(pipe%predict_class(query),dp)
  write(*,'(*(es26.17e3,1x))')real(vote%predict(query),dp)
  write(*,'(*(es26.17e3,1x))')transpose(vote%transform(query))
  write(*,'(*(es26.17e3,1x))')real(bag%predict(query),dp)
  write(*,'(*(i0,1x))')((bag%estimators_samples_(j)%values(i),j=1,4),i=1,bag%max_samples_)
  write(*,'(*(i0,1x))')((bag%estimators_features_(j)%values(i),j=1,4),i=1,bag%max_features_)
  write(*,'(*(i0,1x))')bag%seeds_
  do i=1,4
   select type(model=>bag%estimators_(i)%estimator)
   type is(gradient_boosting_classifier)
    write(*,'(i0,1x)',advance='no')model%random_state
   class default
    call require(.false.,'bagging component must be gradient boosting')
   end select
  end do
  write(*,*)
  write(*,'(*(es26.17e3,1x))')transpose(bag%oob_decision_function_)
  write(*,'(*(es26.17e3,1x))')bag%oob_score_
  if(weighted==1)then
   write(*,'(*(es26.17e3,1x))')pipe%score(x,y,w)
  else
   write(*,'(*(es26.17e3,1x))')pipe%score(x,y)
  end if
  write(*,'(*(es26.17e3,1x))')real(stack%predict(query),dp)
  write(*,'(*(es26.17e3,1x))')transpose(stack%transform(query))
  write(*,'(*(es26.17e3,1x))')transpose(bag%predict_proba(query))
  write(*,'(*(es26.17e3,1x))')transpose(vote%predict_proba(query))
  write(*,'(*(es26.17e3,1x))')transpose(stack%predict_proba(query))
  deallocate(x,y,w,query)
 end do
end program
