program reference_gbr_integration_driver
 use sklearn_common,only:dp,require
 use sklearn_ensemble,only:gradient_boosting_regressor,bagging_regressor,voting_regressor,voting_estimator,stacking_regressor
 use sklearn_pipeline,only:pipeline
 use sklearn_preprocessing,only:standard_scaler
 use sklearn_search,only:grid_search_cv,param
 use sklearn_model_selection_features,only:select_from_model
 implicit none
 type(gradient_boosting_regressor)::base,other
 type(grid_search_cv)::search
 type(select_from_model)::selector,prefit
 type(pipeline)::pipe
 type(standard_scaler)::scale
 type(voting_regressor)::vote
 type(voting_estimator)::voters(2)
 type(bagging_regressor)::bag
 type(stacking_regressor)::stack
 real(dp),allocatable::x(:,:),query(:,:),y(:),w(:)
 integer::n,p,seed,weighted,status,i,j
 character(16)::loss
 character(32)::argument
 logical::stochastic
 call get_command_argument(1,argument);stochastic=trim(argument)=='--subsample'
 do
  read(*,*,iostat=status)n,p,seed,loss,weighted
  if(status/=0)exit
  allocate(x(n,p),y(n),w(n))
  do i=1,n;read(*,*)x(i,:);end do
  read(*,*)y;read(*,*)w;query=x+.137_dp
  base=gradient_boosting_regressor();base%n_estimators=3;base%max_depth=2;base%random_state=seed
  if(stochastic)base%subsample=merge(.65_dp,.8_dp,seed==42)
  base%loss=loss;base%alpha=.5_dp;call base%set_max_features(.7_dp)
  if(weighted==1)then
   call base%fit(x,0*y,w)
  else
   call base%fit(x,0*y)
  end if
  search=grid_search_cv();search%cv=3;allocate(search%estimator,source=base)
  allocate(search%param_grid(1));allocate(search%param_grid(1)%axes(merge(5,4,stochastic)))
  search%param_grid(1)%axes(1)%name='loss'
  search%param_grid(1)%axes(1)%values=[param('squared_error'),param('absolute_error'),param('quantile'),param('huber')]
  search%param_grid(1)%axes(2)%name='criterion'
  search%param_grid(1)%axes(2)%values=[param('friedman_mse'),param('squared_error')]
  search%param_grid(1)%axes(3)%name='max_features';search%param_grid(1)%axes(3)%values=[param(.5_dp),param('none')]
  search%param_grid(1)%axes(4)%name='alpha';search%param_grid(1)%axes(4)%values=[param(.3_dp),param(.8_dp)]
  if(stochastic)then
   search%param_grid(1)%axes(5)%name='subsample';search%param_grid(1)%axes(5)%values=[param(.5_dp),param(.8_dp)]
  end if
  selector=select_from_model();prefit=select_from_model();prefit%prefit=.true.
  pipe=pipeline();allocate(pipe%steps(2));pipe%steps(1)%name='scale';allocate(pipe%steps(1)%model,source=scale)
  pipe%steps(2)%name='boost';allocate(pipe%steps(2)%model,source=base)
  other=base;other%loss='squared_error';other%n_estimators=4;other%random_state=seed+1
  voters(1)%name='first';voters(2)%name='second'
  if(allocated(voters(1)%estimator))deallocate(voters(1)%estimator,voters(2)%estimator)
  allocate(voters(1)%estimator,source=base);allocate(voters(2)%estimator,source=other)
  vote=voting_regressor();vote%weights=[1._dp,2._dp]
  bag=bagging_regressor();bag%n_estimators=4;bag%random_state=seed;bag%oob_score=.true.
  bag%max_samples_fraction=.75_dp;bag%max_features_fraction=.8_dp
  stack=stacking_regressor();stack%cv=3
  if(weighted==1)then
   call search%fit(x,y,w);call selector%fit(x,y,base,w);call pipe%fit(x,y,w)
   call vote%fit(x,y,voters,w);call bag%fit(x,y,estimator=base,sample_weight=w)
   call stack%fit(x,y,voters,sample_weight=w)
  else
   call search%fit(x,y);call selector%fit(x,y,base);call pipe%fit(x,y)
   call vote%fit(x,y,voters);call bag%fit(x,y,estimator=base)
   call stack%fit(x,y,voters)
  end if
  call prefit%fit(x,estimator=base)
  write(*,'(*(es26.17e3,1x))')search%cv_results_%mean_test_score
  write(*,'(*(es26.17e3,1x))')search%cv_results_%std_test_score
  write(*,'(*(es26.17e3,1x))')transpose(search%cv_results_%split_test_score(:,:,1))
  write(*,'(*(i0,1x))')search%cv_results_%rank_test_score
  write(*,'(*(i0,1x))')search%best_index_
  write(*,'(*(es26.17e3,1x))')search%best_score_
  write(*,'(*(es26.17e3,1x))')search%predict(query)
  write(*,'(*(es26.17e3,1x))')selector%feature_importances()
  write(*,'(*(i0,1x))')merge(1,0,selector%get_support())
  write(*,'(*(es26.17e3,1x))')prefit%feature_importances()
  write(*,'(*(i0,1x))')merge(1,0,prefit%get_support())
  write(*,'(*(es26.17e3,1x))')base%predict(query)
  write(*,'(*(es26.17e3,1x))')pipe%predict(query)
  write(*,'(*(es26.17e3,1x))')vote%predict(query)
  write(*,'(*(es26.17e3,1x))')transpose(vote%transform(query))
  write(*,'(*(es26.17e3,1x))')bag%predict(query)
  write(*,'(*(i0,1x))')((bag%estimators_samples_(j)%values(i),j=1,4),i=1,bag%max_samples_)
  write(*,'(*(i0,1x))')((bag%estimators_features_(j)%values(i),j=1,4),i=1,bag%max_features_)
  write(*,'(*(i0,1x))')bag%seeds_
  do i=1,4
   select type(model=>bag%estimators_(i)%estimator)
   type is(gradient_boosting_regressor)
    write(*,'(i0,1x)',advance='no')model%random_state
   class default
    call require(.false.,'bagging component must be gradient boosting')
   end select
  end do
  write(*,*)
  write(*,'(*(es26.17e3,1x))')bag%oob_prediction_
  write(*,'(*(es26.17e3,1x))')bag%oob_score_
  if(weighted==1)then
   write(*,'(*(es26.17e3,1x))')pipe%score(x,y,w)
  else
   write(*,'(*(es26.17e3,1x))')pipe%score(x,y)
  end if
  write(*,'(*(es26.17e3,1x))')stack%predict(query)
  write(*,'(*(es26.17e3,1x))')transpose(stack%transform(query))
  deallocate(x,y,w,query)
 end do
end program
