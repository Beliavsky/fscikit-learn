program reference_quantile_integration_driver
  use sklearn_common,only:dp
  use sklearn_quantile,only:quantile_regressor
  use sklearn_search,only:grid_search_cv,randomized_search_cv,param,parameter_value
  use sklearn_pipeline,only:pipeline
  use sklearn_preprocessing,only:standard_scaler
  use sklearn_model_selection,only:cross_val_score
  use sklearn_sgd_clone,only:clone_quantile_parameters
  implicit none
  type(quantile_regressor)::base,copy
  class(grid_search_cv),allocatable::search
  type(parameter_value)::value
  type(pipeline)::pipe
  type(standard_scaler)::scale
  real(dp),allocatable::x(:,:),y(:),w(:),scores(:),before(:),candidates(:)
  integer::n,p,i,weighted,chain,scoring,random,unit,out
  character(1024)::path
  character(32)::prefix
  call get_command_argument(1,path);open(newunit=unit,file=trim(path),status='old')
  read(unit,*)n,p,weighted,chain,scoring,random;allocate(x(n,p),y(n),w(n))
  do i=1,n
    read(unit,*)x(i,:),y(i),w(i)
  end do
  close(unit);base%alpha=.03_dp;base%quantile=.5_dp;call base%fit(x,y);before=base%coef_
  if(random==1)then
    allocate(randomized_search_cv::search)
    select type(search)
    type is(randomized_search_cv);search%n_iter=4;search%random_state=42
    end select
  else
    allocate(grid_search_cv::search)
  end if
  copy=clone_quantile_parameters(base)
  if(weighted==1)then
    scores=cross_val_score(base,x,y,cv=4,sample_weight=w)
  else if(weighted==2)then
    scores=cross_val_score(base,x,y,cv=4,sample_weight=w(1))
  else if(weighted==3)then
    scores=cross_val_score(base,x,y,cv=4,sample_weight=w)
  else
    scores=cross_val_score(base,x,y,cv=4)
  end if
  prefix=''
  if(chain==1)then
    prefix='model__';allocate(pipe%steps(2));pipe%steps(1)%name='scale';pipe%steps(2)%name='model'
    allocate(pipe%steps(1)%model,source=scale);allocate(pipe%steps(2)%model,source=base)
    allocate(search%estimator,source=pipe)
  else
    allocate(search%estimator,source=base)
  end if
  allocate(search%param_grid(1));allocate(search%param_grid(1)%axes(2))
  search%param_grid(1)%axes(1)%name=trim(prefix)//'alpha'
  search%param_grid(1)%axes(1)%values=[param(0._dp),param(.05_dp)]
  search%param_grid(1)%axes(2)%name=trim(prefix)//'quantile'
  search%param_grid(1)%axes(2)%values=[param(.25_dp),param(.5_dp),param(.75_dp)]
  search%cv=4;search%return_train_score=.true.
  if(scoring==1)search%scoring='neg_mean_squared_error'
  if(weighted>0)then
    call search%fit(x,y,w)
  else
    call search%fit(x,y)
  end if
  call get_command_argument(2,path);open(newunit=out,file=trim(path),status='replace')
  call vector('cv',scores);call vector('mean',search%cv_results_%mean_test_score(:,1))
  call vector('std',search%cv_results_%std_test_score(:,1))
  call vector('train',search%cv_results_%mean_train_score(:,1))
  call vector('rank',real(search%cv_results_%rank_test_score(:,1),dp))
  call vector('split',reshape(search%cv_results_%split_test_score(:,:,1),[size(search%cv_results_%split_test_score(:,:,1))]))
  allocate(candidates(2*size(search%cv_results_%params)))
  do i=1,size(search%cv_results_%params)
    value=search%cv_results_%params(i)%get(trim(prefix)//'alpha');candidates(2*i-1)=value%as_real()
    value=search%cv_results_%params(i)%get(trim(prefix)//'quantile');candidates(2*i)=value%as_real()
  end do
  call vector('params',candidates)
  call vector('predict',search%predict(x+.137_dp));call vector('best',[real(search%best_index_,dp),search%best_score_])
  call vector('clone',real([merge(1,0,allocated(copy%coef_)),merge(1,0,allocated(copy%dual_coef_)), &
    copy%n_features_in_,copy%n_iter_,merge(1,0,copy%converged_)],dp))
  call vector('caller',[maxval(abs(base%coef_-before))]);close(out)
contains
  subroutine vector(name,a)
    character(*),intent(in)::name
    real(dp),intent(in)::a(:)
    write(out,'(a,1x,*(es26.17e3,1x))')name,a
  end subroutine
end program
