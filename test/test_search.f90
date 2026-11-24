program test_search
  use sklearn_common, only: dp,require
  use sklearn_search
  use sklearn_linear_model, only: ridge,logistic_regression
  use sklearn_pipeline, only: pipeline
  use sklearn_preprocessing, only: standard_scaler
  use sklearn_ensemble, only: random_forest_regressor,extra_trees_regressor
  use sklearn_tree, only: decision_tree_regressor,extra_tree_regressor
  implicit none
  type(grid_search_cv) :: search,copied,multi
  type(grid_search_cv) :: criterion_search
  type(decision_tree_regressor) :: count_tree
  type(extra_tree_regressor) :: count_extra
  type(random_forest_regressor) :: count_forest
  type(extra_trees_regressor) :: count_forest_extra
  type(randomized_search_cv) :: random
  type(ridge) :: base
  type(logistic_regression) :: classifier
  type(pipeline) :: chain
  type(standard_scaler) :: scale
  type(parameter_value) :: value
  real(dp) :: x(40,3),y(40),weights(40)
  real(dp), allocatable :: predicted(:),prob(:,:),decision(:,:)
  integer :: i,labels(40)
  integer, allocatable :: classified(:)
  do i=1,40
    x(i,:)=[real(i,dp)/10,sin(real(i,dp)),cos(real(2*i,dp))]
  end do
  y=2*x(:,1)-x(:,2)+.2_dp*x(:,3);weights=1;weights(:2)=0
  allocate(search%estimator,source=base);allocate(search%param_grid(1));allocate(search%param_grid(1)%axes(2))
  search%param_grid(1)%axes(1)%name='fit_intercept'
  search%param_grid(1)%axes(1)%values=[param(.true.),param(.false.)]
  search%param_grid(1)%axes(2)%name='alpha'
  search%param_grid(1)%axes(2)%values=[param(0._dp),param(.1_dp),param(10._dp)]
  search%cv=4;search%return_train_score=.true.;search%scoring='neg_mean_squared_error'
  call search%fit(x,y,weights)
  call require(search%n_splits_==4.and.size(search%cv_results_%params)==6,'search candidate/fold counts')
  predicted=search%predict(x)
  call require(maxval(abs(predicted-y))<1.e-10_dp,'refitted least squares prediction')
  value=search%best_params_%get('alpha');call require(value%as_real()==0,'search selects unregularized exact model')
  call require(.not.allocated(base%coef_),'search does not fit caller estimator')
  call require(all(search%cv_results_%rank_test_score>=1),'positive minimum ranks')
  call require(all(search%cv_results_%fit_time>=0).and.all(search%cv_results_%score_time>=0),'nonnegative timing')
  copied=search;search%refit=.false.;call search%fit(x,y)
  call require(.not.allocated(search%best_estimator_).and.search%best_index_>0,'refit=false retains single score best')
  predicted=copied%predict(x);call require(maxval(abs(predicted-y))<1.e-10_dp,'copied fitted search is independent')
  allocate(random%estimator,source=base);random%param_grid=copied%param_grid;random%n_iter=3;random%random_state=42
  random%scoring='neg_mean_squared_error';call random%fit(x,y)
  call require(size(random%cv_results_%params)==3.and.random%best_index_>0,'randomized finite search')
  labels=merge(9,-4,x(:,2)>.1_dp);classifier%tol=1.e-10_dp
  allocate(multi%estimator,source=classifier);allocate(multi%param_grid(1));allocate(multi%param_grid(1)%axes(1))
  multi%param_grid(1)%axes(1)%name='C';multi%param_grid(1)%axes(1)%values=[param(.1_dp),param(1._dp)]
  allocate(multi%scorers(2));multi%scorers(1)%name='accuracy';multi%scorers(1)%scoring='accuracy'
  multi%scorers(2)%name='loss';multi%scorers(2)%scoring='neg_log_loss';multi%refit_metric='loss';multi%cv=3
  call multi%fit(x,real(labels,dp));classified=multi%predict_class(x);prob=multi%predict_proba(x)
  decision=multi%decision_function(x)
  call require(all(shape(multi%cv_results_%mean_test_score)==[2,2]),'multimetric score storage')
  call require(all(abs(sum(prob,dim=2)-1)<1.e-12_dp).and.size(decision,2)==1,'refit classifier delegation')
  call require(multi%score(x,labels)<0,'score uses selected refit scorer')
  multi%refit_func=>last_candidate;call multi%fit(x,labels)
  call require(multi%best_index_==2.and..not.multi%best_score_available_,'callable refit omits best_score')
  multi%refit_func=>null();multi%refit=.false.;multi%refit_metric='';call multi%fit(x,labels)
  call require(multi%best_index_==0,'multimetric without refit has no best candidate')
  allocate(chain%steps(2));chain%steps(1)%name='scale';chain%steps(2)%name='model'
  allocate(chain%steps(1)%model,source=scale);allocate(chain%steps(2)%model,source=base)
  search=grid_search_cv();allocate(search%estimator,source=chain)
  allocate(search%param_grid(1));allocate(search%param_grid(1)%axes(1))
  search%param_grid(1)%axes(1)%name='model__alpha';search%param_grid(1)%axes(1)%values=[param(.01_dp),param(1._dp)]
  call search%fit(x,labels);predicted=search%predict(x)
  call require(size(predicted)==40.and.search%best_index_>0,'nested Pipeline parameters with integer regression target')
  count_tree%max_depth=2
  allocate(criterion_search%estimator,source=count_tree);allocate(criterion_search%param_grid(1))
  allocate(criterion_search%param_grid(1)%axes(1));criterion_search%param_grid(1)%axes(1)%name='criterion'
  criterion_search%param_grid(1)%axes(1)%values=[param('squared_error'),param('friedman_mse'),param('poisson'),param('absolute_error')]
  criterion_search%cv=3;criterion_search%scoring='neg_mean_squared_error'
  call criterion_search%fit(x,1+abs(y),weights)
  value=criterion_search%best_params_%get('criterion')
  select type(fitted=>criterion_search%best_estimator_)
  class is(decision_tree_regressor)
    call require(fitted%criterion==value%as_text(),'Refitted regression tree retains searched criterion')
  class default
    call require(.false.,'Criterion search returns a tree regressor')
  end select
  count_extra%max_depth=2;deallocate(criterion_search%estimator)
  allocate(criterion_search%estimator,source=count_extra)
  call criterion_search%fit(x,1+abs(y),weights)
  call require(size(criterion_search%cv_results_%params)==4,'ExtraTree criterion parameter adapter')
  count_forest%n_estimators=3;count_forest%max_depth=2
  deallocate(criterion_search%estimator);allocate(criterion_search%estimator,source=count_forest)
  call criterion_search%fit(x,1+abs(y),weights)
  value=criterion_search%best_params_%get('criterion')
  select type(fitted=>criterion_search%best_estimator_)
  class is(random_forest_regressor)
    call require(fitted%criterion==value%as_text(),'Weighted forest criterion search refits best parameters')
  class default
    call require(.false.,'Criterion search returns a forest regressor')
  end select
  count_forest_extra%n_estimators=3;count_forest_extra%max_depth=2
  deallocate(criterion_search%estimator);allocate(criterion_search%estimator,source=count_forest_extra)
  call criterion_search%fit(x,1+abs(y),weights)
  call require(size(criterion_search%cv_results_%params)==4,'Weighted ExtraTrees criterion parameter adapter')
  print *, 'GridSearchCV and RandomizedSearchCV: PASS'
contains
  integer function last_candidate(results) result(index)
    type(search_cv_results), intent(in) :: results
    index=size(results%params)
  end function
end program
