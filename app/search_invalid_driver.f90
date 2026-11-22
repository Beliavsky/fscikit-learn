program search_invalid_driver
  use sklearn_search
  use sklearn_common, only: dp
  use sklearn_linear_model, only: ridge,logistic_regression
  use sklearn_pipeline, only: pipeline
  use sklearn_legacy_rng, only: legacy_random_state
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  implicit none
  type(grid_search_cv) :: model
  type(ridge) :: base
  type(logistic_regression) :: classifier
  type(pipeline) :: chain
  type(parameter_grid) :: empty(0)
  type(parameter_set), allocatable :: sets(:)
  type(parameter_set) :: setting
  type(parameter_value) :: value
  real(dp) :: x(12,2),y(12),w(12),number
  real(dp), allocatable :: predicted(:),matrix(:,:)
  integer :: labels(12),i,integer_value
  logical :: flag
  character(64) :: test_case
  character(:), allocatable :: text
  call get_command_argument(1,test_case)
  x=reshape([(real(i,dp)/10,i=1,24)],[12,2]);y=x(:,1);labels=merge(9,-4,mod([(i,i=1,12)],2)==0);w=1
  allocate(model%estimator,source=base);allocate(model%param_grid(1));allocate(model%param_grid(1)%axes(1))
  model%param_grid(1)%axes(1)%name='alpha';model%param_grid(1)%axes(1)%values=[param(.1_dp),param(1._dp)]
  model%cv=3
  select case(trim(test_case))
  case('missing_estimator');deallocate(model%estimator)
  case('missing_grid');deallocate(model%param_grid)
  case('empty_grids');model%param_grid=empty
  case('axis_name');deallocate(model%param_grid(1)%axes(1)%name)
  case('axis_values');deallocate(model%param_grid(1)%axes(1)%values)
  case('axis_empty');model%param_grid(1)%axes(1)%values=[parameter_value::]
  case('axis_duplicate');model%param_grid(1)%axes=[model%param_grid(1)%axes,model%param_grid(1)%axes]
  case('value_kind');model%param_grid(1)%axes(1)%values(1)%kind=0
  case('value_text');model%param_grid(1)%axes(1)%values(1)%kind=4
  case('grid_distribution')
    deallocate(model%param_grid(1)%axes(1)%values);model%param_grid(1)%axes(1)%rvs=>distribution
  case('axis_both');model%param_grid(1)%axes(1)%rvs=>distribution
  case('grid_index_zero');setting=model%param_grid(1)%at(0)
  case('grid_index_large');setting=model%param_grid(1)%at(3)
  case('sample_zero');sets=sample_parameters(model%param_grid,0,0)
  case('sample_empty');sets=sample_parameters(empty,1,0)
  case('sample_seed');sets=sample_parameters(model%param_grid,1,-1)
  case('parameter_missing');value=setting%get('absent')
  case('wrong_numeric');value=param('x');number=value%as_real()
  case('wrong_integer');value=param(1._dp);integer_value=value%as_integer()
  case('wrong_logical');value=param(1);flag=value%as_logical()
  case('wrong_text');value=param(1);text=value%as_text()
  case('empty_x');call model%fit(x(:0,:),y(:0))
  case('target_count');call model%fit(x,y(:11))
  case('nonfinite_x');x(1,1)=ieee_value(0._dp,ieee_quiet_nan)
  case('nonfinite_y');y(1)=ieee_value(0._dp,ieee_quiet_nan)
  case('invalid_task');model%task='unknown'
  case('cv_one');model%cv=1
  case('cv_large');model%cv=13
  case('cv_empty');allocate(model%cv_splits(0))
  case('cv_unallocated');allocate(model%cv_splits(1))
  case('cv_empty_train','cv_empty_test','cv_train_zero','cv_test_large')
    allocate(model%cv_splits(1));model%cv_splits(1)%train=[1,2,3,4];model%cv_splits(1)%test=[5,6]
    select case(test_case)
    case('cv_empty_train');model%cv_splits(1)%train=[integer::]
    case('cv_empty_test');model%cv_splits(1)%test=[integer::]
    case('cv_train_zero');model%cv_splits(1)%train=[0,1]
    case('cv_test_large');model%cv_splits(1)%test=[13]
    end select
  case('unsupported_scorer');model%scoring='unknown'
  case('error_score');model%error_score='nan'
  case('unknown_parameter');model%param_grid(1)%axes(1)%name='not_a_parameter'
  case('parameter_type');model%param_grid(1)%axes(1)%values=[param('bad')]
  case('nested_parameter');model%param_grid(1)%axes(1)%name='model__alpha'
  case('bad_nested_name');model%param_grid(1)%axes(1)%name='__alpha'
  case('unfitted_predict');predicted=model%predict(x)
  case('no_refit_predict');model%refit=.false.;call model%fit(x,y);predicted=model%predict(x)
  case('wrong_predict_role');call model%fit(x,y);labels=model%predict_class(x)
  case('wrong_proba_role');call model%fit(x,y);matrix=model%predict_proba(x)
  case('wrong_decision_role');call model%fit(x,y);matrix=model%decision_function(x)
  case('multimetric_missing','scorers_empty','scorers_duplicate','scorer_name','scorer_conflict')
    allocate(model%scorers(2));model%scorers(1)%name='one';model%scorers(2)%name='two'
    model%refit_metric='one'
    select case(test_case)
    case('multimetric_missing');model%refit_metric='missing'
    case('scorers_empty');deallocate(model%scorers);allocate(model%scorers(0))
    case('scorers_duplicate');model%scorers(2)%name='one'
    case('scorer_name');model%scorers(1)%name=''
    case('scorer_conflict');model%scorer_func=>custom_score
    end select
  case('refit_index_zero','refit_index_large','refit_conflict')
    model%refit_func=>bad_refit
    if(test_case=='refit_conflict') model%refit=.false.
  case('weight_count');call model%fit(x,y,w(:11))
  case('weight_nan');w(1)=ieee_value(0._dp,ieee_quiet_nan);call model%fit(x,y,w)
  case('weight_negative');w(1)=-1;call model%fit(x,y,w)
  case('classifier_continuous','classifier_large','classifier_cv','log_loss_classes')
    deallocate(model%estimator);allocate(model%estimator,source=classifier)
    model%param_grid(1)%axes(1)%name='C';y=real(labels,dp)
    if(test_case=='classifier_continuous') y(1)=.5_dp
    if(test_case=='classifier_large') y(1)=real(huge(1),dp)+1
    if(test_case=='classifier_cv') model%cv=7
    if(test_case=='log_loss_classes') then
      model%scoring='neg_log_loss';allocate(model%cv_splits(1))
      model%cv_splits(1)%test=[1,3,5];model%cv_splits(1)%train=[2,4,6,7,8,9,10,11,12]
    end if
  case('unknown_pipeline_step')
    allocate(chain%steps(1));chain%steps(1)%name='model';allocate(chain%steps(1)%model,source=base)
    deallocate(model%estimator);allocate(model%estimator,source=chain)
    model%param_grid(1)%axes(1)%name='absent__alpha'
  case default
    error stop 'unknown search invalid case'
  end select
  call model%fit(x,y)
  error stop 'search invalid case unexpectedly accepted'
contains
  function distribution(rng) result(value)
    type(legacy_random_state), intent(inout) :: rng
    type(parameter_value) :: value
    value=param(rng%uniform())
  end function
  integer function bad_refit(results) result(index)
    type(search_cv_results), intent(in) :: results
    index=0
    if(test_case=='refit_index_large') index=size(results%params)+1
  end function
  real(dp) function custom_score(estimator,x,yr,yi) result(value)
    class(*), intent(in) :: estimator
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: yr(:)
    integer, optional, intent(in) :: yi(:)
    value=0
  end function
end program
