program validation_invalid_driver
  use sklearn_validation
  use sklearn_common, only: dp
  use sklearn_linear_model, only: ridge,logistic_regression
  use sklearn_naive_bayes, only: gaussian_nb
  use sklearn_neighbors, only: kneighbors_classifier
  use sklearn_preprocessing, only: standard_scaler
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  implicit none
  class(*), allocatable :: estimator
  type(ridge) :: base
  type(logistic_regression) :: classifier
  type(gaussian_nb) :: bayes
  type(kneighbors_classifier) :: neighbors
  type(standard_scaler) :: scale
  type(cross_validation_options) :: options
  type(cross_validation_result) :: results
  real(dp) :: x(12,3),y(12),weights(12)
  real(dp), allocatable :: prediction(:,:),scores(:)
  integer :: yi(12),i
  integer, allocatable :: labels(:)
  logical :: predict=.false.
  character(64) :: test_case
  call get_command_argument(1,test_case)
  x=reshape([(real(i,dp)/10,i=1,36)],[12,3]);y=x(:,1);weights=1
  yi=merge(9,-4,mod([(i,i=1,12)],2)==0);allocate(estimator,source=base);options%cv=3
  select case(trim(test_case))
  case('empty_x');results=cross_validate(estimator,x(:0,:),y(:0),options)
  case('nonfinite_x');x(1,1)=ieee_value(0._dp,ieee_quiet_nan)
  case('target_count');results=cross_validate(estimator,x,y(:11),options)
  case('nonfinite_y');y(1)=ieee_value(0._dp,ieee_quiet_nan)
  case('unknown_task');options%task='unknown'
  case('error_score');options%error_score='nan'
  case('cv_one');options%cv=1
  case('cv_large');options%cv=13
  case('no_splits');allocate(options%cv_splits(0))
  case('unallocated_indices');allocate(options%cv_splits(1))
  case('empty_train','empty_test','low_train','high_train','low_test','high_test')
    allocate(options%cv_splits(1));options%cv_splits(1)%train=[1,2,3,4];options%cv_splits(1)%test=[5,6]
    select case(test_case)
    case('empty_train');options%cv_splits(1)%train=[integer::]
    case('empty_test');options%cv_splits(1)%test=[integer::]
    case('low_train');options%cv_splits(1)%train=[0,1]
    case('high_train');options%cv_splits(1)%train=[1,13]
    case('low_test');options%cv_splits(1)%test=[0]
    case('high_test');options%cv_splits(1)%test=[13]
    end select
  case('weights_count');results=cross_validate(estimator,x,y,options,weights(:11))
  case('weights_nan');weights(1)=ieee_value(0._dp,ieee_quiet_nan);results=cross_validate(estimator,x,y,options,weights)
  case('weights_negative');weights(1)=-1;results=cross_validate(estimator,x,y,options,weights)
  case('scorers_empty');allocate(options%scorers(0))
  case('scorer_name');allocate(options%scorers(1));options%scorers(1)%name=''
  case('scorer_duplicate');allocate(options%scorers(2))
  case('scorer_conflict');allocate(options%scorers(1));options%scorer_func=>custom_score
  case('unknown_scorer');options%scoring='unknown'
  case('scalar_multiple');allocate(options%scorers(1));scores=cross_val_score(estimator,x,y,options)
  case('unsupervised_scorer')
    deallocate(estimator);allocate(estimator,source=scale);results=cross_validate(estimator,x,options)
  case('unsupervised_weights')
    deallocate(estimator);allocate(estimator,source=scale);options%scorer_func=>custom_score
    results=cross_validate(estimator,x,options,weights)
  case('classifier_no_target')
    deallocate(estimator);allocate(estimator,source=classifier);results=cross_validate(estimator,x,options)
  case('class_continuous','class_large','class_cv')
    deallocate(estimator);allocate(estimator,source=classifier);y=real(yi,dp)
    if(test_case=='class_continuous') y(1)=.5_dp
    if(test_case=='class_large') y(1)=real(huge(1),dp)+1
    if(test_case=='class_cv') options%cv=7
  case('unknown_method');options%method='transform';predict=.true.
  case('regressor_response');options%method='predict_proba';predict=.true.
  case('partition_missing','partition_repeat','partition_duplicate')
    allocate(options%cv_splits(1));options%cv_splits(1)%train=[1,2,3,4]
    options%cv_splits(1)%test=[(i,i=1,12)]
    if(test_case=='partition_missing') options%cv_splits(1)%test=[1,2,3]
    if(test_case=='partition_duplicate') options%cv_splits(1)%test=[(i,i=1,12),1]
    if(test_case=='partition_repeat') options%cv_splits=[options%cv_splits,options%cv_splits]
    predict=.true.
  case('response_samples','response_widths')
    options%cv=5;options%response_func=>bad_response;predict=.true.
  case('classes_empty','classes_range','classes_duplicate','probability_width','decision_binary', &
       'decision_two_column','log_neighbors','decision_bayes')
    deallocate(estimator);allocate(estimator,source=bayes);y=real(yi,dp)
    options%method='predict_proba';predict=.true.
    select case(test_case)
    case('classes_empty','classes_range','classes_duplicate');options%classes_func=>bad_classes
    case('probability_width')
      y=-4;y(1)=9;options%response_func=>bad_response
    case('decision_binary','decision_two_column')
      deallocate(estimator);allocate(estimator,source=classifier);y(1)=15;options%method='decision_function'
      if(test_case=='decision_two_column') options%response_func=>bad_response
    case('log_neighbors')
      deallocate(estimator);allocate(estimator,source=neighbors);options%method='predict_log_proba'
    case('decision_bayes');options%method='decision_function'
    end select
  case('typed_class_regressor');labels=cross_val_predict_class(estimator,x,yi,options)
  case('typed_class_method')
    deallocate(estimator);allocate(estimator,source=classifier);options%method='predict_proba'
    labels=cross_val_predict_class(estimator,x,yi,options)
  case('typed_class_columns')
    deallocate(estimator);allocate(estimator,source=classifier);options%response_func=>bad_response
    labels=cross_val_predict_class(estimator,x,yi,options)
  case('typed_class_nonintegral')
    deallocate(estimator);allocate(estimator,source=classifier);options%response_func=>bad_response
    labels=cross_val_predict_class(estimator,x,yi,options)
  case default
    error stop 'unknown validation invalid case'
  end select
  if(predict) then
    prediction=cross_val_predict(estimator,x,y,options)
  else
    results=cross_validate(estimator,x,y,options)
  end if
  error stop 'validation invalid case unexpectedly accepted'
contains
  real(dp) function custom_score(model,x,yr,yi) result(value)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: yr(:)
    integer, optional, intent(in) :: yi(:)
    value=0
  end function
  function bad_classes(model) result(classes)
    class(*), intent(in) :: model
    integer, allocatable :: classes(:)
    select case(test_case)
    case('classes_empty');allocate(classes(0))
    case('classes_range');classes=[-1,1]
    case('classes_duplicate');classes=[0,0]
    end select
  end function
  function bad_response(model,x,method) result(values)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    character(*), intent(in) :: method
    real(dp), allocatable :: values(:,:)
    select case(test_case)
    case('response_samples');allocate(values(size(x,1)-1,1));values=0
    case('response_widths');allocate(values(size(x,1),merge(1,2,size(x,1)==3)));values=0
    case('probability_width');allocate(values(size(x,1),3));values=1._dp/3
    case('decision_two_column','typed_class_columns');allocate(values(size(x,1),2));values=0
    case('typed_class_nonintegral');allocate(values(size(x,1),1));values=.5_dp
    end select
  end function
end program
