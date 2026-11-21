program threshold_invalid_driver
  use sklearn_threshold
  use sklearn_common, only: dp
  use sklearn_linear_model, only: logistic_regression
  use sklearn_naive_bayes, only: gaussian_nb
  use sklearn_ridge_classification, only: ridge_classifier
  use sklearn_neighbors, only: kneighbors_classifier
  use sklearn_dummy, only: dummy_classifier
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(fixed_threshold_classifier) :: fixed
  type(tuned_threshold_classifier_cv) :: tuned
  type(logistic_regression) :: lr
  type(gaussian_nb) :: nb
  type(ridge_classifier) :: ridge
  type(kneighbors_classifier) :: knn
  type(dummy_classifier) :: dummy
  real(dp) :: x(12,2),w(12),wide(12,3),nan,infinity,value
  real(dp), allocatable :: matrix(:,:)
  integer :: y(12),i
  integer, allocatable :: pred(:)
  character(32) :: command
  call get_command_argument(1,command)
  do i=1,12
    x(i,:)=[real(i,dp),real(mod(i,3),dp)]; y(i)=mod(i,2)
  end do
  w=1; wide=0; nan=ieee_value(0._dp,ieee_quiet_nan); infinity=ieee_value(0._dp,ieee_positive_inf)
  select case(trim(command))
  case('th_unfitted')
    pred=fixed%predict(x)
  case('th_tuned_unfitted')
    pred=tuned%predict(x)
  case('th_multiclass')
    y(1)=2; call fixed%fit(x,y,lr)
  case('th_oneclass')
    y=0; call fixed%fit(x,y,lr)
  case('th_method')
    fixed%response_method='bad'; call fixed%fit(x,y,lr)
  case('th_no_proba')
    fixed%response_method='predict_proba'; call fixed%fit(x,y,ridge)
  case('th_no_decision')
    fixed%response_method='decision_function'; call fixed%fit(x,y,nb)
  case('th_bad_base')
    call fixed%fit(x,y,3._dp)
  case('th_prefit_unfitted')
    fixed%prefit=.true.; call fixed%fit(x,y,lr)
  case('th_prefit_single')
    call nb%fit(x,spread(0,1,12)); fixed%prefit=.true.; call fixed%fit(x,y,nb)
  case('th_cutoff_nan')
    fixed%threshold=nan; call fixed%fit(x,y,lr)
  case('th_positive')
    fixed%pos_label=8; call fixed%fit(x,y,lr)
  case('th_query_width')
    call fixed%fit(x,y,lr); pred=fixed%predict(wide)
  case('th_query_nan')
    call fixed%fit(x,y,lr); x(1,1)=nan; pred=fixed%predict(x)
  case('th_weight_length')
    call fixed%fit(x,y,lr,w(:11))
  case('th_weight_nan')
    w(1)=nan; call fixed%fit(x,y,lr,w)
  case('th_weight_negative')
    w(1)=-1; call fixed%fit(x,y,lr,w)
  case('th_weight_zero')
    w=0; call fixed%fit(x,y,lr,w)
  case('th_weighted_knn')
    call fixed%fit(x,y,knn,w)
  case('th_score_length')
    call fixed%fit(x,y,lr); value=fixed%score(x,y(:11))
  case('th_score_weight_nan')
    call fixed%fit(x,y,lr); w(1)=nan; value=fixed%score(x,y,w)
  case('th_score_weight_zero')
    call fixed%fit(x,y,lr); w=0; value=fixed%score(x,y,w)
  case('th_cv_one')
    tuned%cv=1; call tuned%fit(x,y,lr)
  case('th_cv_many')
    tuned%cv=13; call tuned%fit(x,y,lr)
  case('th_cv_class_count')
    tuned%cv=10; call tuned%fit(x,y,lr)
  case('th_grid_empty')
    allocate(tuned%thresholds(0)); call tuned%fit(x,y,lr)
  case('th_grid_nan')
    tuned%thresholds=[0._dp,nan]; call tuned%fit(x,y,lr)
  case('th_grid_infinity')
    tuned%thresholds=[0._dp,infinity]; call tuned%fit(x,y,lr)
  case('th_grid_order')
    tuned%thresholds=[.8_dp,.2_dp]; call tuned%fit(x,y,lr)
  case('th_grid_close')
    tuned%thresholds=[1._dp,1._dp+1.e-7_dp]; call tuned%fit(x,y,lr)
  case('th_grid_one')
    tuned%n_thresholds=1; call tuned%fit(x,y,lr)
  case('th_grid_count')
    tuned%n_thresholds=0; call tuned%fit(x,y,lr)
  case('th_grid_range')
    tuned%thresholds=[huge(1._dp)/3,huge(1._dp)/2]; call tuned%fit(x,y,lr)
  case('th_scoring')
    tuned%scoring='bad'; call tuned%fit(x,y,lr)
  case('th_prefit_refit')
    tuned%prefit=.true.; call tuned%fit(x,y,lr)
  case('th_no_refit_multiple')
    tuned%refit=.false.; call tuned%fit(x,y,lr)
  case('th_fraction_zero')
    tuned%validation_fraction=0; call tuned%fit(x,y,lr)
  case('th_fraction_nan')
    tuned%validation_fraction=nan; call tuned%fit(x,y,lr)
  case('th_fraction_large')
    tuned%validation_fraction=1.1_dp; call tuned%fit(x,y,lr)
  case('th_fraction_negative')
    tuned%validation_fraction=-.2_dp; call tuned%fit(x,y,lr)
  case('th_splits_empty')
    allocate(tuned%splits(0)); call tuned%fit(x,y,lr)
  case('th_split_unallocated')
    allocate(tuned%splits(1)); call tuned%fit(x,y,lr)
  case('th_split_train_empty')
    allocate(tuned%splits(1)); tuned%splits(1)%train=[integer::]
    tuned%splits(1)%test=[1]; call tuned%fit(x,y,lr)
  case('th_split_test_empty')
    allocate(tuned%splits(1)); tuned%splits(1)%train=[1]
    tuned%splits(1)%test=[integer::]; call tuned%fit(x,y,lr)
  case('th_split_bounds')
    allocate(tuned%splits(1)); tuned%splits(1)%train=[0,1]
    tuned%splits(1)%test=[2]; call tuned%fit(x,y,lr)
  case('th_test_weights_zero')
    allocate(tuned%splits(1)); tuned%splits(1)%train=[1,2,3,4]
    tuned%splits(1)%test=[5,6]; w(5:6)=0
    call tuned%fit(x,y,lr,score_weight=w)
  case('th_train_weights_zero')
    allocate(tuned%splits(1)); tuned%splits(1)%train=[1,2,3,4]
    tuned%splits(1)%test=[5,6]; w(1:4)=0
    call tuned%fit(x,y,lr,w)
  case('th_prefit_labels')
    call lr%fit(x,y); y=y+2; tuned%prefit=.true.; tuned%refit=.false.
    call tuned%fit(x,y,lr)
  case('th_f1_default_label')
    y=merge(-3,7,y==0); tuned%scoring='f1'; call tuned%fit(x,y,lr)
  case('th_constant')
    call tuned%fit(x,y,dummy)
  case('th_objective_nan')
    call tuned%fit(x,y,lr,objective=bad_objective)
  case('th_validation_weight_length')
    call tuned%fit(x,y,lr,score_weight=w(:11))
  case('th_validation_weight_nan')
    w(1)=nan; call tuned%fit(x,y,lr,score_weight=w)
  case('th_validation_weight_negative')
    w(1)=-1; call tuned%fit(x,y,lr,score_weight=w)
  case('th_validation_weight_zero')
    w=0; call tuned%fit(x,y,lr,score_weight=w)
  case('th_forward_proba')
    call fixed%fit(x,y,ridge); matrix=fixed%predict_proba(x)
  case('th_forward_decision')
    call fixed%fit(x,y,nb); matrix=fixed%decision_function(x)
  case('th_tuned_width')
    call tuned%fit(x,y,lr); pred=tuned%predict(wide)
  case default
    error stop 'unknown threshold invalid-input case'
  end select
  print *, 'invalid input was accepted'
contains
  real(dp) function bad_objective(truth,prediction,sample_weight) result(result)
    integer, intent(in) :: truth(:),prediction(:)
    real(dp), intent(in), optional :: sample_weight(:)
    result=ieee_value(0._dp,ieee_quiet_nan)
  end function
end program
