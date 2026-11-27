! sklearn 1.6.1 model_selection/_validation.py and _split.py (BSD-3-Clause).
! Typed result/options storage and native adapters are original MIT contributions.
module sklearn_validation
  use sklearn_classic_gradient_boosting,only:reset_boosting_fit
  use sklearn_common, only: dp,require,check_x,check_xy,unique_labels,label_index,argsort
  use sklearn_model_selection, only: kfold,index_split
  use sklearn_search, only: search_scorer
  use sklearn_parameter_space, only: parameter_value,parameter_set
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_ieee_math, only: ieee_mean
  use sklearn_search_adapters, only: reset_search_clone, search_classifier,fit_search_native,search_regression,search_labels, &
    search_probabilities,search_decisions,search_classes,native_search_score,set_search_params
  use sklearn_feature_composition, only: fit_model
  use sklearn_pipeline, only: pipeline
  use sklearn_linear_model, only: logistic_regression
  use sklearn_naive_bayes, only: gaussian_nb,multinomial_nb
  use sklearn_tree, only: decision_tree_classifier
  use sklearn_ensemble, only: random_forest_classifier,gradient_boosting_classifier
  use sklearn_sgd, only: sgd_classifier,sgd_regressor
  use sklearn_sgd_clone, only: reset_sgd_fit
  use sklearn_dummy, only: dummy_classifier
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value,ieee_negative_inf
  implicit none
  private
  public :: cross_validate,cross_val_predict,cross_val_predict_class,cross_val_score
  public :: cross_validation_options,cross_validation_result,cv_estimator,search_scorer
  public :: learning_curve,validation_curve,permutation_test_score
  public :: learning_curve_options,learning_curve_result,validation_curve_result,permutation_score_result
  abstract interface
    subroutine cv_fit_function(model,x,yr,yi,sample_weight)
      import dp
      class(*), intent(inout) :: model
      real(dp), intent(in) :: x(:,:)
      real(dp), optional, intent(in) :: yr(:),sample_weight(:)
      integer, optional, intent(in) :: yi(:)
    end subroutine
    real(dp) function cv_score_function(model,x,yr,yi) result(value)
      import dp
      class(*), intent(in) :: model
      real(dp), intent(in) :: x(:,:)
      real(dp), optional, intent(in) :: yr(:)
      integer, optional, intent(in) :: yi(:)
    end function
    function cv_response_function(model,x,method) result(values)
      import dp
      class(*), intent(in) :: model
      real(dp), intent(in) :: x(:,:)
      character(*), intent(in) :: method
      real(dp), allocatable :: values(:,:)
    end function
    function cv_classes_function(model) result(classes)
      class(*), intent(in) :: model
      integer, allocatable :: classes(:)
    end function
  end interface
  type :: cross_validation_options
    integer :: cv=5
    type(index_split), allocatable :: cv_splits(:)
    character(64) :: scoring='auto'
    character(24) :: method='predict',task='auto',error_score='raise'
    type(search_scorer), allocatable :: scorers(:)
    procedure(cv_fit_function), pointer, nopass :: fit_func=>null()
    procedure(cv_score_function), pointer, nopass :: scorer_func=>null()
    procedure(cv_response_function), pointer, nopass :: response_func=>null()
    procedure(cv_classes_function), pointer, nopass :: classes_func=>null()
    logical :: return_train_score=.false.,return_estimator=.false.,return_indices=.false.
  end type
  type :: cv_estimator
    class(*), allocatable :: model
  end type
  type :: cross_validation_result
    real(dp), allocatable :: test_score(:,:),train_score(:,:),fit_time(:),score_time(:)
    character(64), allocatable :: metric_names(:)
    type(cv_estimator), allocatable :: estimators(:)
    type(index_split), allocatable :: indices(:)
    integer :: n_splits=0,n_features_in=0
  end type
  abstract interface
    subroutine curve_partial_function(model,x,yr,yi,classes,sample_weight)
      import dp
      class(*), intent(inout) :: model
      real(dp), intent(in) :: x(:,:)
      real(dp), optional, intent(in) :: yr(:),sample_weight(:)
      integer, optional, intent(in) :: yi(:),classes(:)
    end subroutine
    subroutine curve_initialize_function(model)
      class(*), intent(inout) :: model
    end subroutine
    subroutine curve_parameter_function(model,parameters)
      import parameter_set
      class(*), intent(inout) :: model
      type(parameter_set), intent(in) :: parameters
    end subroutine
  end interface
  type, extends(cross_validation_options) :: learning_curve_options
    real(dp), allocatable :: train_sizes(:)
    integer, allocatable :: train_counts(:)
    logical :: shuffle=.false.,exploit_incremental_learning=.false.,return_times=.false.
    integer :: random_state=0
    procedure(curve_partial_function), pointer, nopass :: partial_fit_func=>null()
    procedure(curve_initialize_function), pointer, nopass :: incremental_init_func=>null()
  end type
  type :: learning_curve_result
    integer, allocatable :: train_sizes(:),actual_train_sizes(:,:)
    type(index_split), allocatable :: folds(:)
    real(dp), allocatable :: train_score(:,:),test_score(:,:),fit_time(:,:),score_time(:,:)
  end type
  type :: validation_curve_result
    type(parameter_value), allocatable :: param_range(:)
    type(index_split), allocatable :: folds(:)
    real(dp), allocatable :: train_score(:,:),test_score(:,:)
  end type
  type :: permutation_score_result
    real(dp) :: score=0,pvalue=1
    real(dp), allocatable :: permutation_scores(:)
  end type
  interface learning_curve
    module procedure learning_real,learning_integer,learning_unsupervised
  end interface
  interface validation_curve
    module procedure curve_real,curve_integer,curve_unsupervised
  end interface
  interface permutation_test_score
    module procedure permutation_real,permutation_integer
  end interface
  interface cross_validate
    module procedure validate_real,validate_integer,validate_unsupervised
  end interface
  interface cross_val_predict
    module procedure predict_real,predict_integer
  end interface
  interface cross_val_score
    module procedure score_real,score_integer,score_unsupervised
  end interface
contains
  function labels_from_real(y) result(labels)
    real(dp), intent(in) :: y(:)
    integer, allocatable :: labels(:)
    call require(all(ieee_is_finite(y)),'CV classification targets must be finite')
    call require(all(y>=-real(huge(1),dp)-1).and.all(y<=real(huge(1),dp)), &
      'CV classification target exceeds integer range')
    call require(all(modulo(y,1._dp)==0),'CV classification targets must be integral')
    labels=int(y)
  end function
  subroutine prepare(estimator,x,options,settings,classifier,folds,yr,yi)
    class(*), intent(in) :: estimator
    real(dp), intent(in) :: x(:,:)
    type(cross_validation_options), optional, intent(in) :: options
    type(cross_validation_options), intent(out) :: settings
    logical, intent(out) :: classifier
    type(index_split), allocatable, intent(out) :: folds(:)
    real(dp), optional, intent(in) :: yr(:)
    integer, optional, intent(in) :: yi(:)
    type(kfold) :: splitter
    integer, allocatable :: labels(:)
    integer :: i
    call check_x(x)
    if(present(options)) settings=options
    call require(settings%error_score=='raise','CV supports only error_score=raise')
    select case(settings%task)
    case('auto');classifier=search_classifier(estimator)
    case('classification');classifier=.true.
    case('regression');classifier=.false.
    case default;call require(.false.,'CV task must be auto, classification or regression')
    end select
    if(present(yr)) then
      call check_xy(x,size(yr));call require(all(ieee_is_finite(yr)),'CV targets must be finite')
      if(classifier) labels=labels_from_real(yr)
    else if(present(yi)) then
      call check_xy(x,size(yi));if(classifier) labels=yi
    else
      call require(.not.classifier,'CV classifier requires targets')
    end if
    if(allocated(settings%cv_splits)) then
      folds=settings%cv_splits
    else if(classifier) then
      folds=classification_folds(labels,settings%cv)
    else
      splitter%n_splits=settings%cv;folds=splitter%split(size(x,1))
    end if
    call require(size(folds)>0,'CV requires at least one split')
    do i=1,size(folds)
      call require(allocated(folds(i)%train).and.allocated(folds(i)%test),'CV split indices are unallocated')
      call require(size(folds(i)%train)>0.and.size(folds(i)%test)>0,'CV splits must be nonempty')
      call require(all(folds(i)%train>=1).and.all(folds(i)%train<=size(x,1)),'CV train index out of bounds')
      call require(all(folds(i)%test>=1).and.all(folds(i)%test<=size(x,1)),'CV test index out of bounds')
    end do
  end subroutine
  function classification_folds(y,n) result(folds)
    integer, intent(in) :: y(:),n
    type(index_split), allocatable :: folds(:)
    integer, allocatable :: classes(:),first(:),order(:),counts(:),members(:),assigned(:)
    integer :: i,j,c,k,offset,width,start
    call require(n>=2.and.n<=size(y),'CV invalid stratified fold count')
    classes=unique_labels(y);allocate(first(size(classes)),counts(size(classes)),assigned(size(y)))
    do c=1,size(classes)
      first(c)=findloc(y,classes(c),dim=1);counts(c)=count(y==classes(c))
    end do
    call require(.not.all(counts<n),'CV fold count exceeds every class count')
    order=argsort(real(first,dp));offset=0
    do i=1,size(classes)
      c=order(i);members=pack([(j,j=1,size(y))],y==classes(c));start=1
      do j=1,n
        width=count(modulo([(offset+k-1,k=1,size(members))],n)==j-1)
        assigned(members(start:start+width-1))=j;start=start+width
      end do
      offset=offset+size(members)
    end do
    allocate(folds(n))
    do i=1,n
      folds(i)%test=pack([(j,j=1,size(y))],assigned==i)
      folds(i)%train=pack([(j,j=1,size(y))],assigned/=i)
    end do
  end function
  subroutine normalize_targets(classifier,real_y,integer_y,yr,yi)
    logical, intent(in) :: classifier
    real(dp), optional, intent(in) :: real_y(:)
    integer, optional, intent(in) :: integer_y(:)
    real(dp), allocatable, intent(out) :: yr(:)
    integer, allocatable, intent(out) :: yi(:)
    if(classifier) then
      if(present(integer_y)) then
        yi=integer_y
      else if(present(real_y)) then
        yi=labels_from_real(real_y)
      end if
    else
      if(present(real_y)) then
        yr=real_y
      else if(present(integer_y)) then
        yr=real(integer_y,dp)
      end if
    end if
  end subroutine
  subroutine check_weights(weights,n)
    real(dp), optional, intent(in) :: weights(:)
    integer, intent(in) :: n
    if(.not.present(weights)) return
    call require(size(weights)==n,'CV fit weight count mismatch')
    call require(all(ieee_is_finite(weights)),'CV fit weights must be finite')
  end subroutine
  subroutine fit_fold(estimator,model,x,settings,yr,yi,weights,original_integer)
    class(*), intent(in) :: estimator
    class(*), allocatable, intent(out) :: model
    real(dp), intent(in) :: x(:,:)
    type(cross_validation_options), intent(in) :: settings
    real(dp), optional, intent(in) :: yr(:),weights(:)
    integer, optional, intent(in) :: yi(:),original_integer(:)
    allocate(model,source=estimator)
    call fit_existing(model,x,settings,yr,yi,weights,original_integer)
  end subroutine
  subroutine fit_existing(model,x,settings,yr,yi,weights,original_integer)
    class(*), intent(inout) :: model
    real(dp), intent(in) :: x(:,:)
    type(cross_validation_options), intent(in) :: settings
    real(dp), optional, intent(in) :: yr(:),weights(:)
    integer, optional, intent(in) :: yi(:),original_integer(:)
    call reset_search_clone(model)
    if(associated(settings%fit_func)) then
      call settings%fit_func(model,x,yr,yi,weights)
    else if(.not.present(yr).and..not.present(yi)) then
      call require(.not.present(weights),'CV unsupervised default fit does not accept weights')
      call fit_model(model,x)
    else if(present(original_integer).and.present(yr)) then
      select type(model)
      type is(pipeline)
        call model%fit(x,original_integer,weights)
      class default
        call fit_search_native(model,x,yr,yi,weights)
      end select
    else
      call fit_search_native(model,x,yr,yi,weights)
    end if
  end subroutine
  function configure_scorers(settings,has_target) result(scorers)
    type(cross_validation_options), intent(in) :: settings
    logical, intent(in) :: has_target
    type(search_scorer), allocatable :: scorers(:)
    integer :: i,j
    if(allocated(settings%scorers)) then
      call require(.not.associated(settings%scorer_func),'CV choose scorers or scorer_func')
      call require(size(settings%scorers)>0,'CV scorers must be nonempty');scorers=settings%scorers
    else
      allocate(scorers(1));scorers(1)%scoring=settings%scoring;scorers(1)%func=>settings%scorer_func
    end if
    do i=1,size(scorers)
      call require(len_trim(scorers(i)%name)>0,'CV scorer names must be nonempty')
      do j=1,i-1
        call require(scorers(i)%name/=scorers(j)%name,'CV scorer names must be unique')
      end do
      if(.not.has_target) call require(associated(scorers(i)%func),'CV no-target scoring requires a callback')
    end do
  end function
  real(dp) function evaluate(scorer,model,x,yr,yi) result(value)
    type(search_scorer), intent(in) :: scorer
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: yr(:)
    integer, optional, intent(in) :: yi(:)
    if(associated(scorer%func)) then
      value=scorer%func(model,x,yr,yi)
    else
      value=native_search_score(model,x,scorer%scoring,yr,yi)
    end if
  end function
  real(dp) function clock_elapsed(start) result(seconds)
    integer(int64), intent(in) :: start
    integer(int64) :: now,rate
    call system_clock(now,rate);seconds=max(0._dp,real(now-start,dp)/real(rate,dp))
  end function
  function validate_core(estimator,x,options,sample_weight,real_y,integer_y) result(results)
    class(*), intent(in) :: estimator
    real(dp), intent(in) :: x(:,:)
    type(cross_validation_options), optional, intent(in) :: options
    real(dp), optional, intent(in) :: sample_weight(:),real_y(:)
    integer, optional, intent(in) :: integer_y(:)
    type(cross_validation_result) :: results
    type(cross_validation_options) :: settings
    type(index_split), allocatable :: folds(:)
    type(search_scorer), allocatable :: scorers(:)
    class(*), allocatable :: model
    real(dp), allocatable :: yr(:),train_y(:),test_y(:),weights(:)
    integer, allocatable :: yi(:),train_labels(:),test_labels(:),original_integer(:)
    integer :: i,j,nf,nm
    integer(int64) :: start
    logical :: classifier
    call prepare(estimator,x,options,settings,classifier,folds,real_y,integer_y)
    call check_weights(sample_weight,size(x,1));call normalize_targets(classifier,real_y,integer_y,yr,yi)
    scorers=configure_scorers(settings,present(real_y).or.present(integer_y))
    nf=size(folds);nm=size(scorers);results%n_splits=nf;results%n_features_in=size(x,2)
    allocate(results%test_score(nf,nm),results%fit_time(nf),results%score_time(nf))
    results%metric_names=scorers%name
    if(settings%return_train_score) allocate(results%train_score(nf,nm))
    if(settings%return_estimator) allocate(results%estimators(nf))
    if(settings%return_indices) results%indices=folds
    do i=1,nf
      associate(fold=>folds(i))
        if(allocated(yr)) then
          train_y=yr(fold%train);test_y=yr(fold%test)
          if(present(integer_y)) original_integer=integer_y(fold%train)
        end if
        if(allocated(yi)) then
          train_labels=yi(fold%train);test_labels=yi(fold%test)
        end if
        if(present(sample_weight)) weights=sample_weight(fold%train)
        call system_clock(start)
        call fit_fold(estimator,model,x(fold%train,:),settings,train_y,train_labels,weights,original_integer)
        results%fit_time(i)=clock_elapsed(start);call system_clock(start)
        do j=1,nm
          results%test_score(i,j)=evaluate(scorers(j),model,x(fold%test,:),test_y,test_labels)
        end do
        results%score_time(i)=clock_elapsed(start)
        if(settings%return_train_score) then
          do j=1,nm
            results%train_score(i,j)=evaluate(scorers(j),model,x(fold%train,:),train_y,train_labels)
          end do
        end if
        if(settings%return_estimator) call move_alloc(model,results%estimators(i)%model)
      end associate
    end do
  end function
  function validate_real(estimator,x,y,options,sample_weight) result(results)
    class(*), intent(in) :: estimator
    real(dp), intent(in) :: x(:,:),y(:)
    type(cross_validation_options), optional, intent(in) :: options
    real(dp), optional, intent(in) :: sample_weight(:)
    type(cross_validation_result) :: results
    results=validate_core(estimator,x,options,sample_weight,real_y=y)
  end function
  function validate_integer(estimator,x,y,options,sample_weight) result(results)
    class(*), intent(in) :: estimator
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    type(cross_validation_options), optional, intent(in) :: options
    real(dp), optional, intent(in) :: sample_weight(:)
    type(cross_validation_result) :: results
    results=validate_core(estimator,x,options,sample_weight,integer_y=y)
  end function
  function validate_unsupervised(estimator,x,options,sample_weight) result(results)
    class(*), intent(in) :: estimator
    real(dp), intent(in) :: x(:,:)
    type(cross_validation_options), optional, intent(in) :: options
    real(dp), optional, intent(in) :: sample_weight(:)
    type(cross_validation_result) :: results
    results=validate_core(estimator,x,options,sample_weight)
  end function
  subroutine require_single_scorer(options)
    type(cross_validation_options), optional, intent(in) :: options
    if(present(options)) call require(.not.allocated(options%scorers),'cross_val_score requires a single scalar scorer')
  end subroutine
  function score_settings(options) result(settings)
    type(cross_validation_options), optional, intent(in) :: options
    type(cross_validation_options) :: settings
    if(present(options)) settings=options
    settings%return_train_score=.false.;settings%return_estimator=.false.;settings%return_indices=.false.
  end function
  function score_real(estimator,x,y,options,sample_weight) result(scores)
    class(*), intent(in) :: estimator
    real(dp), intent(in) :: x(:,:),y(:)
    type(cross_validation_options), optional, intent(in) :: options
    real(dp), optional, intent(in) :: sample_weight(:)
    real(dp), allocatable :: scores(:)
    type(cross_validation_result) :: results
    type(cross_validation_options) :: settings
    call require_single_scorer(options);settings=score_settings(options)
    results=cross_validate(estimator,x,y,settings,sample_weight)
    scores=results%test_score(:,1)
  end function
  function score_integer(estimator,x,y,options,sample_weight) result(scores)
    class(*), intent(in) :: estimator
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    type(cross_validation_options), optional, intent(in) :: options
    real(dp), optional, intent(in) :: sample_weight(:)
    real(dp), allocatable :: scores(:)
    type(cross_validation_result) :: results
    type(cross_validation_options) :: settings
    call require_single_scorer(options);settings=score_settings(options)
    results=cross_validate(estimator,x,y,settings,sample_weight)
    scores=results%test_score(:,1)
  end function
  function score_unsupervised(estimator,x,options,sample_weight) result(scores)
    class(*), intent(in) :: estimator
    real(dp), intent(in) :: x(:,:)
    type(cross_validation_options), optional, intent(in) :: options
    real(dp), optional, intent(in) :: sample_weight(:)
    real(dp), allocatable :: scores(:)
    type(cross_validation_result) :: results
    type(cross_validation_options) :: settings
    call require_single_scorer(options);settings=score_settings(options)
    results=cross_validate(estimator,x,settings,sample_weight)
    scores=results%test_score(:,1)
  end function
  function log_probabilities(prob) result(values)
    real(dp), intent(in) :: prob(:,:)
    real(dp), allocatable :: values(:,:)
    integer :: i,j
    allocate(values(size(prob,1),size(prob,2)))
    do j=1,size(prob,2)
      do i=1,size(prob,1)
        call require(prob(i,j)>=0,'CV negative probability')
        if(prob(i,j)==0) then
          values(i,j)=ieee_value(0._dp,ieee_negative_inf)
        else
          values(i,j)=log(prob(i,j))
        end if
      end do
    end do
  end function
  recursive function native_log_response(model,x) result(values)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:,:),prepared(:,:)
    type(pipeline) :: prefix
    integer :: n
    select type(model)
    type is(pipeline)
      n=size(model%steps)
      call require(model%steps(n)%action=='model','CV Pipeline final passthrough has no log probabilities')
      if(n>1) then
        prefix=model%slice(1,n-1);prepared=prefix%transform(x)
      else
        prepared=x
      end if
      values=native_log_response(model%steps(n)%model,prepared)
    type is(gaussian_nb);values=model%predict_log_proba(x)
    class is(multinomial_nb);values=model%predict_log_proba(x)
    type is(gradient_boosting_classifier);values=model%predict_log_proba(x)
    class is(logistic_regression);values=log_probabilities(model%predict_proba(x))
    class is(random_forest_classifier);values=log_probabilities(model%predict_proba(x))
    class is(decision_tree_classifier);values=log_probabilities(model%predict_proba(x))
    type is(sgd_classifier);values=log_probabilities(model%predict_proba(x))
    type is(dummy_classifier);values=log_probabilities(model%predict_proba(x))
    class default
      call require(.false.,'CV estimator has no native predict_log_proba')
    end select
  end function
  function response(model,x,settings,classifier) result(values)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    type(cross_validation_options), intent(in) :: settings
    logical, intent(in) :: classifier
    real(dp), allocatable :: values(:,:),vector(:)
    integer, allocatable :: labels(:)
    if(associated(settings%response_func)) then
      values=settings%response_func(model,x,trim(settings%method));return
    end if
    select case(settings%method)
    case('predict')
      if(classifier) then
        labels=search_labels(model,x);values=reshape(real(labels,dp),[size(labels),1])
      else
        vector=search_regression(model,x);values=reshape(vector,[size(vector),1])
      end if
    case('predict_proba');values=search_probabilities(model,x)
    case('predict_log_proba');values=native_log_response(model,x)
    case('decision_function');values=search_decisions(model,x)
    end select
  end function
  function predict_core(estimator,x,options,sample_weight,real_y,integer_y) result(predictions)
    class(*), intent(in) :: estimator
    real(dp), intent(in) :: x(:,:)
    type(cross_validation_options), optional, intent(in) :: options
    real(dp), optional, intent(in) :: sample_weight(:),real_y(:)
    integer, optional, intent(in) :: integer_y(:)
    real(dp), allocatable :: predictions(:,:)
    type(cross_validation_options) :: settings
    type(index_split), allocatable :: folds(:)
    class(*), allocatable :: model
    real(dp), allocatable :: yr(:),train_y(:),weights(:),part(:,:),aligned(:,:)
    integer, allocatable :: yi(:),classes(:),train_labels(:),original_integer(:),fold_classes(:),visits(:)
    integer :: i,j,k,n,width
    logical :: classifier,encode
    call prepare(estimator,x,options,settings,classifier,folds,real_y,integer_y)
    call check_weights(sample_weight,size(x,1));call normalize_targets(classifier,real_y,integer_y,yr,yi)
    call require(any(settings%method==[character(24)::'predict','predict_proba','predict_log_proba','decision_function']), &
      'CV unsupported response method')
    call require(classifier.or.settings%method=='predict','CV classifier response requires a classifier')
    n=size(x,1);allocate(visits(n));visits=0
    do i=1,size(folds)
      do j=1,size(folds(i)%test)
        k=folds(i)%test(j);visits(k)=visits(k)+1
      end do
    end do
    call require(all(visits==1),'cross_val_predict requires test indices to form a partition')
    encode=classifier.and.settings%method/='predict'
    if(encode) then
      classes=unique_labels(yi);yi=[(label_index(classes,yi(i))-1,i=1,size(yi))]
    end if
    do i=1,size(folds)
      associate(fold=>folds(i))
        if(allocated(yr)) then
          train_y=yr(fold%train)
          if(present(integer_y)) original_integer=integer_y(fold%train)
        end if
        if(allocated(yi)) train_labels=yi(fold%train)
        if(present(sample_weight)) weights=sample_weight(fold%train)
        call fit_fold(estimator,model,x(fold%train,:),settings,train_y,train_labels,weights,original_integer)
        part=response(model,x(fold%test,:),settings,classifier)
        call require(size(part,1)==size(fold%test),'CV response sample count mismatch')
        if(encode) then
          if(associated(settings%classes_func)) then
            fold_classes=settings%classes_func(model)
          else
            fold_classes=search_classes(model)
          end if
          call require(size(fold_classes)>0,'CV fold classes must be nonempty')
          call require(all(fold_classes>=0).and.all(fold_classes<size(classes)),'CV encoded class index out of bounds')
          do j=1,size(fold_classes)
            call require(count(fold_classes==fold_classes(j))==1,'CV fold classes must be unique')
          end do
          if(size(fold_classes)/=size(classes)) then
            if(settings%method=='decision_function') then
              call require(size(part,2)==size(fold_classes),'CV irregular missing-class decision output')
              call require(size(fold_classes)>2,'CV missing-class decision function requires more than two fold classes')
            end if
            call require(size(part,2)==size(fold_classes),'CV probability class width mismatch')
            allocate(aligned(size(part,1),size(classes)));aligned=0
            if(settings%method/='predict_proba') aligned=-huge(1._dp)
            aligned(:,fold_classes+1)=part;call move_alloc(aligned,part)
          end if
        end if
        if(.not.allocated(predictions)) then
          width=size(part,2);allocate(predictions(n,width))
        end if
        call require(size(part,2)==size(predictions,2),'CV fold response widths differ')
        predictions(fold%test,:)=part
      end associate
    end do
  end function
  function predict_real(estimator,x,y,options,sample_weight) result(predictions)
    class(*), intent(in) :: estimator
    real(dp), intent(in) :: x(:,:),y(:)
    type(cross_validation_options), optional, intent(in) :: options
    real(dp), optional, intent(in) :: sample_weight(:)
    real(dp), allocatable :: predictions(:,:)
    predictions=predict_core(estimator,x,options,sample_weight,real_y=y)
  end function
  function predict_integer(estimator,x,y,options,sample_weight) result(predictions)
    class(*), intent(in) :: estimator
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    type(cross_validation_options), optional, intent(in) :: options
    real(dp), optional, intent(in) :: sample_weight(:)
    real(dp), allocatable :: predictions(:,:)
    predictions=predict_core(estimator,x,options,sample_weight,integer_y=y)
  end function
  function cross_val_predict_class(estimator,x,y,options,sample_weight) result(predictions)
    class(*), intent(in) :: estimator
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    type(cross_validation_options), optional, intent(in) :: options
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, allocatable :: predictions(:)
    real(dp), allocatable :: values(:,:)
    type(cross_validation_options) :: settings
    if(present(options)) settings=options
    call require(settings%method=='predict','cross_val_predict_class requires method=predict')
    if(settings%task=='auto') call require(search_classifier(estimator),'cross_val_predict_class requires a classifier')
    call require(settings%task/='regression','cross_val_predict_class requires a classifier')
    values=cross_val_predict(estimator,x,y,settings,sample_weight)
    call require(size(values,2)==1,'cross_val_predict_class requires one response column')
    predictions=labels_from_real(values(:,1))
  end function
  function absolute_train_sizes(settings,nmax) result(counts)
    type(learning_curve_options), intent(in) :: settings
    integer, intent(in) :: nmax
    integer, allocatable :: counts(:)
    real(dp), allocatable :: fractions(:)
    call require(.not.(allocated(settings%train_sizes).and.allocated(settings%train_counts)), &
      'learning_curve choose fractional sizes or absolute counts')
    if(allocated(settings%train_counts)) then
      call require(size(settings%train_counts)>0,'learning_curve training counts must be nonempty')
      call require(all(settings%train_counts>=1).and.all(settings%train_counts<=nmax), &
        'learning_curve training counts out of range')
      counts=unique_labels(settings%train_counts)
    else
      fractions=[.1_dp,.325_dp,.55_dp,.775_dp,1._dp]
      if(allocated(settings%train_sizes)) fractions=settings%train_sizes
      call require(size(fractions)>0,'learning_curve training fractions must be nonempty')
      call require(all(ieee_is_finite(fractions)),'learning_curve training fractions must be finite')
      call require(all(fractions>0).and.all(fractions<=1),'learning_curve training fractions must be in (0,1]')
      counts=unique_labels(max(1,min(nmax,int(fractions*nmax))))
    end if
  end function
  subroutine permute_indices(indices,rng)
    integer, intent(inout) :: indices(:)
    type(legacy_random_state), intent(inout) :: rng
    integer :: i,j,temporary
    do i=size(indices),2,-1
      j=rng%integer_index(i);temporary=indices(i);indices(i)=indices(j);indices(j)=temporary
    end do
  end subroutine
  logical function incremental_supported(estimator) result(supported)
    class(*), intent(in) :: estimator
    supported=.true.
    select type(estimator)
    type is(sgd_regressor)
    type is(sgd_classifier)
    type is(gaussian_nb)
    class default;supported=.false.
    end select
  end function
  subroutine initialize_incremental(model,settings)
    class(*), intent(inout) :: model
    type(learning_curve_options), intent(in) :: settings
    type(sgd_regressor) :: reg
    type(sgd_classifier) :: classifier
    type(gaussian_nb) :: bayes
    select type(model)
    type is(sgd_regressor)
      call reset_sgd_fit(model)
      call reset_boosting_fit(model)
    type is(sgd_classifier)
      call reset_sgd_fit(model)
      call reset_boosting_fit(model)
    type is(gaussian_nb)
      bayes%var_smoothing=model%var_smoothing
      if(allocated(model%priors)) bayes%priors=model%priors
      model=bayes
    end select
    if(associated(settings%incremental_init_func)) call settings%incremental_init_func(model)
  end subroutine
  subroutine partial_fit_curve(model,x,settings,yr,yi,classes,weights)
    class(*), intent(inout) :: model
    real(dp), intent(in) :: x(:,:)
    type(learning_curve_options), intent(in) :: settings
    real(dp), optional, intent(in) :: yr(:),weights(:)
    integer, optional, intent(in) :: yi(:),classes(:)
    if(associated(settings%partial_fit_func)) then
      call settings%partial_fit_func(model,x,yr,yi,classes,weights)
    else
      select type(model)
      type is(sgd_regressor)
        call require(present(yr),'incremental regression requires real targets');call model%partial_fit(x,yr,weights)
      type is(sgd_classifier)
        call require(present(yi),'incremental classification requires integer targets')
        call model%partial_fit(x,yi,classes,weights)
      type is(gaussian_nb)
        call require(present(yi),'incremental classification requires integer targets')
        call model%partial_fit(x,yi,classes,weights)
      class default
        call require(.false.,'learning_curve estimator has no native partial_fit')
      end select
    end if
  end subroutine
  function learning_core(estimator,x,options,sample_weight,real_y,integer_y) result(results)
    class(*), intent(in) :: estimator
    real(dp), intent(in) :: x(:,:)
    type(learning_curve_options), optional, intent(in) :: options
    real(dp), optional, intent(in) :: sample_weight(:),real_y(:)
    integer, optional, intent(in) :: integer_y(:)
    type(learning_curve_result) :: results
    type(learning_curve_options) :: settings
    type(cross_validation_options) :: base
    type(index_split), allocatable :: folds(:)
    type(search_scorer), allocatable :: scorers(:)
    type(legacy_random_state) :: rng
    class(*), allocatable :: model
    real(dp), allocatable :: yr(:),train_y(:),test_y(:),weights(:),partial_y(:)
    integer, allocatable :: yi(:),train_labels(:),test_labels(:),original_integer(:),partial_labels(:),classes(:),ids(:)
    integer :: i,j,ns,nf,nt,first,last,previous
    integer(int64) :: start
    real(dp) :: fit_time,score_time
    logical :: classifier
    if(present(options)) settings=options
    call require(settings%random_state>=0,'learning_curve random_state must be nonnegative')
    call require_single_scorer(settings%cross_validation_options)
    if(settings%exploit_incremental_learning) call require(associated(settings%partial_fit_func).or. &
      incremental_supported(estimator),'learning_curve estimator has no native partial_fit')
    call prepare(estimator,x,settings%cross_validation_options,base,classifier,folds,real_y,integer_y)
    call check_weights(sample_weight,size(x,1));call normalize_targets(classifier,real_y,integer_y,yr,yi)
    scorers=configure_scorers(base,present(real_y).or.present(integer_y))
    results%train_sizes=absolute_train_sizes(settings,size(folds(1)%train));ns=size(results%train_sizes);nf=size(folds)
    if(settings%shuffle) then
      call rng%seed(settings%random_state)
      do i=1,nf
        call permute_indices(folds(i)%train,rng)
      end do
    end if
    results%folds=folds;allocate(results%train_score(ns,nf),results%test_score(ns,nf),results%actual_train_sizes(ns,nf))
    if(settings%return_times) allocate(results%fit_time(ns,nf),results%score_time(ns,nf))
    if(classifier) classes=unique_labels(yi)
    do i=1,nf
      if(settings%exploit_incremental_learning) then
        if(allocated(model)) deallocate(model)
        allocate(model,source=estimator);call initialize_incremental(model,settings)
      end if
      previous=0
      do j=1,ns
        nt=min(results%train_sizes(j),size(folds(i)%train));ids=folds(i)%train(:nt)
        results%actual_train_sizes(j,i)=nt
        if(allocated(yr)) then
          train_y=yr(ids);test_y=yr(folds(i)%test)
          if(present(integer_y)) original_integer=integer_y(ids)
        end if
        if(allocated(yi)) then
          train_labels=yi(ids);test_labels=yi(folds(i)%test)
        end if
        if(settings%exploit_incremental_learning) then
          first=min(previous,size(folds(i)%train))+1;last=nt
          if(allocated(yr)) partial_y=yr(folds(i)%train(first:last))
          if(allocated(yi)) partial_labels=yi(folds(i)%train(first:last))
          call system_clock(start)
          ! Pinned sklearn 1.6.1 forwards fit parameters unsliced on this path.
          call partial_fit_curve(model,x(folds(i)%train(first:last),:),settings,partial_y,partial_labels, &
            classes,sample_weight)
        else
          if(present(sample_weight)) weights=sample_weight(ids)
          call system_clock(start)
          call fit_fold(estimator,model,x(ids,:),base,train_y,train_labels,weights,original_integer)
        end if
        fit_time=clock_elapsed(start);call system_clock(start)
        results%test_score(j,i)=evaluate(scorers(1),model,x(folds(i)%test,:),test_y,test_labels)
        score_time=clock_elapsed(start)
        results%train_score(j,i)=evaluate(scorers(1),model,x(ids,:),train_y,train_labels)
        if(settings%exploit_incremental_learning) score_time=clock_elapsed(start)
        if(settings%return_times) then
          results%fit_time(j,i)=fit_time;results%score_time(j,i)=score_time
        end if
        previous=results%train_sizes(j)
      end do
    end do
  end function
  function learning_real(estimator,x,y,options,sample_weight) result(results)
    class(*), intent(in) :: estimator
    real(dp), intent(in) :: x(:,:),y(:)
    type(learning_curve_options), optional, intent(in) :: options
    real(dp), optional, intent(in) :: sample_weight(:)
    type(learning_curve_result) :: results
    results=learning_core(estimator,x,options,sample_weight,real_y=y)
  end function
  function learning_integer(estimator,x,y,options,sample_weight) result(results)
    class(*), intent(in) :: estimator
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    type(learning_curve_options), optional, intent(in) :: options
    real(dp), optional, intent(in) :: sample_weight(:)
    type(learning_curve_result) :: results
    results=learning_core(estimator,x,options,sample_weight,integer_y=y)
  end function
  function learning_unsupervised(estimator,x,options,sample_weight) result(results)
    class(*), intent(in) :: estimator
    real(dp), intent(in) :: x(:,:)
    type(learning_curve_options), optional, intent(in) :: options
    real(dp), optional, intent(in) :: sample_weight(:)
    type(learning_curve_result) :: results
    results=learning_core(estimator,x,options,sample_weight)
  end function
  function curve_core(estimator,x,param_name,param_range,options,sample_weight,set_params_func,real_y,integer_y) &
      result(results)
    class(*), intent(in) :: estimator
    real(dp), intent(in) :: x(:,:)
    character(*), intent(in) :: param_name
    type(parameter_value), intent(in) :: param_range(:)
    type(cross_validation_options), optional, intent(in) :: options
    real(dp), optional, intent(in) :: sample_weight(:),real_y(:)
    integer, optional, intent(in) :: integer_y(:)
    procedure(curve_parameter_function), optional :: set_params_func
    type(validation_curve_result) :: results
    type(cross_validation_options) :: settings
    type(index_split), allocatable :: folds(:)
    type(search_scorer), allocatable :: scorers(:)
    type(parameter_set) :: parameters
    class(*), allocatable :: model
    real(dp), allocatable :: yr(:),train_y(:),test_y(:),weights(:)
    integer, allocatable :: yi(:),train_labels(:),test_labels(:),original_integer(:)
    integer :: i,j
    logical :: classifier
    call require(size(param_range)>0,'validation_curve param_range must be nonempty')
    call require_single_scorer(options)
    call prepare(estimator,x,options,settings,classifier,folds,real_y,integer_y)
    call check_weights(sample_weight,size(x,1));call normalize_targets(classifier,real_y,integer_y,yr,yi)
    scorers=configure_scorers(settings,present(real_y).or.present(integer_y))
    results%folds=folds;results%param_range=param_range
    allocate(results%train_score(size(param_range),size(folds)),results%test_score(size(param_range),size(folds)))
    allocate(parameters%items(1));parameters%items(1)%name=param_name
    do i=1,size(folds)
      if(allocated(yr)) then
        train_y=yr(folds(i)%train);test_y=yr(folds(i)%test)
        if(present(integer_y)) original_integer=integer_y(folds(i)%train)
      end if
      if(allocated(yi)) then
        train_labels=yi(folds(i)%train);test_labels=yi(folds(i)%test)
      end if
      if(present(sample_weight)) weights=sample_weight(folds(i)%train)
      do j=1,size(param_range)
        if(allocated(model)) deallocate(model)
        allocate(model,source=estimator);parameters%items(1)%value=param_range(j)
        if(present(set_params_func)) then
          call set_params_func(model,parameters)
        else
          call set_search_params(model,parameters)
        end if
        call fit_existing(model,x(folds(i)%train,:),settings,train_y,train_labels,weights,original_integer)
        results%test_score(j,i)=evaluate(scorers(1),model,x(folds(i)%test,:),test_y,test_labels)
        results%train_score(j,i)=evaluate(scorers(1),model,x(folds(i)%train,:),train_y,train_labels)
      end do
    end do
  end function
  function curve_real(estimator,x,y,param_name,param_range,options,sample_weight,set_params_func) result(results)
    class(*), intent(in) :: estimator
    real(dp), intent(in) :: x(:,:),y(:)
    character(*), intent(in) :: param_name
    type(parameter_value), intent(in) :: param_range(:)
    type(cross_validation_options), optional, intent(in) :: options
    real(dp), optional, intent(in) :: sample_weight(:)
    procedure(curve_parameter_function), optional :: set_params_func
    type(validation_curve_result) :: results
    results=curve_core(estimator,x,param_name,param_range,options,sample_weight,set_params_func,real_y=y)
  end function
  function curve_integer(estimator,x,y,param_name,param_range,options,sample_weight,set_params_func) result(results)
    class(*), intent(in) :: estimator
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    character(*), intent(in) :: param_name
    type(parameter_value), intent(in) :: param_range(:)
    type(cross_validation_options), optional, intent(in) :: options
    real(dp), optional, intent(in) :: sample_weight(:)
    procedure(curve_parameter_function), optional :: set_params_func
    type(validation_curve_result) :: results
    results=curve_core(estimator,x,param_name,param_range,options,sample_weight,set_params_func,integer_y=y)
  end function
  function curve_unsupervised(estimator,x,param_name,param_range,options,sample_weight,set_params_func) result(results)
    class(*), intent(in) :: estimator
    real(dp), intent(in) :: x(:,:)
    character(*), intent(in) :: param_name
    type(parameter_value), intent(in) :: param_range(:)
    type(cross_validation_options), optional, intent(in) :: options
    real(dp), optional, intent(in) :: sample_weight(:)
    procedure(curve_parameter_function), optional :: set_params_func
    type(validation_curve_result) :: results
    results=curve_core(estimator,x,param_name,param_range,options,sample_weight,set_params_func)
  end function
  real(dp) function permutation_average(estimator,x,options,sample_weight,real_y,integer_y) result(score)
    class(*), intent(in) :: estimator
    real(dp), intent(in) :: x(:,:)
    type(cross_validation_options), optional, intent(in) :: options
    real(dp), optional, intent(in) :: sample_weight(:),real_y(:)
    integer, optional, intent(in) :: integer_y(:)
    type(cross_validation_options) :: settings
    type(index_split), allocatable :: folds(:)
    type(search_scorer), allocatable :: scorers(:)
    class(*), allocatable :: model
    real(dp), allocatable :: yr(:),train_y(:),test_y(:),weights(:),scores(:)
    integer, allocatable :: yi(:),train_labels(:),test_labels(:),original_integer(:)
    integer :: i
    logical :: classifier
    call prepare(estimator,x,options,settings,classifier,folds,real_y,integer_y)
    call check_weights(sample_weight,size(x,1));call normalize_targets(classifier,real_y,integer_y,yr,yi)
    scorers=configure_scorers(settings,.true.);allocate(scores(size(folds)));allocate(model,source=estimator)
    do i=1,size(folds)
      if(allocated(yr)) then
        train_y=yr(folds(i)%train);test_y=yr(folds(i)%test)
        if(present(integer_y)) original_integer=integer_y(folds(i)%train)
      end if
      if(allocated(yi)) then
        train_labels=yi(folds(i)%train);test_labels=yi(folds(i)%test)
      end if
      if(present(sample_weight)) weights=sample_weight(folds(i)%train)
      call fit_existing(model,x(folds(i)%train,:),settings,train_y,train_labels,weights,original_integer)
      scores(i)=evaluate(scorers(1),model,x(folds(i)%test,:),test_y,test_labels)
    end do
    score=ieee_mean(scores)
  end function
  function permutation_core(estimator,x,options,n_permutations,random_state,groups,sample_weight,real_y,integer_y) &
      result(results)
    class(*), intent(in) :: estimator
    real(dp), intent(in) :: x(:,:)
    type(cross_validation_options), optional, intent(in) :: options
    integer, optional, intent(in) :: n_permutations,random_state,groups(:),integer_y(:)
    real(dp), optional, intent(in) :: sample_weight(:),real_y(:)
    type(permutation_score_result) :: results
    type(legacy_random_state) :: rng
    integer, allocatable :: ids(:),levels(:),members(:),shuffled(:),permuted_yi(:)
    real(dp), allocatable :: permuted_yr(:)
    integer :: nperm,seed,i,j,exceed
    nperm=100;seed=0
    if(present(n_permutations)) nperm=n_permutations
    if(present(random_state)) seed=random_state
    call require(nperm>0,'permutation_test_score n_permutations must be positive')
    call require_single_scorer(options);call rng%seed(seed)
    if(present(groups)) then
      call require(size(groups)==size(x,1),'permutation_test_score group count mismatch');levels=unique_labels(groups)
    end if
    results%score=permutation_average(estimator,x,options,sample_weight,real_y,integer_y)
    allocate(results%permutation_scores(nperm));exceed=0
    do i=1,nperm
      ids=[(j,j=1,size(x,1))]
      if(present(groups)) then
        do j=1,size(levels)
          members=pack(ids,groups==levels(j));shuffled=members;call permute_indices(shuffled,rng);ids(members)=shuffled
        end do
      else
        call permute_indices(ids,rng)
      end if
      if(present(real_y)) permuted_yr=real_y(ids)
      if(present(integer_y)) permuted_yi=integer_y(ids)
      results%permutation_scores(i)=permutation_average(estimator,x,options,sample_weight,permuted_yr,permuted_yi)
      if(.not.ieee_is_nan(results%score).and..not.ieee_is_nan(results%permutation_scores(i))) then
        if(results%permutation_scores(i)>=results%score) exceed=exceed+1
      end if
    end do
    results%pvalue=(real(exceed,dp)+1)/(real(nperm,dp)+1)
  end function
  function permutation_real(estimator,x,y,options,n_permutations,random_state,groups,sample_weight) result(results)
    class(*), intent(in) :: estimator
    real(dp), intent(in) :: x(:,:),y(:)
    type(cross_validation_options), optional, intent(in) :: options
    integer, optional, intent(in) :: n_permutations,random_state,groups(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    type(permutation_score_result) :: results
    results=permutation_core(estimator,x,options,n_permutations,random_state,groups,sample_weight,real_y=y)
  end function
  function permutation_integer(estimator,x,y,options,n_permutations,random_state,groups,sample_weight) result(results)
    class(*), intent(in) :: estimator
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    type(cross_validation_options), optional, intent(in) :: options
    integer, optional, intent(in) :: n_permutations,random_state,groups(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    type(permutation_score_result) :: results
    results=permutation_core(estimator,x,options,n_permutations,random_state,groups,sample_weight,integer_y=y)
  end function
end module
