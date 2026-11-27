! Multiclass reductions adapted from sklearn 1.6.1 multiclass.py (BSD-3-Clause).
! See THIRD_PARTY_NOTICES.md and LICENSES/scikit-learn-COPYING.
module sklearn_multiclass
  use sklearn_classic_gradient_boosting,only:reset_boosting_fit
  use sklearn_ieee_math, only: ieee_difference,ieee_product,ieee_quotient,ieee_total
  use sklearn_common, only: dp,require,check_x,check_xy,unique_labels,label_index
  use sklearn_linear_model, only: logistic_regression,perceptron,passive_aggressive_classifier,linear_regression
  use sklearn_ridge_classification, only: ridge_classifier
  use sklearn_sgd, only: sgd_classifier,sgd_regressor
  use sklearn_sgd_clone, only: clone_sgd_parameters,reset_sgd_fit
  use sklearn_svm, only: svc,linear_svc
  use sklearn_naive_bayes, only: gaussian_nb,multinomial_nb
  use sklearn_neighbors, only: kneighbors_classifier,kneighbors_regressor
  use sklearn_tree, only: decision_tree_classifier,decision_tree_regressor
  use sklearn_legacy_rng, only: legacy_random_state
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value,ieee_quiet_nan,ieee_negative_inf
  implicit none
  private
  public :: one_vs_rest_classifier,one_vs_one_classifier,output_code_classifier
  type :: binary_estimator
    class(*), allocatable :: estimator_
    logical :: constant_=.false.
    integer :: constant_value_=0,n_features_in_=0
  end type
  type :: one_vs_rest_classifier
    type(binary_estimator), allocatable :: estimators_(:)
    integer, allocatable :: classes_(:)
    integer :: n_features_in_=0,n_classes_=0
    logical :: multilabel_=.false.
    logical, private :: fitted_=.false.
  contains
    procedure, private :: ovr_fit_labels,ovr_fit_indicator,ovr_partial_labels,ovr_partial_indicator
    generic :: fit=>ovr_fit_labels,ovr_fit_indicator
    generic :: partial_fit=>ovr_partial_labels,ovr_partial_indicator
    procedure :: predict=>ovr_predict
    procedure :: predict_indicator=>ovr_predict_indicator
    procedure :: predict_proba=>ovr_proba
    procedure :: decision_function=>ovr_decision
    procedure, private :: ovr_score_labels,ovr_score_indicator
    generic :: score=>ovr_score_labels,ovr_score_indicator
  end type
  type :: one_vs_one_classifier
    type(binary_estimator), allocatable :: estimators_(:)
    integer, allocatable :: classes_(:)
    integer :: n_features_in_=0,n_classes_=0
    logical, private :: fitted_=.false.
  contains
    procedure :: fit=>ovo_fit
    procedure :: partial_fit=>ovo_partial
    procedure :: predict=>ovo_predict
    procedure :: decision_function=>ovo_decision
    procedure :: score=>ovo_score
  end type
  type :: output_code_classifier
    type(binary_estimator), allocatable :: estimators_(:)
    integer, allocatable :: classes_(:)
    real(dp), allocatable :: code_book_(:,:)
    real(dp) :: code_size=1.5_dp
    integer :: random_state=0,n_features_in_=0,n_classes_=0
    logical, private :: fitted_=.false.
  contains
    procedure :: fit=>code_fit
    procedure :: predict=>code_predict
    procedure :: score=>code_score
  end type
contains
  subroutine multiclass_weights(weights,n)
    real(dp), intent(in), optional :: weights(:)
    integer, intent(in) :: n
    if(.not.present(weights)) return
    call require(size(weights)==n,'multiclass sample weight length mismatch')
    call require(all(ieee_is_finite(weights)),'multiclass sample weights must be finite')
  end subroutine

  logical function is_regression(model) result(value)
    class(*), intent(in) :: model
    value=.true.
    select type(model)
    class is(linear_regression)
    type is(sgd_regressor)
    type is(kneighbors_regressor)
    class is(decision_tree_regressor)
    class default
      value=.false.
    end select
  end function

  logical function has_decision(model) result(value)
    class(*), intent(in) :: model
    value=.true.
    select type(model)
    class is(logistic_regression)
    type is(sgd_classifier)
    type is(perceptron)
    type is(passive_aggressive_classifier)
    class is(ridge_classifier)
    type is(linear_svc)
    type is(svc)
    class default
      value=.false.
    end select
  end function

  logical function has_probability(model) result(value)
    class(*), intent(in) :: model
    value=.true.
    select type(model)
    class is(logistic_regression)
    type is(gaussian_nb)
    class is(multinomial_nb)
    type is(kneighbors_classifier)
    class is(decision_tree_classifier)
    type is(sgd_classifier)
      value=model%loss=='log_loss'.or.model%loss=='modified_huber'
    class default
      value=.false.
    end select
  end function

  subroutine supported(model,weighted,streaming)
    class(*), intent(in) :: model
    logical, intent(in) :: weighted,streaming
    select type(model)
    type is(gaussian_nb)
      return
    type is(sgd_classifier)
      return
    type is(sgd_regressor)
      call require(.not.streaming,'multiclass partial_fit requires a classifier base'); return
    class is(logistic_regression)
    class is(ridge_classifier)
    type is(linear_svc)
    type is(perceptron)
      call require(.not.streaming,'multiclass base has no native partial_fit')
      call require(.not.weighted,'multiclass base has no native weighted fit'); return
    type is(passive_aggressive_classifier)
      call require(.not.streaming,'multiclass base has no native partial_fit')
      call require(.not.weighted,'multiclass base has no native weighted fit'); return
    type is(svc)
      call require(.not.streaming,'multiclass base has no native partial_fit')
      call require(.not.weighted,'multiclass base has no native weighted fit'); return
    type is(kneighbors_classifier)
      call require(.not.streaming,'multiclass base has no native partial_fit')
      call require(.not.weighted,'multiclass base has no native weighted fit'); return
    class is(decision_tree_classifier)
      call require(.not.streaming,'multiclass base has no native partial_fit'); return
    class is(multinomial_nb)
      call require(.not.streaming,'multiclass base has no native partial_fit')
      call require(.not.weighted,'multiclass base has no native weighted fit'); return
    class is(linear_regression)
      call require(.not.streaming,'multiclass base has no native partial_fit')
      call require(.not.weighted,'multiclass base has no native weighted fit'); return
    type is(kneighbors_regressor)
      call require(.not.streaming,'multiclass base has no native partial_fit')
      call require(.not.weighted,'multiclass base has no native weighted fit'); return
    class is(decision_tree_regressor)
      call require(.not.streaming,'multiclass base has no native partial_fit'); return
    class default
      call require(.false.,'unsupported native multiclass base')
    end select
    call require(.not.streaming,'multiclass base has no native partial_fit')
  end subroutine

  subroutine binary_fit(slot,x,y,base,weights)
    type(binary_estimator), intent(out) :: slot
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    class(*), intent(in) :: base
    real(dp), intent(in), optional :: weights(:)
    slot%n_features_in_=size(x,2)
    if(all(y==y(1))) then
      slot%constant_=.true.; slot%constant_value_=y(1); return
    end if
    allocate(slot%estimator_,source=base)
    call reset_sgd_fit(slot%estimator_)
    call reset_boosting_fit(slot%estimator_)
    call train(slot%estimator_,x,y,weights,.false.)
  end subroutine

  subroutine train(model,x,y,weights,streaming)
    class(*), intent(inout) :: model
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(in), optional :: weights(:)
    logical, intent(in) :: streaming
    select type(model)
    type is(gaussian_nb)
      if(streaming) then
        call model%partial_fit(x,y,[0,1],weights)
      else
        call model%fit(x,y,weights)
      end if
    type is(sgd_classifier)
      if(streaming) then
        call model%partial_fit(x,y,[0,1],weights)
      else
        call model%fit(x,y,weights)
      end if
    type is(sgd_regressor)
      if(streaming) then
        call model%partial_fit(x,real(y,dp),weights)
      else
        call model%fit(x,real(y,dp),weights)
      end if
    class is(logistic_regression)
      call model%fit(x,y,weights)
    class is(ridge_classifier)
      call model%fit(x,y,weights)
    type is(linear_svc)
      call model%fit(x,y,weights)
    type is(perceptron)
      call model%fit(x,y)
    type is(passive_aggressive_classifier)
      call model%fit(x,y)
    type is(svc)
      call model%fit(x,y)
    type is(kneighbors_classifier)
      call model%fit(x,y)
    class is(decision_tree_classifier)
      call model%fit(x,y,weights)
    class is(multinomial_nb)
      call model%fit(x,y)
    class is(linear_regression)
      call model%fit(x,real(y,dp))
    type is(kneighbors_regressor)
      call model%fit(x,real(y,dp))
    class is(decision_tree_regressor)
      call model%fit(x,real(y,dp),weights)
    end select
  end subroutine

  subroutine fresh_slot(slot,base)
    type(binary_estimator), intent(out) :: slot
    class(*), intent(in) :: base
    type(gaussian_nb) :: nb
    type(sgd_classifier) :: clf
    select type(base)
    type is(gaussian_nb)
      nb%var_smoothing=base%var_smoothing
      if(allocated(base%priors)) nb%priors=base%priors
      allocate(slot%estimator_,source=nb)
    type is(sgd_classifier)
      clf=clone_sgd_parameters(base)
      allocate(slot%estimator_,source=clf)
    class default
      call require(.false.,'multiclass base has no native partial_fit')
    end select
  end subroutine

  subroutine binary_partial(slot,x,y,weights)
    type(binary_estimator), intent(inout) :: slot
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(in), optional :: weights(:)
    call require(.not.slot%constant_,'constant multiclass head has no partial_fit')
    call require(allocated(slot%estimator_),'multiclass streaming head is missing')
    call supported(slot%estimator_,present(weights),.true.)
    call train(slot%estimator_,x,y,weights,.true.); slot%n_features_in_=size(x,2)
  end subroutine

  function predict_base(model,x) result(values)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:)
    select type(model)
    class is(logistic_regression)
      values=model%predict(x)
    type is(gaussian_nb)
      values=real(model%predict(x),dp)
    type is(sgd_classifier)
      values=real(model%predict(x),dp)
    type is(sgd_regressor)
      values=model%predict(x)
    type is(perceptron)
      values=real(model%predict(x),dp)
    type is(passive_aggressive_classifier)
      values=real(model%predict(x),dp)
    class is(ridge_classifier)
      values=real(model%predict(x),dp)
    type is(linear_svc)
      values=real(model%predict(x),dp)
    type is(svc)
      values=real(model%predict(x),dp)
    type is(kneighbors_classifier)
      values=real(model%predict(x),dp)
    class is(decision_tree_classifier)
      values=real(model%predict(x),dp)
    class is(multinomial_nb)
      values=real(model%predict(x),dp)
    class is(linear_regression)
      values=model%predict(x)
    type is(kneighbors_regressor)
      values=model%predict(x)
    class is(decision_tree_regressor)
      values=model%predict(x)
    class default
      call require(.false.,'unsupported native multiclass base')
    end select
  end function

  function decision_base(model,x) result(values)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:),matrix(:,:)
    call require(has_decision(model),'multiclass base lacks decision_function')
    select type(model)
    class is(logistic_regression)
      matrix=model%decision_function(x)
    type is(sgd_classifier)
      matrix=model%decision_function(x)
    type is(perceptron)
      matrix=model%decision_function(x)
    type is(passive_aggressive_classifier)
      matrix=model%decision_function(x)
    class is(ridge_classifier)
      matrix=model%decision_function(x)
    type is(linear_svc)
      matrix=model%decision_function(x)
    type is(svc)
      matrix=model%decision_function(x)
    end select
    call require(size(matrix,2)==1,'multiclass binary head decision must be scalar')
    values=matrix(:,1)
  end function

  function probability_base(model,x) result(values)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:),matrix(:,:)
    call require(has_probability(model),'multiclass base lacks predict_proba')
    select type(model)
    class is(logistic_regression)
      matrix=model%predict_proba(x)
    type is(gaussian_nb)
      matrix=model%predict_proba(x)
    type is(sgd_classifier)
      matrix=model%predict_proba(x)
    type is(kneighbors_classifier)
      matrix=model%predict_proba(x)
    class is(decision_tree_classifier)
      matrix=model%predict_proba(x)
    class is(multinomial_nb)
      matrix=model%predict_proba(x)
    end select
    call require(size(matrix,2)==2,'multiclass binary head probability must have two columns')
    values=matrix(:,2)
  end function

  function binary_response(slot,x,method) result(values)
    type(binary_estimator), intent(in) :: slot
    real(dp), intent(in) :: x(:,:)
    character(*), intent(in) :: method
    real(dp), allocatable :: values(:)
    call require(slot%n_features_in_>0,'multiclass head is not fitted')
    call require(size(x,2)==slot%n_features_in_,'multiclass head feature count mismatch')
    if(slot%constant_) then
      allocate(values(size(x,1))); values=real(slot%constant_value_,dp); return
    end if
    call require(allocated(slot%estimator_),'multiclass head is not fitted')
    select case(method)
    case('predict')
      values=predict_base(slot%estimator_,x)
    case('decision')
      values=decision_base(slot%estimator_,x)
    case('proba')
      values=probability_base(slot%estimator_,x)
    case default
      if(is_regression(slot%estimator_)) then
        values=predict_base(slot%estimator_,x)
      else if(has_decision(slot%estimator_)) then
        values=decision_base(slot%estimator_,x)
      else
        values=probability_base(slot%estimator_,x)
      end if
    end select
  end function

  real(dp) function binary_threshold(slot) result(value)
    type(binary_estimator), intent(in) :: slot
    value=.5_dp
    if(slot%constant_) return
    value=0
    if(is_regression(slot%estimator_).or..not.has_decision(slot%estimator_)) value=.5_dp
  end function

  subroutine fitted_input(fitted,features,x)
    logical, intent(in) :: fitted
    integer, intent(in) :: features
    real(dp), intent(in) :: x(:,:)
    call require(fitted,'multiclass estimator is not fitted'); call check_x(x)
    call require(size(x,2)==features,'multiclass feature count mismatch')
  end subroutine

  function binarize_labels(y,classes) result(targets)
    integer, intent(in) :: y(:),classes(:)
    integer, allocatable :: targets(:,:)
    integer :: heads,j
    heads=size(classes); if(heads<=2) heads=1
    allocate(targets(size(y),heads)); targets=0
    if(size(classes)==1) return
    do j=1,heads
      if(size(classes)==2) then
        targets(:,j)=merge(1,0,y==classes(2))
      else
        targets(:,j)=merge(1,0,y==classes(j))
      end if
    end do
  end function

  subroutine declare_classes(current,classes,y,first)
    integer, allocatable, intent(inout) :: current(:)
    integer, intent(in), optional :: classes(:)
    integer, intent(in) :: y(:)
    logical, intent(in) :: first
    integer, allocatable :: declared(:)
    integer :: j
    if(first) then
      call require(present(classes),'first multiclass partial_fit requires classes')
      call require(size(classes)>0,'multiclass classes must not be empty')
      current=unique_labels(classes)
    else if(present(classes)) then
      declared=unique_labels(classes)
      call require(size(declared)==size(current),'multiclass partial_fit classes changed')
      call require(all(declared==current),'multiclass partial_fit classes changed')
    end if
    do j=1,size(y)
      call require(any(current==y(j)),'multiclass batch contains an undeclared class')
    end do
  end subroutine

  real(dp) function subset_accuracy(correct,weights) result(value)
    logical, intent(in) :: correct(:)
    real(dp), intent(in), optional :: weights(:)
    real(dp) :: w(size(correct))
    call require(size(correct)>0,'multiclass score requires samples'); w=1
    call multiclass_weights(weights,size(correct))
    if(present(weights)) then
      call require(maxval(abs(weights))<=huge(1._dp)/(16*real(size(correct),dp)), &
        'multiclass score weight mass exceeds working range')
      w=weights
    end if
    call require(sum(w)/=0,'multiclass score weights require nonzero total')
    value=sum(w*merge(1._dp,0._dp,correct))/sum(w)
  end function

  subroutine ovr_fit_labels(self,x,y,estimator,sample_weight)
    class(one_vs_rest_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    class(*), intent(in) :: estimator
    real(dp), intent(in), optional :: sample_weight(:)
    integer, allocatable :: targets(:,:)
    integer :: j
    self%fitted_=.false.; call check_xy(x,size(y)); call multiclass_weights(sample_weight,size(y))
    call supported(estimator,present(sample_weight),.false.)
    self%classes_=unique_labels(y); self%n_classes_=size(self%classes_); self%multilabel_=.false.
    targets=binarize_labels(y,self%classes_)
    if(allocated(self%estimators_)) deallocate(self%estimators_)
    allocate(self%estimators_(size(targets,2)))
    do j=1,size(targets,2)
      call binary_fit(self%estimators_(j),x,targets(:,j),estimator,sample_weight)
    end do
    self%n_features_in_=size(x,2); self%fitted_=.true.
  end subroutine

  subroutine ovr_fit_indicator(self,x,y,estimator,sample_weight)
    class(one_vs_rest_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:,:)
    class(*), intent(in) :: estimator
    real(dp), intent(in), optional :: sample_weight(:)
    integer :: j
    call check_xy(x,size(y,1)); call require(size(y,2)>0,'multiclass indicator targets need columns')
    if(size(y,2)==1) then
      call self%ovr_fit_labels(x,y(:,1),estimator,sample_weight); return
    end if
    self%fitted_=.false.; call multiclass_weights(sample_weight,size(y,1))
    call require(size(unique_labels(reshape(y,[size(y)])))<=2,'OVR matrix targets must be a multilabel indicator')
    call supported(estimator,present(sample_weight),.false.)
    self%classes_=[(j-1,j=1,size(y,2))]; self%n_classes_=size(y,2); self%multilabel_=.true.
    if(allocated(self%estimators_)) deallocate(self%estimators_)
    allocate(self%estimators_(size(y,2)))
    do j=1,size(y,2)
      call binary_fit(self%estimators_(j),x,y(:,j),estimator,sample_weight)
    end do
    self%n_features_in_=size(x,2); self%fitted_=.true.
  end subroutine

  subroutine ovr_partial_labels(self,x,y,estimator,classes,sample_weight)
    class(one_vs_rest_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    class(*), intent(in), optional :: estimator
    integer, intent(in), optional :: classes(:)
    real(dp), intent(in), optional :: sample_weight(:)
    integer, allocatable :: targets(:,:)
    integer :: j
    logical :: first
    call check_xy(x,size(y)); call multiclass_weights(sample_weight,size(y)); first=.not.self%fitted_
    call require(.not.self%multilabel_,'OVR multilabel model requires matrix partial_fit targets')
    call declare_classes(self%classes_,classes,y,first)
    targets=binarize_labels(y,self%classes_)
    if(first) then
      call require(present(estimator),'first multiclass partial_fit requires an estimator')
      call supported(estimator,present(sample_weight),.true.)
      if(allocated(self%estimators_)) deallocate(self%estimators_)
      allocate(self%estimators_(size(targets,2)))
      do j=1,size(targets,2)
        call fresh_slot(self%estimators_(j),estimator)
      end do
    else
      call require(size(x,2)==self%n_features_in_,'multiclass feature count mismatch')
    end if
    do j=1,size(targets,2)
      call binary_partial(self%estimators_(j),x,targets(:,j),sample_weight)
    end do
    self%n_features_in_=size(x,2); self%n_classes_=size(self%classes_); self%fitted_=.true.
  end subroutine

  subroutine ovr_partial_indicator(self,x,y,estimator,classes,sample_weight)
    class(one_vs_rest_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:,:)
    class(*), intent(in), optional :: estimator
    integer, intent(in), optional :: classes(:)
    real(dp), intent(in), optional :: sample_weight(:)
    integer :: j
    call check_xy(x,size(y,1)); call require(size(y,2)>0,'multiclass indicator targets need columns')
    if(size(y,2)==1) then
      call self%ovr_partial_labels(x,y(:,1),estimator,classes,sample_weight); return
    end if
    call require(self%fitted_.and.self%multilabel_,'OVR multilabel partial_fit requires an initial multilabel fit')
    call require(size(x,2)==self%n_features_in_,'multiclass feature count mismatch')
    call require(size(y,2)==size(self%estimators_),'OVR indicator output count changed')
    call require(size(unique_labels(reshape(y,[size(y)])))<=2,'OVR matrix targets must be a multilabel indicator')
    call multiclass_weights(sample_weight,size(y,1))
    call declare_classes(self%classes_,classes,reshape(y,[size(y)]),.false.)
    do j=1,size(y,2)
      call binary_partial(self%estimators_(j),x,y(:,j),sample_weight)
    end do
  end subroutine

  function ovr_responses(self,x,method) result(values)
    class(one_vs_rest_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    character(*), intent(in) :: method
    real(dp), allocatable :: values(:,:)
    integer :: j
    call fitted_input(self%fitted_,self%n_features_in_,x)
    allocate(values(size(x,1),size(self%estimators_)))
    do j=1,size(self%estimators_)
      values(:,j)=binary_response(self%estimators_(j),x,method)
    end do
  end function

  function ovr_decision(self,x) result(values)
    class(one_vs_rest_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:,:)
    values=ovr_responses(self,x,'decision')
  end function

  function ovr_proba(self,x) result(values)
    class(one_vs_rest_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:,:),raw(:,:)
    real(dp) :: total
    integer :: i,j
    raw=ovr_responses(self,x,'proba')
    if(size(raw,2)==1) then
      allocate(values(size(x,1),2)); values(:,1)=1-raw(:,1); values(:,2)=raw(:,1)
    else
      values=raw
    end if
    if(.not.self%multilabel_) then
      do i=1,size(values,1)
        total=ieee_total(values(i,:))
        do j=1,size(values,2)
          values(i,j)=ieee_quotient(values(i,j),total)
        end do
      end do
    end if
  end function

  function ovr_predict(self,x) result(labels)
    class(one_vs_rest_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: labels(:)
    real(dp), allocatable :: values(:,:)
    real(dp) :: maximum,threshold
    integer :: i,j,best
    call require(.not.self%multilabel_,'OVR multilabel model requires predict_indicator')
    values=ovr_responses(self,x,'binary'); allocate(labels(size(x,1)))
    if(self%n_classes_<=2) then
      labels=self%classes_(1)
      if(self%n_classes_==1) return
      threshold=binary_threshold(self%estimators_(1))
      do i=1,size(labels)
        if(ieee_is_nan(values(i,1))) cycle
        if(values(i,1)>threshold) labels(i)=self%classes_(2)
      end do
    else
      do i=1,size(labels)
        maximum=ieee_value(0._dp,ieee_negative_inf); best=1
        do j=1,size(values,2)
          if(ieee_is_nan(maximum)) cycle
          if(ieee_is_nan(values(i,j))) then
            maximum=values(i,j); cycle
          end if
          if(values(i,j)>=maximum) then
            maximum=values(i,j); best=j
          end if
        end do
        labels(i)=self%classes_(best)
      end do
    end if
  end function

  function ovr_predict_indicator(self,x) result(labels)
    class(one_vs_rest_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: labels(:,:)
    real(dp), allocatable :: values(:,:)
    real(dp) :: threshold
    integer :: i,j
    call require(self%multilabel_,'OVR predict_indicator requires a multilabel model')
    values=ovr_responses(self,x,'binary'); threshold=binary_threshold(self%estimators_(1))
    allocate(labels(size(values,1),size(values,2))); labels=0
    do j=1,size(values,2)
      do i=1,size(values,1)
        if(ieee_is_nan(values(i,j))) cycle
        if(values(i,j)>threshold) labels(i,j)=1
      end do
    end do
  end function

  real(dp) function ovr_score_labels(self,x,y,sample_weight) result(value)
    class(one_vs_rest_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    integer, allocatable :: predicted(:)
    predicted=self%predict(x); call require(size(y)==size(predicted),'multiclass score target length mismatch')
    value=subset_accuracy(y==predicted,sample_weight)
  end function

  real(dp) function ovr_score_indicator(self,x,y,sample_weight) result(value)
    class(one_vs_rest_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:,:)
    real(dp), intent(in), optional :: sample_weight(:)
    integer, allocatable :: predicted(:,:)
    if(.not.self%multilabel_.and.size(y,2)==1) then
      value=self%ovr_score_labels(x,y(:,1),sample_weight); return
    end if
    predicted=self%predict_indicator(x)
    call require(all(shape(y)==shape(predicted)),'multiclass score target shape mismatch')
    call require(size(unique_labels(reshape(y,[size(y)])))<=2,'multiclass score requires indicator targets')
    value=subset_accuracy(all(y==predicted,dim=2),sample_weight)
  end function

  subroutine ovo_fit(self,x,y,estimator,sample_weight)
    class(one_vs_one_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    class(*), intent(in) :: estimator
    real(dp), intent(in), optional :: sample_weight(:)
    integer, allocatable :: rows(:),target(:),ids(:)
    integer :: i,j,k,nclasses
    self%fitted_=.false.; call check_xy(x,size(y)); call multiclass_weights(sample_weight,size(y))
    call supported(estimator,present(sample_weight),.false.)
    self%classes_=unique_labels(y); nclasses=size(self%classes_)
    call require(nclasses>=2,'OVO fitting requires at least two classes')
    call require(real(nclasses,dp)*(nclasses-1)/2<=huge(1),'multiclass pair count exceeds integer range')
    if(allocated(self%estimators_)) deallocate(self%estimators_)
    allocate(self%estimators_(int(real(nclasses,dp)*(nclasses-1)/2))); ids=[(i,i=1,size(y))]; k=0
    do i=1,nclasses
      do j=i+1,nclasses
        k=k+1; rows=pack(ids,y==self%classes_(i).or.y==self%classes_(j))
        target=merge(1,0,y(rows)==self%classes_(j))
        if(present(sample_weight)) then
          call binary_fit(self%estimators_(k),x(rows,:),target,estimator,sample_weight(rows))
        else
          call binary_fit(self%estimators_(k),x(rows,:),target,estimator)
        end if
      end do
    end do
    self%n_classes_=nclasses; self%n_features_in_=size(x,2); self%fitted_=.true.
  end subroutine

  subroutine ovo_partial(self,x,y,estimator,classes,sample_weight)
    class(one_vs_one_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    class(*), intent(in), optional :: estimator
    integer, intent(in), optional :: classes(:)
    real(dp), intent(in), optional :: sample_weight(:)
    integer, allocatable :: rows(:),target(:),ids(:)
    integer :: i,j,k,nclasses
    logical :: first
    call check_xy(x,size(y)); call multiclass_weights(sample_weight,size(y)); first=.not.self%fitted_
    call declare_classes(self%classes_,classes,y,first); nclasses=size(self%classes_)
    call require(nclasses>=2,'OVO fitting requires at least two classes')
    call require(real(nclasses,dp)*(nclasses-1)/2<=huge(1),'multiclass pair count exceeds integer range')
    if(first) then
      call require(present(estimator),'first multiclass partial_fit requires an estimator')
      call supported(estimator,present(sample_weight),.true.)
      if(allocated(self%estimators_)) deallocate(self%estimators_)
      allocate(self%estimators_(int(real(nclasses,dp)*(nclasses-1)/2)))
      do k=1,size(self%estimators_)
        call fresh_slot(self%estimators_(k),estimator)
      end do
    else
      call require(size(x,2)==self%n_features_in_,'multiclass feature count mismatch')
    end if
    ids=[(i,i=1,size(y))]; k=0
    do i=1,nclasses
      do j=i+1,nclasses
        k=k+1; rows=pack(ids,y==self%classes_(i).or.y==self%classes_(j))
        if(size(rows)==0) cycle
        target=merge(1,0,y(rows)==self%classes_(j))
        if(present(sample_weight)) then
          call binary_partial(self%estimators_(k),x(rows,:),target,sample_weight(rows))
        else
          call binary_partial(self%estimators_(k),x(rows,:),target)
        end if
      end do
    end do
    self%n_classes_=nclasses; self%n_features_in_=size(x,2); self%fitted_=.true.
  end subroutine

  function ovo_decision(self,x) result(values)
    class(one_vs_one_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:,:),votes(:,:),confidence(:,:),score(:),pred(:)
    integer :: i,j,k,row
    call fitted_input(self%fitted_,self%n_features_in_,x)
    allocate(votes(size(x,1),self%n_classes_),confidence(size(x,1),self%n_classes_)); votes=0; confidence=0; k=0
    do i=1,self%n_classes_
      do j=i+1,self%n_classes_
        k=k+1; pred=binary_response(self%estimators_(k),x,'predict')
        score=binary_response(self%estimators_(k),x,'binary')
        do row=1,size(x,1)
          confidence(row,i)=ieee_difference(confidence(row,i),score(row))
          confidence(row,j)=ieee_difference(confidence(row,j),-score(row))
          if(ieee_is_nan(pred(row))) cycle
          if(pred(row)==0) votes(row,i)=votes(row,i)+1
          if(pred(row)==1) votes(row,j)=votes(row,j)+1
        end do
      end do
    end do
    allocate(values(size(x,1),self%n_classes_))
    do i=1,self%n_classes_
      do row=1,size(x,1)
        values(row,i)=ieee_difference(votes(row,i), &
          -ieee_quotient(confidence(row,i),ieee_product(3._dp,ieee_difference(abs(confidence(row,i)),-1._dp))))
      end do
    end do
    if(self%n_classes_==2) values=values(:,2:2)
  end function

  function ovo_predict(self,x) result(labels)
    class(one_vs_one_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: labels(:)
    real(dp), allocatable :: values(:,:)
    real(dp) :: threshold
    integer :: row,j,best
    values=self%decision_function(x); allocate(labels(size(x,1)))
    if(self%n_classes_==2) then
      threshold=binary_threshold(self%estimators_(1)); labels=self%classes_(1)
      do row=1,size(labels)
        if(ieee_is_nan(values(row,1))) cycle
        if(values(row,1)>threshold) labels(row)=self%classes_(2)
      end do
    else
      do row=1,size(labels)
        best=0
        do j=1,size(values,2)
          if(.not.ieee_is_nan(values(row,j))) cycle
          best=j; exit
        end do
        if(best==0) best=maxloc(values(row,:),dim=1)
        labels(row)=self%classes_(best)
      end do
    end if
  end function

  real(dp) function ovo_score(self,x,y,sample_weight) result(value)
    class(one_vs_one_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    integer, allocatable :: predicted(:)
    predicted=self%predict(x); call require(size(y)==size(predicted),'multiclass score target length mismatch')
    value=subset_accuracy(y==predicted,sample_weight)
  end function

  subroutine code_fit(self,x,y,estimator,sample_weight)
    class(output_code_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    class(*), intent(in) :: estimator
    real(dp), intent(in), optional :: sample_weight(:)
    type(legacy_random_state) :: rng
    integer, allocatable :: target(:),encoded(:)
    integer :: i,j,bits
    real(dp) :: negative
    self%fitted_=.false.; call check_xy(x,size(y)); call multiclass_weights(sample_weight,size(y))
    call supported(estimator,present(sample_weight),.false.)
    call require(has_decision(estimator).or.has_probability(estimator), &
      'output-code base requires decision_function or predict_proba')
    call require(ieee_is_finite(self%code_size),'output-code size must be finite')
    call require(self%code_size>0,'output-code size must be positive')
    call require(self%random_state>=0,'output-code random_state must be nonnegative')
    self%classes_=unique_labels(y); self%n_classes_=size(self%classes_)
    call require(self%code_size<=real(huge(1)-1,dp)/self%n_classes_,'output-code bit count exceeds integer range')
    bits=int(self%n_classes_*self%code_size); call require(bits>=1,'output-code size produces no binary heads')
    if(allocated(self%code_book_)) deallocate(self%code_book_)
    if(allocated(self%estimators_)) deallocate(self%estimators_)
    allocate(self%code_book_(self%n_classes_,bits),self%estimators_(bits))
    call rng%seed(self%random_state); negative=0
    if(has_decision(estimator)) negative=-1
    do i=1,self%n_classes_
      do j=1,bits
        self%code_book_(i,j)=negative
        if(rng%uniform()>.5_dp) self%code_book_(i,j)=1
      end do
    end do
    encoded=[(label_index(self%classes_,y(i)),i=1,size(y))]
    do j=1,bits
      target=int(self%code_book_(encoded,j))
      call binary_fit(self%estimators_(j),x,target,estimator,sample_weight)
    end do
    self%n_features_in_=size(x,2); self%fitted_=.true.
  end subroutine

  function code_predict(self,x) result(labels)
    class(output_code_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: labels(:)
    real(dp), allocatable :: values(:,:),distance(:)
    integer :: j,row
    call fitted_input(self%fitted_,self%n_features_in_,x)
    allocate(values(size(x,1),size(self%estimators_)),labels(size(x,1)),distance(self%n_classes_))
    do j=1,size(self%estimators_)
      values(:,j)=binary_response(self%estimators_(j),x,'binary')
    end do
    call require(all(ieee_is_finite(values)),'output-code responses must be finite')
    call require(maxval(abs(values))<=sqrt(huge(1._dp))/(2*sqrt(real(size(values,2),dp))), &
      'output-code response magnitude exceeds working range')
    do row=1,size(x,1)
      do j=1,self%n_classes_
        distance(j)=sum((values(row,:)-self%code_book_(j,:))**2)
      end do
      labels(row)=self%classes_(minloc(distance,dim=1))
    end do
  end function

  real(dp) function code_score(self,x,y,sample_weight) result(value)
    class(output_code_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    integer, allocatable :: predicted(:)
    predicted=self%predict(x); call require(size(y)==size(predicted),'multiclass score target length mismatch')
    value=subset_accuracy(y==predicted,sample_weight)
  end function
end module
