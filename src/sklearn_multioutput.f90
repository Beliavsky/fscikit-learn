! Independent-output wrappers adapted from sklearn 1.6.1 multioutput.py.
! BSD attribution: THIRD_PARTY_NOTICES.md and LICENSES/scikit-learn-COPYING.
module sklearn_multioutput
  use sklearn_common, only: dp,require,check_x,check_xy
  use sklearn_linear_model, only: linear_regression,logistic_regression
  use sklearn_naive_bayes, only: gaussian_nb,multinomial_nb
  use sklearn_sgd, only: sgd_regressor,sgd_classifier
  use sklearn_neighbors, only: kneighbors_regressor,kneighbors_classifier
  use sklearn_tree, only: decision_tree_regressor,decision_tree_classifier
  use sklearn_label_binarization, only: label_set
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_value,ieee_quiet_nan
  implicit none
  private
  public :: multi_output_regressor,multi_output_classifier,multioutput_probability,label_set
  type :: estimator_slot
    class(*), allocatable :: estimator_
  end type
  type :: multioutput_probability
    integer, allocatable :: classes(:)
    real(dp), allocatable :: values(:,:)
  end type
  type :: multi_output_regressor
    type(estimator_slot), allocatable :: estimators_(:)
    integer :: n_features_in_=0,n_outputs_=0
    logical, private :: fitted_=.false.
  contains
    procedure :: fit=>reg_fit
    procedure :: partial_fit=>reg_partial
    procedure :: predict=>reg_predict
    procedure :: score=>reg_score
  end type
  type :: multi_output_classifier
    type(estimator_slot), allocatable :: estimators_(:)
    type(label_set), allocatable :: classes_(:)
    integer :: n_features_in_=0,n_outputs_=0
    logical, private :: fitted_=.false.
  contains
    procedure :: fit=>clf_fit
    procedure :: partial_fit=>clf_partial
    procedure :: predict=>clf_predict
    procedure :: predict_proba=>clf_proba
    procedure :: score=>clf_score
  end type
contains
  subroutine targets(x,n,outputs)
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: n,outputs
    call check_xy(x,n)
    call require(outputs>0,'multioutput targets must have at least one column')
  end subroutine

  subroutine reg_supported(model,weighted,streaming)
    class(*), intent(in) :: model
    logical, intent(in) :: weighted,streaming
    select type(model)
    type is(sgd_regressor)
      return
    class is(linear_regression)
    type is(kneighbors_regressor)
    type is(decision_tree_regressor)
    class default
      call require(.false.,'unsupported multioutput regressor base')
    end select
    call require(.not.streaming,'multioutput regressor base has no native partial_fit')
    call require(.not.weighted,'multioutput regressor base has no native sample-weight fit')
  end subroutine

  subroutine reg_train(model,x,y,weights,streaming)
    class(*), intent(inout) :: model
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), intent(in), optional :: weights(:)
    logical, intent(in) :: streaming
    select type(model)
    type is(sgd_regressor)
      if(streaming) then
        call model%partial_fit(x,y,weights)
      else
        call model%fit(x,y,weights)
      end if
    class is(linear_regression)
      call model%fit(x,y)
    type is(kneighbors_regressor)
      call model%fit(x,y)
    type is(decision_tree_regressor)
      call model%fit(x,y)
    end select
  end subroutine

  function clone_sgdr(source) result(model)
    type(sgd_regressor), intent(in) :: source
    type(sgd_regressor) :: model
    model%loss=source%loss; model%penalty=source%penalty; model%learning_rate=source%learning_rate
    model%alpha=source%alpha; model%eta0=source%eta0; model%power_t=source%power_t
    model%epsilon=source%epsilon; model%tol=source%tol; model%max_iter=source%max_iter
    model%n_iter_no_change=source%n_iter_no_change; model%average=source%average
    model%random_state=source%random_state; model%fit_intercept=source%fit_intercept; model%shuffle=source%shuffle
  end function

  function clone_sgdc(source) result(model)
    type(sgd_classifier), intent(in) :: source
    type(sgd_classifier) :: model
    model%loss=source%loss; model%penalty=source%penalty; model%learning_rate=source%learning_rate
    model%alpha=source%alpha; model%eta0=source%eta0; model%power_t=source%power_t
    model%tol=source%tol; model%max_iter=source%max_iter; model%n_iter_no_change=source%n_iter_no_change
    model%average=source%average; model%random_state=source%random_state
    model%fit_intercept=source%fit_intercept; model%shuffle=source%shuffle
    model%class_weight_balanced=source%class_weight_balanced
  end function

  subroutine reg_fit(self,x,y,estimator,sample_weight)
    class(multi_output_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    class(*), intent(in) :: estimator
    real(dp), intent(in), optional :: sample_weight(:)
    integer :: j
    self%fitted_=.false.; call targets(x,size(y,1),size(y,2)); call check_x(y)
    call reg_supported(estimator,present(sample_weight),.false.)
    if(allocated(self%estimators_)) deallocate(self%estimators_)
    allocate(self%estimators_(size(y,2)))
    do j=1,size(y,2)
      allocate(self%estimators_(j)%estimator_,source=estimator)
      call reg_train(self%estimators_(j)%estimator_,x,y(:,j),sample_weight,.false.)
    end do
    self%n_features_in_=size(x,2); self%n_outputs_=size(y,2); self%fitted_=.true.
  end subroutine

  subroutine reg_partial(self,x,y,estimator,sample_weight)
    class(multi_output_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    class(*), intent(in), optional :: estimator
    real(dp), intent(in), optional :: sample_weight(:)
    type(sgd_regressor) :: fresh
    integer :: j
    logical :: first
    call targets(x,size(y,1),size(y,2)); call check_x(y); first=.not.self%fitted_
    if(first) then
      call require(present(estimator),'first multioutput partial_fit requires an estimator')
      call reg_supported(estimator,present(sample_weight),.true.)
      if(allocated(self%estimators_)) deallocate(self%estimators_)
      allocate(self%estimators_(size(y,2)))
      select type(estimator)
      type is(sgd_regressor)
        fresh=clone_sgdr(estimator)
        do j=1,size(y,2)
          allocate(self%estimators_(j)%estimator_,source=fresh)
        end do
      end select
    else
      call require(size(x,2)==self%n_features_in_,'multioutput feature count mismatch')
      call require(size(y,2)<=self%n_outputs_,'multioutput partial_fit cannot add targets')
      do j=1,size(y,2)
        call reg_supported(self%estimators_(j)%estimator_,present(sample_weight),.true.)
      end do
    end if
    if(size(y,2)<size(self%estimators_)) self%estimators_=self%estimators_(:size(y,2))
    do j=1,size(y,2)
      call reg_train(self%estimators_(j)%estimator_,x,y(:,j),sample_weight,.true.)
    end do
    self%n_features_in_=size(x,2); self%n_outputs_=size(y,2); self%fitted_=.true.
  end subroutine

  function reg_predict(self,x) result(y)
    class(multi_output_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:,:)
    integer :: j
    call require(self%fitted_,'multioutput regressor is not fitted')
    call check_x(x); call require(size(x,2)==self%n_features_in_,'multioutput feature count mismatch')
    allocate(y(size(x,1),self%n_outputs_))
    do j=1,self%n_outputs_
      select type(model=>self%estimators_(j)%estimator_)
      class is(linear_regression)
        y(:,j)=model%predict(x)
      type is(sgd_regressor)
        y(:,j)=model%predict(x)
      type is(kneighbors_regressor)
        y(:,j)=model%predict(x)
      type is(decision_tree_regressor)
        y(:,j)=model%predict(x)
      end select
    end do
  end function

  real(dp) function reg_score(self,x,y,sample_weight) result(value)
    class(multi_output_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: predicted(:,:),w(:)
    real(dp) :: mass,mean,numerator,denominator
    integer :: j
    predicted=self%predict(x)
    call require(all(shape(y)==shape(predicted)),'multioutput score target shape mismatch')
    call check_x(y); allocate(w(size(y,1))); w=1
    if(present(sample_weight)) then
      call require(size(sample_weight)==size(w),'multioutput score weight length mismatch')
      call require(all(ieee_is_finite(sample_weight)),'multioutput score weights must be finite')
      call require(all(sample_weight>=0).and.any(sample_weight>0),'multioutput score weights need positive mass')
      call require(maxval(sample_weight)<=huge(1._dp)/(16*real(size(w),dp)), &
        'multioutput score weight mass exceeds working range')
      w=sample_weight
    end if
    if(size(y,1)<2) then
      value=ieee_value(0._dp,ieee_quiet_nan); return
    end if
    mass=sum(w); value=0
    do j=1,size(y,2)
      mean=sum(w*y(:,j))/mass
      numerator=sum(w*(y(:,j)-predicted(:,j))**2)
      denominator=sum(w*(y(:,j)-mean)**2)
      ! Constant targets have exactly zero variance; reduction roundoff in
      ! weighted means must not defeat source force_finite semantics.
      if(maxval(y(:,j),mask=w>0)==minval(y(:,j),mask=w>0)) denominator=0
      if(denominator==0) then
        value=value+merge(1._dp,0._dp,numerator==0)
      else
        value=value+1-numerator/denominator
      end if
    end do
    value=value/size(y,2)
  end function

  subroutine clf_supported(model,weighted,streaming)
    class(*), intent(in) :: model
    logical, intent(in) :: weighted,streaming
    select type(model)
    type is(gaussian_nb)
      return
    type is(sgd_classifier)
      return
    type is(logistic_regression)
      call require(.not.streaming,'multioutput classifier base has no native partial_fit'); return
    type is(kneighbors_classifier)
    type is(decision_tree_classifier)
    class is(multinomial_nb)
    class default
      call require(.false.,'unsupported multioutput classifier base')
    end select
    call require(.not.streaming,'multioutput classifier base has no native partial_fit')
    call require(.not.weighted,'multioutput classifier base has no native sample-weight fit')
  end subroutine

  subroutine clf_train(model,x,y,weights,streaming,classes)
    class(*), intent(inout) :: model
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(in), optional :: weights(:)
    logical, intent(in) :: streaming
    integer, intent(in), optional :: classes(:)
    select type(model)
    type is(gaussian_nb)
      if(streaming) then
        call model%partial_fit(x,y,classes,weights)
      else
        call model%fit(x,y,weights)
      end if
    type is(sgd_classifier)
      if(streaming) then
        call model%partial_fit(x,y,classes,weights)
      else
        call model%fit(x,y,weights)
      end if
    type is(logistic_regression)
      call model%fit(x,y,weights)
    type is(kneighbors_classifier)
      call model%fit(x,y)
    type is(decision_tree_classifier)
      call model%fit(x,y)
    class is(multinomial_nb)
      call model%fit(x,y)
    end select
  end subroutine

  function clf_classes(model) result(labels)
    class(*), intent(in) :: model
    integer, allocatable :: labels(:)
    select type(model)
    type is(gaussian_nb)
      labels=model%classes_
    type is(sgd_classifier)
      labels=model%classes_
    type is(logistic_regression)
      labels=model%classes_
    type is(kneighbors_classifier)
      labels=model%classes_
    type is(decision_tree_classifier)
      labels=model%classes_
    class is(multinomial_nb)
      labels=model%classes_
    end select
  end function

  subroutine clf_fit(self,x,y,estimator,sample_weight)
    class(multi_output_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:,:)
    class(*), intent(in) :: estimator
    real(dp), intent(in), optional :: sample_weight(:)
    integer :: j
    self%fitted_=.false.; call targets(x,size(y,1),size(y,2))
    call clf_supported(estimator,present(sample_weight),.false.)
    if(allocated(self%estimators_)) deallocate(self%estimators_)
    if(allocated(self%classes_)) deallocate(self%classes_)
    allocate(self%estimators_(size(y,2)),self%classes_(size(y,2)))
    do j=1,size(y,2)
      allocate(self%estimators_(j)%estimator_,source=estimator)
      call clf_train(self%estimators_(j)%estimator_,x,y(:,j),sample_weight,.false.)
      self%classes_(j)%values=clf_classes(self%estimators_(j)%estimator_)
    end do
    self%n_features_in_=size(x,2); self%n_outputs_=size(y,2); self%fitted_=.true.
  end subroutine

  subroutine clf_partial(self,x,y,estimator,classes,sample_weight)
    class(multi_output_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:,:)
    class(*), intent(in), optional :: estimator
    type(label_set), intent(in), optional :: classes(:)
    real(dp), intent(in), optional :: sample_weight(:)
    type(gaussian_nb) :: nb
    type(sgd_classifier) :: sgd
    logical :: first
    integer :: j
    call targets(x,size(y,1),size(y,2)); first=.not.self%fitted_
    if(present(classes)) call require(size(classes)>=size(y,2),'multioutput classes list is too short')
    if(first) then
      call require(present(estimator),'first multioutput partial_fit requires an estimator')
      call clf_supported(estimator,present(sample_weight),.true.)
      if(allocated(self%estimators_)) deallocate(self%estimators_)
      if(allocated(self%classes_)) deallocate(self%classes_)
      allocate(self%estimators_(size(y,2)))
      do j=1,size(y,2)
        select type(estimator)
        type is(gaussian_nb)
          nb=gaussian_nb()
          nb%var_smoothing=estimator%var_smoothing
          if(allocated(estimator%priors)) nb%priors=estimator%priors
          allocate(self%estimators_(j)%estimator_,source=nb)
        type is(sgd_classifier)
          sgd=clone_sgdc(estimator); allocate(self%estimators_(j)%estimator_,source=sgd)
        end select
      end do
    else
      call require(size(x,2)==self%n_features_in_,'multioutput feature count mismatch')
      call require(size(y,2)<=self%n_outputs_,'multioutput partial_fit cannot add targets')
      do j=1,size(y,2)
        call clf_supported(self%estimators_(j)%estimator_,present(sample_weight),.true.)
      end do
    end if
    if(size(y,2)<size(self%estimators_)) self%estimators_=self%estimators_(:size(y,2))
    do j=1,size(y,2)
      if(present(classes)) then
        call clf_train(self%estimators_(j)%estimator_,x,y(:,j),sample_weight,.true.,classes(j)%values)
      else
        call clf_train(self%estimators_(j)%estimator_,x,y(:,j),sample_weight,.true.)
      end if
    end do
    ! Source 1.6.1 sets top-level classes_ only in fit, not in partial_fit.
    self%n_features_in_=size(x,2); self%n_outputs_=size(y,2); self%fitted_=.true.
  end subroutine

  function clf_predict(self,x) result(y)
    class(multi_output_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: y(:,:)
    integer :: j
    call require(self%fitted_,'multioutput classifier is not fitted')
    call check_x(x); call require(size(x,2)==self%n_features_in_,'multioutput feature count mismatch')
    allocate(y(size(x,1),self%n_outputs_))
    do j=1,self%n_outputs_
      select type(model=>self%estimators_(j)%estimator_)
      type is(gaussian_nb)
        y(:,j)=model%predict(x)
      type is(sgd_classifier)
        y(:,j)=model%predict(x)
      type is(logistic_regression)
        y(:,j)=model%predict(x)
      type is(kneighbors_classifier)
        y(:,j)=model%predict(x)
      type is(decision_tree_classifier)
        y(:,j)=model%predict(x)
      class is(multinomial_nb)
        y(:,j)=model%predict(x)
      end select
    end do
  end function

  function clf_proba(self,x) result(prob)
    class(multi_output_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    type(multioutput_probability), allocatable :: prob(:)
    integer :: j
    call require(self%fitted_,'multioutput classifier is not fitted')
    call check_x(x); call require(size(x,2)==self%n_features_in_,'multioutput feature count mismatch')
    allocate(prob(self%n_outputs_))
    do j=1,self%n_outputs_
      prob(j)%classes=clf_classes(self%estimators_(j)%estimator_)
      select type(model=>self%estimators_(j)%estimator_)
      type is(gaussian_nb)
        prob(j)%values=model%predict_proba(x)
      type is(sgd_classifier)
        prob(j)%values=model%predict_proba(x)
      type is(logistic_regression)
        prob(j)%values=model%predict_proba(x)
      type is(kneighbors_classifier)
        prob(j)%values=model%predict_proba(x)
      type is(decision_tree_classifier)
        prob(j)%values=model%predict_proba(x)
      class is(multinomial_nb)
        prob(j)%values=model%predict_proba(x)
      end select
    end do
  end function

  real(dp) function clf_score(self,x,y) result(value)
    class(multi_output_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:,:)
    integer, allocatable :: predicted(:,:)
    predicted=self%predict(x)
    call require(all(shape(y)==shape(predicted)),'multioutput score target shape mismatch')
    value=real(count(all(y==predicted,dim=2)),dp)/size(y,1)
  end function
end module
