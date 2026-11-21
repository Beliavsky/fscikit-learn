! Independent-output wrappers adapted from sklearn 1.6.1 multioutput.py.
! BSD attribution: THIRD_PARTY_NOTICES.md and LICENSES/scikit-learn-COPYING.
module sklearn_multioutput
  use sklearn_common, only: dp,require,check_x,check_xy,unique_labels
  use sklearn_linear_model, only: linear_regression,logistic_regression
  use sklearn_naive_bayes, only: gaussian_nb,multinomial_nb
  use sklearn_sgd, only: sgd_regressor,sgd_classifier
  use sklearn_neighbors, only: kneighbors_regressor,kneighbors_classifier
  use sklearn_tree, only: decision_tree_regressor,decision_tree_classifier
  use sklearn_model_selection, only: kfold,index_split
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_label_binarization, only: label_set
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_value,ieee_quiet_nan,ieee_negative_inf
  implicit none
  private
  public :: regressor_chain,classifier_chain
  public :: multi_output_regressor,multi_output_classifier,multioutput_probability,label_set
  type :: estimator_slot
    class(*), allocatable :: estimator_
  end type
  type :: multioutput_probability
    integer, allocatable :: classes(:)
    real(dp), allocatable :: values(:,:)
  end type
  type :: classifier_chain
    integer, allocatable :: order(:),order_(:)
    integer :: cv=0,random_state=0,n_features_in_=0,n_outputs_=0
    logical :: random_order=.false.
    character(20) :: chain_method='predict',chain_method_=''
    character(20), allocatable :: chain_methods(:)
    type(estimator_slot), allocatable :: estimators_(:)
    type(label_set), allocatable :: classes_(:)
    logical, private :: fitted_=.false.
  contains
    procedure :: fit=>classification_chain_fit
    procedure :: predict=>classification_chain_predict
    procedure :: predict_proba=>classification_chain_proba
    procedure :: decision_function=>classification_chain_decision
    procedure :: score=>classification_chain_score
  end type
  type :: regressor_chain
    integer, allocatable :: order(:),order_(:)
    integer :: cv=0,random_state=0,n_features_in_=0,n_outputs_=0
    logical :: random_order=.false.
    type(estimator_slot), allocatable :: estimators_(:)
    logical, private :: fitted_=.false.
  contains
    procedure :: fit=>chain_fit
    procedure :: predict=>chain_predict
    procedure :: score=>chain_score
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
    class is(decision_tree_regressor)
      call require(.not.streaming,'multioutput regressor base has no native partial_fit'); return
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
    class is(decision_tree_regressor)
      call model%fit(x,y,weights)
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
      class is(decision_tree_regressor)
        y(:,j)=model%predict(x)
      end select
    end do
  end function

  real(dp) function reg_score(self,x,y,sample_weight) result(value)
    class(multi_output_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: predicted(:,:)
    predicted=self%predict(x)
    value=multioutput_r2(y,predicted,sample_weight)
  end function

  real(dp) function multioutput_r2(y,predicted,sample_weight) result(value)
    real(dp), intent(in) :: y(:,:),predicted(:,:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: w(:)
    real(dp) :: mass,mean,numerator,denominator
    integer :: j
    call require(all(shape(y)==shape(predicted)),'multioutput score target shape mismatch')
    call check_x(y)
    call require(all(ieee_is_finite(predicted)),'multioutput regression score predictions must be finite')
    allocate(w(size(y,1))); w=1
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
    class is(decision_tree_classifier)
      call require(.not.streaming,'multioutput classifier base has no native partial_fit'); return
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
    class is(decision_tree_classifier)
      call model%fit(x,y,weights)
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
    class is(decision_tree_classifier)
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
      class is(decision_tree_classifier)
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
      class is(decision_tree_classifier)
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
  function regression_response(model,x) result(y)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:)
    select type(model)
    class is(linear_regression)
      y=model%predict(x)
    type is(sgd_regressor)
      y=model%predict(x)
    type is(kneighbors_regressor)
      y=model%predict(x)
    class is(decision_tree_regressor)
      y=model%predict(x)
    class default
      call require(.false.,'unsupported regression chain base')
    end select
  end function

  subroutine chain_fit(self,x,y,estimator,sample_weight,splits)
    class(regressor_chain), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    class(*), intent(in) :: estimator
    real(dp), intent(in), optional :: sample_weight(:)
    type(index_split), intent(in), optional :: splits(:)
    type(index_split), allocatable :: folds(:)
    type(kfold) :: splitter
    type(legacy_random_state) :: rng
    class(*), allocatable :: fold_model
    real(dp), allocatable :: augmented(:,:)
    integer, allocatable :: seen(:)
    integer :: n,p,t,j,k,i,pick,tmp,width
    logical :: cross_validated
    self%fitted_=.false.
    call targets(x,size(y,1),size(y,2)); call check_x(y)
    call reg_supported(estimator,present(sample_weight),.false.)
    n=size(x,1); p=size(x,2); t=size(y,2)
    call require(self%random_state>=0,'regression chain random_state must be nonnegative')
    call require(self%cv==0.or.self%cv>=2,'regression chain cv must be zero or at least two')
    call require(.not.(present(splits).and.self%cv/=0),'regression chain choose cv or explicit splits')
    call require(.not.(allocated(self%order).and.self%random_order),'regression chain choose explicit or random order')
    if(allocated(self%order_)) deallocate(self%order_)
    self%order_=[(j,j=1,t)]
    if(allocated(self%order)) then
      call require(size(self%order)==t,'regression chain order length mismatch')
      call require(all(self%order>=1).and.all(self%order<=t),'regression chain order index out of range')
      do j=1,t
        call require(count(self%order==j)==1,'regression chain order must be a permutation')
      end do
      self%order_=self%order
    else if(self%random_order) then
      call rng%seed(self%random_state)
      do j=t,2,-1
        pick=rng%integer_index(j); tmp=self%order_(j)
        self%order_(j)=self%order_(pick); self%order_(pick)=tmp
      end do
    end if
    cross_validated=self%cv/=0.or.present(splits)
    if(present(splits).and.t>1) then
      call require(size(splits)>0,'regression chain splits must not be empty')
      folds=splits
    else if(self%cv/=0.and.t>1) then
      splitter%n_splits=self%cv; folds=splitter%split(n)
    end if
    if(cross_validated.and.t>1) then
      allocate(seen(n)); seen=0
      do k=1,size(folds)
        call require(allocated(folds(k)%train).and.allocated(folds(k)%test),'regression chain split indices missing')
        call require(size(folds(k)%train)>0.and.size(folds(k)%test)>0,'regression chain split must be nonempty')
        call require(all(folds(k)%train>=1).and.all(folds(k)%train<=n),'regression chain train index out of range')
        call require(all(folds(k)%test>=1).and.all(folds(k)%test<=n),'regression chain test index out of range')
        do i=1,size(folds(k)%test)
          j=folds(k)%test(i)
          seen(j)=seen(j)+1
        end do
      end do
      call require(all(seen==1),'regression chain test folds must partition all rows')
    end if
    if(allocated(self%estimators_)) deallocate(self%estimators_)
    allocate(self%estimators_(t),augmented(n,p+t)); augmented=0; augmented(:,:p)=x
    if(.not.cross_validated) augmented(:,p+1:)=y(:,self%order_)
    do j=1,t
      width=p+j-1
      allocate(self%estimators_(j)%estimator_,source=estimator)
      call reg_train(self%estimators_(j)%estimator_,augmented(:,:width),y(:,self%order_(j)),sample_weight,.false.)
      if(cross_validated.and.j<t) then
        do k=1,size(folds)
          allocate(fold_model,source=estimator)
          ! Source cross_val_predict does not receive the full-fit sample weights.
          call reg_train(fold_model,augmented(folds(k)%train,:width),y(folds(k)%train,self%order_(j)),streaming=.false.)
          augmented(folds(k)%test,p+j)=regression_response(fold_model,augmented(folds(k)%test,:width))
          deallocate(fold_model)
        end do
      end if
    end do
    self%n_features_in_=p; self%n_outputs_=t; self%fitted_=.true.
  end subroutine

  function chain_predict(self,x) result(y)
    class(regressor_chain), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:,:),augmented(:,:)
    integer :: p,j
    call require(self%fitted_,'regression chain is not fitted')
    call check_x(x); call require(size(x,2)==self%n_features_in_,'regression chain feature count mismatch')
    p=self%n_features_in_; allocate(augmented(size(x,1),p+self%n_outputs_))
    augmented=0; augmented(:,:p)=x; allocate(y(size(x,1),self%n_outputs_))
    do j=1,self%n_outputs_
      augmented(:,p+j)=regression_response(self%estimators_(j)%estimator_,augmented(:,:p+j-1))
      y(:,self%order_(j))=augmented(:,p+j)
    end do
  end function

  real(dp) function chain_score(self,x,y,sample_weight) result(value)
    class(regressor_chain), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: prediction(:,:)
    prediction=self%predict(x); value=multioutput_r2(y,prediction,sample_weight)
  end function

  logical function chain_method_available(model,method) result(available)
    class(*), intent(in) :: model
    character(*), intent(in) :: method
    available=.false.
    select case(method)
    case('predict')
      available=.true.
    case('predict_proba','predict_log_proba')
      available=.true.
      select type(model)
      type is(sgd_classifier)
        available=model%loss=='log_loss'.or.model%loss=='modified_huber'
      type is(kneighbors_classifier)
        available=method=='predict_proba'
      end select
    case('decision_function')
      select type(model)
      type is(logistic_regression)
        available=.true.
      type is(sgd_classifier)
        available=.true.
      end select
    end select
  end function

  function classification_response(model,x,method) result(values)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    character(*), intent(in) :: method
    real(dp), allocatable :: values(:,:),prob(:,:)
    integer :: i,j
    call require(chain_method_available(model,method),'classification chain base lacks requested response method')
    if(method=='decision_function') then
      select type(model)
      type is(logistic_regression)
        values=model%decision_function(x)
      type is(sgd_classifier)
        values=model%decision_function(x)
      end select
      return
    end if
    if(method=='predict_log_proba') then
      select type(model)
      type is(gaussian_nb)
        values=model%predict_log_proba(x); return
      class is(multinomial_nb)
        values=model%predict_log_proba(x); return
      end select
    end if
    if(method=='predict') allocate(values(size(x,1),1))
    select type(model)
    type is(logistic_regression)
      if(method=='predict') then
        values(:,1)=real(model%predict(x),dp)
      else
        values=model%predict_proba(x)
      end if
    type is(gaussian_nb)
      if(method=='predict') then
        values(:,1)=real(model%predict(x),dp)
      else
        values=model%predict_proba(x)
      end if
    type is(sgd_classifier)
      if(method=='predict') then
        values(:,1)=real(model%predict(x),dp)
      else
        values=model%predict_proba(x)
      end if
    type is(kneighbors_classifier)
      if(method=='predict') then
        values(:,1)=real(model%predict(x),dp)
      else
        values=model%predict_proba(x)
      end if
    class is(decision_tree_classifier)
      if(method=='predict') then
        values(:,1)=real(model%predict(x),dp)
      else
        values=model%predict_proba(x)
      end if
    class is(multinomial_nb)
      if(method=='predict') then
        values(:,1)=real(model%predict(x),dp)
      else
        values=model%predict_proba(x)
      end if
    end select
    if(method=='predict_log_proba') then
      do j=1,size(values,2)
        do i=1,size(values,1)
          if(values(i,j)==0) then
            values(i,j)=ieee_value(0._dp,ieee_negative_inf)
          else
            values(i,j)=log(values(i,j))
          end if
        end do
      end do
    end if
  end function

  function classification_vector(model,x,method) result(values)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    character(*), intent(in) :: method
    real(dp), allocatable :: values(:),matrix(:,:)
    integer, allocatable :: labels(:)
    matrix=classification_response(model,x,method)
    if(method=='predict_proba'.or.method=='predict_log_proba') then
      labels=clf_classes(model)
      call require(size(labels)==2,'classification chain probability response requires two classes')
      values=matrix(:,2)
    else
      call require(size(matrix,2)==1,'classification chain requires a scalar response per output')
      values=matrix(:,1)
    end if
  end function

  function classification_cv_vector(model,x,method,labels) result(values)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    character(*), intent(in) :: method
    integer, intent(in) :: labels(:)
    real(dp), allocatable :: values(:),matrix(:,:)
    integer, allocatable :: fitted_labels(:)
    integer :: column,j
    matrix=classification_response(model,x,method); fitted_labels=clf_classes(model)
    if(method=='predict') then
      values=matrix(:,1)
    else if(method=='predict_proba'.or.method=='predict_log_proba') then
      call require(size(labels)>=2,'classification chain CV probabilities require at least two global classes')
      ! Source cross_val_predict returns global class columns, then takes column 2.
      column=0
      do j=1,size(fitted_labels)
        if(fitted_labels(j)==labels(2)) column=j
      end do
      if(column>0) then
        values=matrix(:,column)
      else
        allocate(values(size(x,1))); values=0
        if(method=='predict_log_proba') values=-huge(1._dp)
      end if
    else
      call require(size(fitted_labels)==size(labels),'classification chain CV decision classes are incomplete')
      if(size(labels)==2) then
        values=matrix(:,1)
      else
        values=matrix(:,2)
      end if
    end if
  end function

  function classification_folds(y,n_splits) result(folds)
    integer, intent(in) :: y(:),n_splits
    type(index_split), allocatable :: folds(:)
    integer, allocatable :: classes(:),ordered(:),counts(:),assignment(:),ids(:),members(:),allocation(:)
    integer :: i,j,k,c,offset,position,nclasses
    classes=unique_labels(y); allocate(ordered(size(classes)),counts(size(classes)),assignment(size(y)))
    nclasses=0
    do i=1,size(y)
      if(any(ordered(:nclasses)==y(i))) cycle
      nclasses=nclasses+1; ordered(nclasses)=y(i)
    end do
    do c=1,nclasses
      counts(c)=count(y==ordered(c))
    end do
    call require(n_splits<=maxval(counts),'classification chain cv exceeds every class count')
    ids=[(i,i=1,size(y))]; allocate(folds(n_splits),allocation(n_splits)); offset=0
    do c=1,nclasses
      allocation=0
      do i=1,counts(c)
        k=1+mod(offset+i-1,n_splits); allocation(k)=allocation(k)+1
      end do
      members=pack(ids,y==ordered(c)); position=1
      do k=1,n_splits
        assignment(members(position:position+allocation(k)-1))=k
        position=position+allocation(k)
      end do
      offset=offset+counts(c)
    end do
    do k=1,n_splits
      folds(k)%train=pack(ids,assignment/=k); folds(k)%test=pack(ids,assignment==k)
    end do
  end function

  subroutine classification_check_folds(folds,n)
    type(index_split), intent(in) :: folds(:)
    integer, intent(in) :: n
    integer :: seen(n),k,i
    call require(size(folds)>0,'classification chain splits must not be empty'); seen=0
    do k=1,size(folds)
      call require(allocated(folds(k)%train).and.allocated(folds(k)%test),'classification chain split indices missing')
      call require(size(folds(k)%train)>0.and.size(folds(k)%test)>0,'classification chain split must be nonempty')
      call require(all(folds(k)%train>=1).and.all(folds(k)%train<=n),'classification chain train index out of range')
      call require(all(folds(k)%test>=1).and.all(folds(k)%test<=n),'classification chain test index out of range')
      do i=1,size(folds(k)%test)
        seen(folds(k)%test(i))=seen(folds(k)%test(i))+1
      end do
    end do
    call require(all(seen==1),'classification chain test folds must partition all rows')
  end subroutine

  subroutine classification_chain_fit(self,x,y,estimator,sample_weight,splits)
    class(classifier_chain), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:,:)
    class(*), intent(in) :: estimator
    real(dp), intent(in), optional :: sample_weight(:)
    type(index_split), intent(in), optional :: splits(:)
    type(index_split), allocatable :: folds(:)
    type(legacy_random_state) :: rng
    class(*), allocatable :: fold_model
    real(dp), allocatable :: augmented(:,:)
    integer, allocatable :: labels(:)
    integer :: n,p,t,j,k,pick,tmp,width
    logical :: cross_validated
    self%fitted_=.false.; call targets(x,size(y,1),size(y,2))
    call clf_supported(estimator,present(sample_weight),.false.)
    n=size(x,1); p=size(x,2); t=size(y,2)
    call require(self%random_state>=0,'classification chain random_state must be nonnegative')
    call require(self%cv==0.or.self%cv>=2,'classification chain cv must be zero or at least two')
    call require(.not.(present(splits).and.self%cv/=0),'classification chain choose cv or explicit splits')
    call require(.not.(allocated(self%order).and.self%random_order),'classification chain choose explicit or random order')
    if(allocated(self%order_)) deallocate(self%order_)
    self%order_=[(j,j=1,t)]
    if(allocated(self%order)) then
      call require(size(self%order)==t,'classification chain order length mismatch')
      call require(all(self%order>=1).and.all(self%order<=t),'classification chain order index out of range')
      do j=1,t
        call require(count(self%order==j)==1,'classification chain order must be a permutation')
      end do
      self%order_=self%order
    else if(self%random_order) then
      call rng%seed(self%random_state)
      do j=t,2,-1
        pick=rng%integer_index(j); tmp=self%order_(j)
        self%order_(j)=self%order_(pick); self%order_(pick)=tmp
      end do
    end if
    self%chain_method_=''
    if(allocated(self%chain_methods)) then
      call require(size(self%chain_methods)>0,'classification chain response preference list must not be empty')
      do j=1,size(self%chain_methods)
        if(.not.chain_method_available(estimator,trim(self%chain_methods(j)))) cycle
        self%chain_method_=self%chain_methods(j); exit
      end do
    else
      if(chain_method_available(estimator,trim(self%chain_method))) self%chain_method_=self%chain_method
    end if
    call require(len_trim(self%chain_method_)>0,'classification chain base lacks requested response method')
    cross_validated=self%cv/=0.or.present(splits)
    if(present(splits).and.t>1) then
      call classification_check_folds(splits,n); folds=splits
    end if
    if(allocated(self%estimators_)) deallocate(self%estimators_)
    if(allocated(self%classes_)) deallocate(self%classes_)
    allocate(self%estimators_(t),self%classes_(t),augmented(n,p+t)); augmented=0; augmented(:,:p)=x
    if(.not.cross_validated) augmented(:,p+1:)=real(y(:,self%order_),dp)
    do j=1,t
      width=p+j-1
      allocate(self%estimators_(j)%estimator_,source=estimator)
      call clf_train(self%estimators_(j)%estimator_,augmented(:,:width),y(:,self%order_(j)),sample_weight,.false.)
      self%classes_(j)%values=clf_classes(self%estimators_(j)%estimator_)
      if(cross_validated.and.j<t) then
        if(self%cv/=0) folds=classification_folds(y(:,self%order_(j)),self%cv)
        labels=unique_labels(y(:,self%order_(j)))
        do k=1,size(folds)
          allocate(fold_model,source=estimator)
          call clf_train(fold_model,augmented(folds(k)%train,:width),y(folds(k)%train,self%order_(j)),streaming=.false.)
          augmented(folds(k)%test,p+j)=classification_cv_vector(fold_model, &
            augmented(folds(k)%test,:width),trim(self%chain_method_),labels)
          deallocate(fold_model)
        end do
      end if
    end do
    self%n_features_in_=p; self%n_outputs_=t; self%fitted_=.true.
  end subroutine

  function classification_chain_output(self,x,method) result(y)
    class(classifier_chain), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    character(*), intent(in) :: method
    real(dp), allocatable :: y(:,:),augmented(:,:)
    integer :: p,j
    call require(self%fitted_,'classification chain is not fitted')
    call check_x(x); call require(size(x,2)==self%n_features_in_,'classification chain feature count mismatch')
    p=self%n_features_in_; allocate(augmented(size(x,1),p+self%n_outputs_))
    augmented=0; augmented(:,:p)=x; allocate(y(size(x,1),self%n_outputs_))
    do j=1,self%n_outputs_
      augmented(:,p+j)=classification_vector(self%estimators_(j)%estimator_, &
        augmented(:,:p+j-1),trim(self%chain_method_))
      y(:,self%order_(j))=classification_vector(self%estimators_(j)%estimator_,augmented(:,:p+j-1),method)
    end do
  end function

  function classification_chain_predict(self,x) result(y)
    class(classifier_chain), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: y(:,:)
    y=nint(classification_chain_output(self,x,'predict'))
  end function

  function classification_chain_proba(self,x) result(y)
    class(classifier_chain), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:,:)
    y=classification_chain_output(self,x,'predict_proba')
  end function

  function classification_chain_decision(self,x) result(y)
    class(classifier_chain), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:,:)
    y=classification_chain_output(self,x,'decision_function')
  end function

  real(dp) function classification_chain_score(self,x,y,sample_weight) result(value)
    class(classifier_chain), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:,:)
    real(dp), intent(in), optional :: sample_weight(:)
    integer, allocatable :: predicted(:,:)
    real(dp), allocatable :: w(:)
    predicted=self%predict(x)
    call require(all(shape(y)==shape(predicted)),'classification chain score target shape mismatch')
    if(size(y,2)>1) then
      call require(size(unique_labels(reshape(y,[size(y)])))<=2.and. &
        size(unique_labels(reshape(predicted,[size(predicted)])))<=2, &
        'classification chain score requires multilabel indicator targets')
    end if
    allocate(w(size(y,1))); w=1
    if(present(sample_weight)) then
      call require(size(sample_weight)==size(w),'classification chain score weight length mismatch')
      call require(all(ieee_is_finite(sample_weight)),'classification chain score weights must be finite')
      call require(maxval(abs(sample_weight))<=huge(1._dp)/(16*real(size(w),dp)), &
        'classification chain score weight mass exceeds working range')
      w=sample_weight
    end if
    call require(sum(w)/=0,'classification chain score weights require nonzero total')
    value=sum(w*merge(1._dp,0._dp,all(y==predicted,dim=2)))/sum(w)
  end function

end module
