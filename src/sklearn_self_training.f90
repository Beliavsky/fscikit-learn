! Self-training loop adapted from sklearn 1.6.1 semi_supervised/_self_training.py.
! See THIRD_PARTY_NOTICES.md and LICENSES/scikit-learn-COPYING (BSD-3-Clause).
module sklearn_self_training
  use sklearn_common, only: dp,require,check_xy,check_x
  use sklearn_metrics, only: accuracy_score
  use sklearn_linear_model, only: logistic_regression
  use sklearn_naive_bayes, only: gaussian_nb
  use sklearn_neighbors, only: kneighbors_classifier
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_value,ieee_negative_inf
  implicit none
  private
  public :: self_training_classifier
  type :: self_training_classifier
    real(dp) :: threshold=.75_dp
    character(16) :: criterion='threshold'
    integer :: k_best=10,max_iter=10,n_iter_=0,n_features_in_=0
    character(16) :: termination_condition_=''
    integer, allocatable :: classes_(:),transduction_(:),labeled_iter_(:)
    class(*), allocatable :: estimator_
    logical, private :: fitted_=.false.
  contains
    procedure :: fit=>self_fit
    procedure :: predict_proba=>self_proba
    procedure :: predict_log_proba=>self_log_proba
    procedure :: predict=>self_predict
    procedure :: decision_function=>self_decision
    procedure :: score=>self_score
  end type
contains
  subroutine base_fit(model,x,y)
    class(*), intent(inout) :: model
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    select type(model)
    type is(logistic_regression)
      call model%fit(x,y)
    type is(gaussian_nb)
      call model%fit(x,y)
    type is(kneighbors_classifier)
      call model%fit(x,y)
    class default
      call require(.false.,'unsupported self-training estimator')
    end select
  end subroutine

  function base_classes(model) result(labels)
    class(*), intent(in) :: model
    integer, allocatable :: labels(:)
    select type(model)
    type is(logistic_regression)
      labels=model%classes_
    type is(gaussian_nb)
      labels=model%classes_
    type is(kneighbors_classifier)
      labels=model%classes_
    end select
  end function

  function base_proba(model,x) result(prob)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: prob(:,:)
    select type(model)
    type is(logistic_regression)
      prob=model%predict_proba(x)
    type is(gaussian_nb)
      prob=model%predict_proba(x)
    type is(kneighbors_classifier)
      prob=model%predict_proba(x)
    end select
  end function

  ! Stable merge ordering: equal confidence selects earlier input rows first.
  ! sklearn leaves argpartition's tied-cutoff choice unspecified.
  function confidence_order(values) result(order)
    real(dp), intent(in) :: values(:)
    integer :: order(size(values)),work(size(values)),width,left,middle,right,i,j,k,n
    n=size(values); order=[(i,i=1,n)]; width=1
    do while(width<n)
      left=1
      do while(left<=n-width)
        middle=left+width; right=min(n,left+2*width-1); i=left; j=middle
        do k=left,right
          if(i>=middle) then
            work(k)=order(j); j=j+1
          else if(j>right) then
            work(k)=order(i); i=i+1
          else if(values(order(i))>=values(order(j))) then
            work(k)=order(i); i=i+1
          else
            work(k)=order(j); j=j+1
          end if
        end do
        order(left:right)=work(left:right); left=right+1
      end do
      if(width>n/2) exit
      width=2*width
    end do
  end function

  subroutine self_fit(self,x,y,estimator)
    class(self_training_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    class(*), intent(in) :: estimator
    logical :: labeled(size(y)),selected(size(y))
    integer :: ids(size(y)),i,n,take
    integer, allocatable :: train(:),query(:),labels(:),prediction(:),order(:)
    real(dp), allocatable :: prob(:,:),confidence(:)
    self%fitted_=.false.
    call check_xy(x,size(y)); n=size(y)
    call require(ieee_is_finite(self%threshold),'self-training threshold must be finite')
    call require(self%threshold>=0.and.self%threshold<1,'self-training threshold must lie in [0,1)')
    call require(self%criterion=='threshold'.or.self%criterion=='k_best','unknown self-training criterion')
    call require(self%k_best>0,'self-training k_best must be positive')
    call require(self%max_iter>=-1,'self-training max_iter must be nonnegative or -1 for unlimited')
    call require(any(y/=-1),'self-training requires labeled samples')
    select type(estimator)
    type is(logistic_regression)
    type is(gaussian_nb)
    type is(kneighbors_classifier)
    class default
      call require(.false.,'unsupported self-training estimator')
    end select
    if(allocated(self%estimator_)) deallocate(self%estimator_)
    allocate(self%estimator_,source=estimator)
    self%n_features_in_=size(x,2); self%transduction_=y
    self%labeled_iter_=merge(0,-1,y/=-1); labeled=y/=-1; ids=[(i,i=1,n)]
    self%n_iter_=0; self%termination_condition_=''
    do while(.not.all(labeled))
      if(self%max_iter/=-1.and.self%n_iter_>=self%max_iter) exit
      self%n_iter_=self%n_iter_+1; train=pack(ids,labeled); query=pack(ids,.not.labeled)
      call base_fit(self%estimator_,x(train,:),self%transduction_(train))
      labels=base_classes(self%estimator_); prob=base_proba(self%estimator_,x(query,:))
      confidence=maxval(prob,dim=2); prediction=labels(maxloc(prob,dim=2)); selected=.false.
      if(self%criterion=='threshold') then
        selected(query)=confidence>self%threshold
      else
        take=min(self%k_best,size(query)); order=confidence_order(confidence)
        selected(query(order(:take)))=.true.
      end if
      do i=1,size(query)
        if(.not.selected(query(i))) cycle
        self%transduction_(query(i))=prediction(i); self%labeled_iter_(query(i))=self%n_iter_
      end do
      labeled=labeled.or.selected
      if(.not.any(selected)) then
        self%termination_condition_='no_change'; exit
      end if
    end do
    if(self%n_iter_==self%max_iter) self%termination_condition_='max_iter'
    if(all(labeled)) self%termination_condition_='all_labeled'
    train=pack(ids,labeled)
    ! The source always refits once more, including no-change/zero-iteration fits.
    call base_fit(self%estimator_,x(train,:),self%transduction_(train))
    self%classes_=base_classes(self%estimator_); self%fitted_=.true.
  end subroutine

  subroutine check_query(self,x)
    class(self_training_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    call require(self%fitted_,'self-training classifier is not fitted')
    call check_x(x)
    call require(size(x,2)==self%n_features_in_,'self-training feature count mismatch')
  end subroutine

  function self_proba(self,x) result(prob)
    class(self_training_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: prob(:,:)
    call check_query(self,x); prob=base_proba(self%estimator_,x)
  end function

  function self_log_proba(self,x) result(prob)
    class(self_training_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: prob(:,:)
    integer :: i,j
    call check_query(self,x)
    select type(model=>self%estimator_)
    type is(gaussian_nb)
      prob=model%predict_log_proba(x)
    type is(logistic_regression)
      prob=base_proba(self%estimator_,x)
      do j=1,size(prob,2)
        do i=1,size(prob,1)
          if(prob(i,j)>0) then
            prob(i,j)=log(prob(i,j))
          else
            prob(i,j)=ieee_value(0._dp,ieee_negative_inf)
          end if
        end do
      end do
    class default
      call require(.false.,'self-training estimator has no predict_log_proba')
    end select
  end function

  function self_predict(self,x) result(labels)
    class(self_training_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: labels(size(x,1))
    real(dp), allocatable :: prob(:,:)
    prob=self%predict_proba(x); labels=self%classes_(maxloc(prob,dim=2))
  end function

  function self_decision(self,x) result(values)
    class(self_training_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:,:)
    call check_query(self,x)
    select type(model=>self%estimator_)
    type is(logistic_regression)
      values=model%decision_function(x)
    class default
      call require(.false.,'self-training estimator has no decision_function')
    end select
  end function

  real(dp) function self_score(self,x,y) result(value)
    class(self_training_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    value=accuracy_score(y,self%predict(x))
  end function
end module
