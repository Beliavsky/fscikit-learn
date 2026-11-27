! Voting ensembles adapted from sklearn 1.6.1 ensemble/_voting.py (BSD-3-Clause).
! See THIRD_PARTY_NOTICES.md and LICENSES/scikit-learn-COPYING.
module sklearn_voting
  use sklearn_classic_gradient_boosting,only:reset_boosting_fit
  use sklearn_base_dispatch, only: train_classifier,train_regressor,labels,regression,probabilities
  use sklearn_ieee_math, only: ieee_mean
  use sklearn_common, only: dp,require,check_x,check_xy,unique_labels,label_index
  use sklearn_linear_model, only: linear_regression,elastic_net,logistic_regression,perceptron, &
    passive_aggressive_classifier,passive_aggressive_regressor
  use sklearn_ridge_classification, only: ridge_classifier
  use sklearn_sgd, only: sgd_classifier,sgd_regressor
  use sklearn_sgd_clone, only: reset_sgd_fit
  use sklearn_naive_bayes, only: gaussian_nb,multinomial_nb
  use sklearn_neighbors, only: kneighbors_classifier,kneighbors_regressor
  use sklearn_tree, only: decision_tree_classifier,decision_tree_regressor
  use sklearn_svm, only: svc,linear_svc
  use sklearn_dummy, only: dummy_classifier,dummy_regressor
  use sklearn_kernel_ridge, only: kernel_ridge
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value,ieee_quiet_nan
  implicit none
  private
  public :: voting_classifier,voting_regressor,voting_estimator
  type :: voting_estimator
    character(64) :: name=''
    class(*), allocatable :: estimator
    logical :: drop=.false.
  end type
  type :: voting_classifier
    character(4) :: voting='hard'
    logical :: flatten_transform=.true.
    real(dp), allocatable :: weights(:)
    type(voting_estimator), allocatable :: estimators_(:)
    integer, allocatable :: classes_(:),active_indices_(:)
    integer :: n_features_in_=0
    integer, private :: configured_count_=0
    logical, private :: fitted_=.false.
  contains
    procedure :: fit=>classifier_fit
    procedure :: predict=>classifier_predict
    procedure :: predict_proba=>classifier_proba
    procedure :: transform=>classifier_transform
    procedure :: transform_proba=>classifier_tensor
    procedure :: fit_transform=>classifier_fit_transform
    procedure :: score=>classifier_score
  end type
  type :: voting_regressor
    real(dp), allocatable :: weights(:)
    type(voting_estimator), allocatable :: estimators_(:)
    integer, allocatable :: active_indices_(:)
    integer :: n_features_in_=0
    integer, private :: configured_count_=0
    logical, private :: fitted_=.false.
  contains
    procedure :: fit=>regressor_fit
    procedure :: predict=>regressor_predict
    procedure :: transform=>regressor_transform
    procedure :: fit_transform=>regressor_fit_transform
    procedure :: score=>regressor_score
  end type
contains
  subroutine validate_list(estimators,weights,indices,classifier)
    type(voting_estimator), intent(in) :: estimators(:)
    real(dp), allocatable, intent(in) :: weights(:)
    logical, intent(in) :: classifier
    integer, allocatable, intent(out) :: indices(:)
    integer :: i,j
    call require(size(estimators)>0,'voting requires at least one estimator')
    do i=1,size(estimators)
      call require(len_trim(estimators(i)%name)>0,'voting estimator name must be nonempty')
      call require(index(estimators(i)%name,'__')==0,'voting estimator name cannot contain __')
      call require(.not.any(trim(estimators(i)%name)==[character(20) :: &
        'estimators','weights','n_jobs','verbose']),'voting estimator name conflicts with a parameter')
      if(classifier) call require(.not.any(trim(estimators(i)%name)==[character(20) :: &
        'voting','flatten_transform']),'voting estimator name conflicts with a parameter')
      do j=1,i-1
        call require(estimators(i)%name/=estimators(j)%name,'voting estimator names must be unique')
      end do
      if(.not.estimators(i)%drop) &
        call require(allocated(estimators(i)%estimator),'active voting estimator is missing')
    end do
    indices=pack([(i,i=1,size(estimators))],.not.estimators%drop)
    call require(size(indices)>0,'all voting estimators are dropped')
    if(allocated(weights)) then
      call require(size(weights)==size(estimators),'voting weight count must match configured estimators')
      call require(all(ieee_is_finite(weights)),'voting weights must be finite')
    end if
  end subroutine

  function active_weights(weights,indices,total,average) result(values)
    real(dp), allocatable, intent(in) :: weights(:)
    integer, intent(in) :: indices(:),total
    logical, intent(in) :: average
    real(dp), allocatable :: values(:)
    real(dp) :: mass
    allocate(values(size(indices))); values=1
    if(allocated(weights)) then
      call require(size(weights)==total,'voting weight count must match configured estimators')
      call require(all(ieee_is_finite(weights)),'voting weights must be finite')
      values=weights(indices)
    end if
    if(average) then
      mass=sum(values)
      call require(ieee_is_finite(mass).and.mass/=0,'voting active weights must have a finite nonzero sum')
    end if
  end function

  subroutine training_weights(weights,n)
    real(dp), intent(in), optional :: weights(:)
    integer, intent(in) :: n
    if(.not.present(weights)) return
    call require(size(weights)==n,'voting sample weight length mismatch')
    call require(all(ieee_is_finite(weights)),'voting sample weights must be finite')
  end subroutine

  subroutine query(fitted,width,x)
    logical, intent(in) :: fitted
    integer, intent(in) :: width
    real(dp), intent(in) :: x(:,:)
    call require(fitted,'voting estimator is not fitted'); call check_x(x)
    call require(size(x,2)==width,'voting feature count mismatch')
  end subroutine

  integer function source_argmax(values) result(index_)
    real(dp), intent(in) :: values(:)
    integer :: i
    index_=1
    do i=2,size(values)
      if(ieee_is_nan(values(index_))) return
      if(ieee_is_nan(values(i))) then
        index_=i; return
      end if
      if(values(i)>values(index_)) index_=i
    end do
  end function

  real(dp) function weighted_mean(values,weights) result(value)
    real(dp), intent(in) :: values(:)
    real(dp), intent(in), optional :: weights(:)
    call training_weights(weights,size(values))
    if(present(weights)) then
      call require(sum(weights)/=0,'voting score weights must have a nonzero sum')
      value=sum(values*weights)/sum(weights)
    else
      value=sum(values)/size(values)
    end if
  end function

  subroutine classifier_fit(self,x,y,estimators,sample_weight)
    class(voting_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    type(voting_estimator), intent(in) :: estimators(:)
    real(dp), intent(in), optional :: sample_weight(:)
    integer, allocatable :: encoded(:)
    integer :: i
    self%fitted_=.false.; call check_xy(x,size(y)); call training_weights(sample_weight,size(y))
    call require(self%voting=='hard'.or.self%voting=='soft','voting must be hard or soft')
    call validate_list(estimators,self%weights,self%active_indices_,.true.)
    self%classes_=unique_labels(y); encoded=[(label_index(self%classes_,y(i))-1,i=1,size(y))]
    if(allocated(self%estimators_)) deallocate(self%estimators_)
    allocate(self%estimators_(size(self%active_indices_)))
    do i=1,size(self%estimators_)
      self%estimators_(i)=estimators(self%active_indices_(i))
      call reset_sgd_fit(self%estimators_(i)%estimator)
      call reset_boosting_fit(self%estimators_(i)%estimator)
      call train_classifier(self%estimators_(i)%estimator,x,encoded,sample_weight)
    end do
    self%n_features_in_=size(x,2); self%configured_count_=size(estimators); self%fitted_=.true.
  end subroutine

  subroutine regressor_fit(self,x,y,estimators,sample_weight)
    class(voting_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    type(voting_estimator), intent(in) :: estimators(:)
    real(dp), intent(in), optional :: sample_weight(:)
    integer :: i
    self%fitted_=.false.; call check_xy(x,size(y)); call training_weights(sample_weight,size(y))
    call require(all(ieee_is_finite(y)),'voting regression targets must be finite')
    call validate_list(estimators,self%weights,self%active_indices_,.false.)
    if(allocated(self%estimators_)) deallocate(self%estimators_)
    allocate(self%estimators_(size(self%active_indices_)))
    do i=1,size(self%estimators_)
      self%estimators_(i)=estimators(self%active_indices_(i))
      call reset_sgd_fit(self%estimators_(i)%estimator)
      call reset_boosting_fit(self%estimators_(i)%estimator)
      call train_regressor(self%estimators_(i)%estimator,x,y,sample_weight)
    end do
    self%n_features_in_=size(x,2); self%configured_count_=size(estimators); self%fitted_=.true.
  end subroutine

  function classifier_labels(self,x) result(values)
    class(voting_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: values(:,:)
    integer :: j
    call query(self%fitted_,self%n_features_in_,x)
    allocate(values(size(x,1),size(self%estimators_)))
    do j=1,size(self%estimators_)
      values(:,j)=labels(self%estimators_(j)%estimator,x)
    end do
    call require(all(values>=0).and.all(values<size(self%classes_)),'voting base returned an invalid encoded class')
  end function

  function classifier_tensor(self,x) result(values)
    class(voting_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:,:,:),response(:,:)
    integer :: j
    call query(self%fitted_,self%n_features_in_,x)
    call require(self%voting=='soft','voting probabilities require soft voting')
    allocate(values(size(self%estimators_),size(x,1),size(self%classes_)))
    do j=1,size(self%estimators_)
      response=probabilities(self%estimators_(j)%estimator,x)
      call require(all(shape(response)==[size(x,1),size(self%classes_)]),'voting base probability shape mismatch')
      values(j,:,:)=response
    end do
  end function

  function classifier_proba(self,x) result(values)
    class(voting_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:,:),tensor(:,:,:),w(:)
    integer :: i,k
    tensor=self%transform_proba(x)
    w=active_weights(self%weights,self%active_indices_,self%configured_count_,.true.)
    allocate(values(size(x,1),size(self%classes_)))
    do k=1,size(values,2)
      do i=1,size(values,1)
        if(any(ieee_is_nan(tensor(:,i,k)))) then
          values(i,k)=ieee_value(0._dp,ieee_quiet_nan)
        else
          values(i,k)=ieee_mean(tensor(:,i,k),w)
        end if
      end do
    end do
  end function

  function classifier_predict(self,x) result(values)
    class(voting_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: values(:),votes(:,:)
    real(dp), allocatable :: prob(:,:),counts(:),w(:)
    integer :: i,j,limit
    call query(self%fitted_,self%n_features_in_,x); allocate(values(size(x,1)))
    call require(self%voting=='hard'.or.self%voting=='soft','voting must be hard or soft')
    if(self%voting=='soft') then
      prob=self%predict_proba(x)
      do i=1,size(values)
        values(i)=self%classes_(source_argmax(prob(i,:)))
      end do
    else
      votes=classifier_labels(self,x)
      w=active_weights(self%weights,self%active_indices_,self%configured_count_,.false.)
      allocate(counts(size(self%classes_)))
      do i=1,size(values)
        counts=0; limit=maxval(votes(i,:))+1
        do j=1,size(votes,2)
          counts(votes(i,j)+1)=counts(votes(i,j)+1)+w(j)
        end do
        values(i)=self%classes_(maxloc(counts(:limit),dim=1))
      end do
    end if
  end function

  function classifier_transform(self,x) result(values)
    class(voting_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:,:),tensor(:,:,:)
    integer :: j,k
    call query(self%fitted_,self%n_features_in_,x)
    call require(self%voting=='hard'.or.self%voting=='soft','voting must be hard or soft')
    if(self%voting=='hard') then
      values=real(classifier_labels(self,x),dp)
    else
      call require(self%flatten_transform,'use transform_proba for an unflattened voting tensor')
      tensor=self%transform_proba(x); k=size(tensor,3)
      allocate(values(size(x,1),size(tensor,1)*k))
      do j=1,size(tensor,1)
        values(:,(j-1)*k+1:j*k)=tensor(j,:,:)
      end do
    end if
  end function

  function regressor_transform(self,x) result(values)
    class(voting_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:,:)
    integer :: j
    call query(self%fitted_,self%n_features_in_,x)
    allocate(values(size(x,1),size(self%estimators_)))
    do j=1,size(self%estimators_)
      values(:,j)=regression(self%estimators_(j)%estimator,x)
    end do
  end function

  function regressor_predict(self,x) result(values)
    class(voting_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:),response(:,:),w(:)
    integer :: i
    response=self%transform(x)
    w=active_weights(self%weights,self%active_indices_,self%configured_count_,.true.)
    allocate(values(size(x,1)))
    do i=1,size(values)
      values(i)=ieee_mean(response(i,:),w)
    end do
  end function

  function classifier_fit_transform(self,x,y,estimators,sample_weight) result(values)
    class(voting_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    type(voting_estimator), intent(in) :: estimators(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: values(:,:)
    call self%fit(x,y,estimators,sample_weight); values=self%transform(x)
  end function

  function regressor_fit_transform(self,x,y,estimators,sample_weight) result(values)
    class(voting_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    type(voting_estimator), intent(in) :: estimators(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: values(:,:)
    call self%fit(x,y,estimators,sample_weight); values=self%transform(x)
  end function

  real(dp) function classifier_score(self,x,y,sample_weight) result(value)
    class(voting_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    integer, allocatable :: prediction(:)
    prediction=self%predict(x); call require(size(y)==size(prediction),'voting score target length mismatch')
    value=weighted_mean(merge(1._dp,0._dp,prediction==y),sample_weight)
  end function

  real(dp) function regressor_score(self,x,y,sample_weight) result(value)
    class(voting_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: prediction(:)
    real(dp) :: numerator,denominator,center
    prediction=self%predict(x); call require(size(y)==size(prediction),'voting score target length mismatch')
    call require(all(ieee_is_finite(y)),'voting score targets must be finite')
    call require(all(ieee_is_finite(prediction)),'voting regression score predictions must be finite')
    call training_weights(sample_weight,size(y))
    if(size(y)<2) then
      value=ieee_value(0._dp,ieee_quiet_nan); return
    end if
    center=weighted_mean(y,sample_weight)
    numerator=weighted_mean((y-prediction)**2,sample_weight)
    denominator=weighted_mean((y-center)**2,sample_weight)
    value=1
    if(numerator/=0) then
      value=0
      if(denominator/=0) value=1-numerator/denominator
    end if
  end function

end module
