module sklearn_ensemble
  use sklearn_common
  use sklearn_tree
  use sklearn_metrics, only: accuracy_score,r2_score
  implicit none
  private
  public :: random_forest_classifier,random_forest_regressor,extra_trees_classifier,extra_trees_regressor
  public :: gradient_boosting_regressor
  type :: random_forest_classifier
    integer :: n_estimators=100,max_depth=100,min_samples_leaf=1,min_samples_split=2
    integer :: max_features=0,random_state=0
    logical :: bootstrap=.true.,random_thresholds=.false.
    integer, allocatable :: classes_(:)
    type(decision_tree_classifier), allocatable :: estimators_(:)
    real(dp), allocatable :: feature_importances_(:)
  contains
    procedure :: fit=>forest_classifier_fit
    procedure :: predict_proba=>forest_proba
    procedure :: predict=>forest_classifier_predict
    procedure :: score=>forest_classifier_score
  end type
  type, extends(random_forest_classifier) :: extra_trees_classifier
  contains
    procedure :: fit=>extra_classifier_fit
  end type
  type :: random_forest_regressor
    integer :: n_estimators=100,max_depth=100,min_samples_leaf=1,min_samples_split=2
    integer :: max_features=0,random_state=0
    logical :: bootstrap=.true.,random_thresholds=.false.
    type(decision_tree_regressor), allocatable :: estimators_(:)
    real(dp), allocatable :: feature_importances_(:)
  contains
    procedure :: fit=>forest_regressor_fit
    procedure :: predict=>forest_regressor_predict
    procedure :: score=>forest_regressor_score
  end type
  type, extends(random_forest_regressor) :: extra_trees_regressor
  contains
    procedure :: fit=>extra_regressor_fit
  end type
  type :: gradient_boosting_regressor
    integer :: n_estimators=100,max_depth=3,min_samples_leaf=1,random_state=0
    real(dp) :: learning_rate=0.1_dp,init_=0
    type(decision_tree_regressor), allocatable :: estimators_(:)
    real(dp), allocatable :: train_score_(:),feature_importances_(:)
  contains
    procedure :: fit=>boost_fit
    procedure :: predict=>boost_predict
    procedure :: score=>boost_score
  end type
contains
  function sample_indices(n,bootstrap,seed) result(ids)
    integer, intent(in) :: n,seed
    logical, intent(in) :: bootstrap
    integer :: ids(n),i
    real(dp) :: r(n)
    ids=[(i,i=1,n)]
    if(bootstrap) then
      call seed_rng(seed); call random_number(r); ids=1+int(r*n)
    end if
  end function
  subroutine forest_classifier_fit(self,x,y)
    class(random_forest_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    integer :: i,n,p,features,ids(size(y))
    call check_xy(x,size(y)); call require(self%n_estimators>0,'forest needs positive n_estimators')
    n=size(y); p=size(x,2); self%classes_=unique_labels(y)
    features=self%max_features; if(features==0) features=max(1,int(sqrt(real(p,dp))))
    if(allocated(self%estimators_)) deallocate(self%estimators_)
    allocate(self%estimators_(self%n_estimators)); self%feature_importances_=spread(0._dp,1,p)
    do i=1,self%n_estimators
      ids=sample_indices(n,self%bootstrap,self%random_state+104729*i)
      self%estimators_(i)%max_depth=self%max_depth
      self%estimators_(i)%min_samples_leaf=self%min_samples_leaf
      self%estimators_(i)%min_samples_split=self%min_samples_split
      self%estimators_(i)%max_features=features
      self%estimators_(i)%random_thresholds=self%random_thresholds
      self%estimators_(i)%random_state=self%random_state+7919*i
      call self%estimators_(i)%fit(x(ids,:),y(ids))
      self%feature_importances_=self%feature_importances_+self%estimators_(i)%feature_importances_
    end do
    if(sum(self%feature_importances_)>0) &
      self%feature_importances_=self%feature_importances_/sum(self%feature_importances_)
  end subroutine
  subroutine extra_classifier_fit(self,x,y)
    class(extra_trees_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    self%bootstrap=.false.; self%random_thresholds=.true.
    call forest_classifier_fit(self,x,y)
  end subroutine
  function forest_proba(self,x) result(prob)
    class(random_forest_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: prob(:,:),tree_prob(:,:)
    integer :: i,j,c
    call require(allocated(self%estimators_),'forest is not fitted')
    allocate(prob(size(x,1),size(self%classes_))); prob=0
    do i=1,size(self%estimators_)
      tree_prob=self%estimators_(i)%predict_proba(x)
      do j=1,size(self%estimators_(i)%classes_)
        c=label_index(self%classes_,self%estimators_(i)%classes_(j))
        prob(:,c)=prob(:,c)+tree_prob(:,j)
      end do
    end do
    prob=prob/size(self%estimators_)
  end function
  function forest_classifier_predict(self,x) result(y)
    class(random_forest_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: y(size(x,1)),i
    real(dp), allocatable :: prob(:,:)
    prob=self%predict_proba(x)
    do i=1,size(y)
      y(i)=self%classes_(maxloc(prob(i,:),dim=1))
    end do
  end function
  real(dp) function forest_classifier_score(self,x,y) result(v)
    class(random_forest_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    v=accuracy_score(y,self%predict(x))
  end function
  subroutine forest_regressor_fit(self,x,y)
    class(random_forest_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    integer :: i,n,p,ids(size(y))
    call check_xy(x,size(y)); call require(self%n_estimators>0,'forest needs positive n_estimators')
    n=size(y); p=size(x,2)
    if(allocated(self%estimators_)) deallocate(self%estimators_)
    allocate(self%estimators_(self%n_estimators)); self%feature_importances_=spread(0._dp,1,p)
    do i=1,self%n_estimators
      ids=sample_indices(n,self%bootstrap,self%random_state+104729*i)
      self%estimators_(i)%max_depth=self%max_depth
      self%estimators_(i)%min_samples_leaf=self%min_samples_leaf
      self%estimators_(i)%min_samples_split=self%min_samples_split
      self%estimators_(i)%max_features=self%max_features
      self%estimators_(i)%random_thresholds=self%random_thresholds
      self%estimators_(i)%random_state=self%random_state+7919*i
      call self%estimators_(i)%fit(x(ids,:),y(ids))
      self%feature_importances_=self%feature_importances_+self%estimators_(i)%feature_importances_
    end do
    if(sum(self%feature_importances_)>0) &
      self%feature_importances_=self%feature_importances_/sum(self%feature_importances_)
  end subroutine
  subroutine extra_regressor_fit(self,x,y)
    class(extra_trees_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    self%bootstrap=.false.; self%random_thresholds=.true.
    call forest_regressor_fit(self,x,y)
  end subroutine
  function forest_regressor_predict(self,x) result(y)
    class(random_forest_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: y(size(x,1))
    integer :: i
    call require(allocated(self%estimators_),'forest is not fitted'); y=0
    do i=1,size(self%estimators_)
      y=y+self%estimators_(i)%predict(x)
    end do
    y=y/size(self%estimators_)
  end function
  real(dp) function forest_regressor_score(self,x,y) result(v)
    class(random_forest_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    v=r2_score(y,self%predict(x))
  end function
  subroutine boost_fit(self,x,y)
    class(gradient_boosting_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp) :: residual(size(y)),prediction(size(y))
    integer :: i
    call check_xy(x,size(y)); call require(self%n_estimators>0.and.self%learning_rate>0,'invalid boosting controls')
    self%init_=sum(y)/size(y); prediction=self%init_
    if(allocated(self%estimators_)) deallocate(self%estimators_)
    allocate(self%estimators_(self%n_estimators)); self%train_score_=spread(0._dp,1,self%n_estimators)
    self%feature_importances_=spread(0._dp,1,size(x,2))
    do i=1,self%n_estimators
      residual=y-prediction; self%estimators_(i)%max_depth=self%max_depth
      self%estimators_(i)%min_samples_leaf=self%min_samples_leaf
      self%estimators_(i)%random_state=self%random_state+i
      call self%estimators_(i)%fit(x,residual)
      prediction=prediction+self%learning_rate*self%estimators_(i)%predict(x)
      self%train_score_(i)=sum((y-prediction)**2)/size(y)
      self%feature_importances_=self%feature_importances_+self%estimators_(i)%feature_importances_
    end do
    if(sum(self%feature_importances_)>0) &
      self%feature_importances_=self%feature_importances_/sum(self%feature_importances_)
  end subroutine
  function boost_predict(self,x) result(y)
    class(gradient_boosting_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: y(size(x,1))
    integer :: i
    call require(allocated(self%estimators_),'gradient boosting is not fitted'); y=self%init_
    do i=1,size(self%estimators_)
      y=y+self%learning_rate*self%estimators_(i)%predict(x)
    end do
  end function
  real(dp) function boost_score(self,x,y) result(v)
    class(gradient_boosting_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    v=r2_score(y,self%predict(x))
  end function
end module
