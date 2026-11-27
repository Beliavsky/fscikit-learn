module sklearn_adaboost_regression
  use sklearn_common, only: dp,require,check_x,check_xy,argsort
  use sklearn_tree_features,only:feature_count_control,sample_count_control
  use sklearn_tree, only: decision_tree_regressor
  use sklearn_metrics, only: r2_score
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_boosting_loss,only:boosting_sum
  use iso_fortran_env, only: real32
  use ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: adaboost_regressor
  ! AdaBoost.R2 and dense squared-error CART, following sklearn 1.6.1.
  type :: adaboost_regressor
    integer :: n_estimators=50,n_estimators_=0,n_features_in_=0,random_state=0
    integer :: max_depth=3,min_samples_split=2,min_samples_leaf=1,max_features=0
    real(dp) :: learning_rate=1,min_impurity_decrease=0
    character(16) :: loss='linear'
    type(decision_tree_regressor), allocatable :: estimators_(:)
    real(dp), allocatable :: estimator_weights_(:),estimator_errors_(:),feature_importances_(:)
    integer::max_leaf_nodes=0
    real(dp)::ccp_alpha=0,min_weight_fraction_leaf=0
    type(feature_count_control),private::feature_control_
    type(sample_count_control),private::leaf_control_,split_control_
    type(decision_tree_regressor),private::prototype_
    type(decision_tree_regressor)::estimator_
    integer::tree_random_state=0
    character(16)::criterion='squared_error',splitter='best'
  contains
    procedure,private::features_integer=>adar_features_integer,features_real=>adar_features_real,features_name=>adar_features_name
    generic::set_max_features=>features_integer,features_real,features_name
    procedure,private::leaf_integer=>adar_leaf_integer,leaf_real=>adar_leaf_real
    procedure,private::split_integer=>adar_split_integer,split_real=>adar_split_real
    generic::set_min_samples_leaf=>leaf_integer,leaf_real
    generic::set_min_samples_split=>split_integer,split_real
    procedure :: fit
    procedure :: predict
    procedure :: staged_predict
    procedure :: score
    procedure :: staged_score
  end type
contains
  subroutine adar_features_integer(self,value)
    class(adaboost_regressor),intent(inout)::self
    integer,intent(in)::value
    call self%prototype_%set_max_features(value)
    self%max_features=value
    call self%feature_control_%reset()
  end subroutine
  subroutine adar_features_real(self,value)
    class(adaboost_regressor),intent(inout)::self
    real(dp),intent(in)::value
    call self%prototype_%set_max_features(value)
    self%max_features=0
    call self%feature_control_%set(value)
  end subroutine
  subroutine adar_features_name(self,value)
    class(adaboost_regressor),intent(inout)::self
    character(*),intent(in)::value
    call self%prototype_%set_max_features(value)
    self%max_features=0
    call self%feature_control_%set(value)
  end subroutine
  subroutine adar_leaf_integer(self,value)
    class(adaboost_regressor),intent(inout)::self
    integer,intent(in)::value
    call self%prototype_%set_min_samples_leaf(value)
    self%min_samples_leaf=value
    call self%leaf_control_%reset()
  end subroutine
  subroutine adar_leaf_real(self,value)
    class(adaboost_regressor),intent(inout)::self
    real(dp),intent(in)::value
    call self%prototype_%set_min_samples_leaf(value)
    self%min_samples_leaf=0
    call self%leaf_control_%set(value)
  end subroutine
  subroutine adar_split_integer(self,value)
    class(adaboost_regressor),intent(inout)::self
    integer,intent(in)::value
    call self%prototype_%set_min_samples_split(value)
    self%min_samples_split=value
    call self%split_control_%reset()
  end subroutine
  subroutine adar_split_real(self,value)
    class(adaboost_regressor),intent(inout)::self
    real(dp),intent(in)::value
    call self%prototype_%set_min_samples_split(value)
    self%min_samples_split=0
    call self%split_control_%set(value)
  end subroutine
  function float_features(x) result(z)
    real(dp), intent(in) :: x(:,:)
    real(dp) :: z(size(x,1),size(x,2))
    call check_x(x)
    call require(maxval(abs(x))<=real(huge(0._real32),dp),'AdaBoost features exceed float32 range')
    z=real(real(x,real32),dp)
  end function
  subroutine fit(self,x,y,sample_weight)
    class(adaboost_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    type(legacy_random_state) :: rng
    real(dp) :: z(size(x,1),size(x,2)),w(size(y)),cdf(size(y)),err(size(y)),v,e,beta,total,u
    logical :: positive(size(y))
    integer :: i,j,k,n,seed,ids(size(y)),min_leaf,min_split,nf
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    call require(self%n_estimators>=1.and.self%random_state>=0,'invalid AdaBoost estimator count or seed')
    call require(ieee_is_finite(self%learning_rate),'invalid AdaBoost learning rate')
    call require(self%learning_rate>0,'invalid AdaBoost learning rate')
    call require(self%loss=='linear'.or.self%loss=='square'.or.self%loss=='exponential','invalid AdaBoost loss')
    call require(self%max_depth>=1.and.(self%min_samples_split>=2.or.self%split_control_%fractional(self%min_samples_split)).and. &
      (self%min_samples_leaf>=1.or.self%leaf_control_%fractional(self%min_samples_leaf)),'invalid CART size')
    call require(self%max_features>=0,'invalid CART max_features')
    call require(any(self%criterion==[character(16)::'squared_error','friedman_mse','absolute_error','poisson']),'invalid AdaBoost regression criterion')
    call require(self%splitter=='best'.or.self%splitter=='random','invalid AdaBoost regression splitter')
    call require(ieee_is_finite(self%min_impurity_decrease),'invalid impurity decrease')
    call require(self%min_impurity_decrease>=0,'invalid impurity decrease')
    min_leaf=self%leaf_control_%resolve(self%min_samples_leaf,size(y),1,.false.)
    min_split=self%split_control_%resolve(self%min_samples_split,size(y),2,.true.)
    nf=self%feature_control_%resolve(self%max_features,size(x,2),.false.)
    call require(self%max_leaf_nodes==0.or.self%max_leaf_nodes>=2,'AdaBoost max_leaf_nodes must be zero or at least two')
    call require(ieee_is_finite(self%ccp_alpha),'AdaBoost ccp_alpha must be finite')
    call require(self%ccp_alpha>=0,'AdaBoost ccp_alpha must be nonnegative')
    call require(ieee_is_finite(self%min_weight_fraction_leaf),'AdaBoost weighted leaf fraction must be finite')
    call require(self%min_weight_fraction_leaf>=0.and.self%min_weight_fraction_leaf<=.5_dp,'AdaBoost weighted leaf fraction must be in [0,.5]')
    n=size(y); z=float_features(x); w=1
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'sample weight shape mismatch')
      call require(all(ieee_is_finite(sample_weight)),'invalid sample weights')
      call require(all(sample_weight>=0),'invalid sample weights')
      w=sample_weight
    end if
    total=boosting_sum(w)
    call require(ieee_is_finite(total).and.total>0,'sample weights must have positive finite sum')
    w=w/total; positive=w>0
    if(allocated(self%estimators_)) deallocate(self%estimators_)
    allocate(self%estimators_(self%n_estimators))
    self%estimator_weights_=spread(0._dp,1,self%n_estimators)
    self%estimator_errors_=spread(1._dp,1,self%n_estimators)
    self%feature_importances_=spread(0._dp,1,size(x,2))
    self%n_features_in_=size(x,2); self%n_estimators_=0
    self%estimator_=self%prototype_
    self%estimator_%max_depth=self%max_depth;self%estimator_%min_samples_split=self%min_samples_split
    self%estimator_%min_samples_leaf=self%min_samples_leaf;self%estimator_%max_features=self%max_features
    self%estimator_%criterion=self%criterion;self%estimator_%random_thresholds=self%splitter=='random'
    self%estimator_%random_state=self%tree_random_state;self%estimator_%min_impurity_decrease=self%min_impurity_decrease
    self%estimator_%max_leaf_nodes=self%max_leaf_nodes;self%estimator_%ccp_alpha=self%ccp_alpha
    self%estimator_%min_weight_fraction_leaf=self%min_weight_fraction_leaf
    call rng%seed(self%random_state)
    do k=1,self%n_estimators
      where(positive) w=max(w,epsilon(1._dp))
      seed=rng%integer_index(2147483647)-1
      cdf(1)=w(1)
      do i=2,n
        cdf(i)=cdf(i-1)+w(i)
      end do
      cdf=cdf/cdf(n)
      do i=1,n
        u=rng%uniform(); j=1
        do while(j<n)
          if(u<cdf(j)) exit
          j=j+1
        end do
        ids(i)=j
      end do
      self%estimators_(k)=self%estimator_
      self%estimators_(k)%random_state=seed
      call self%estimators_(k)%fit(z(ids,:),y(ids))
      err=abs(self%estimators_(k)%predict(z)-y); v=maxval(err,mask=positive)
      where(.not.positive) err=0
      if(v>0) err=err/v
      select case(trim(self%loss))
      case('square'); err=err**2
      case('exponential'); err=1-exp(-err)
      end select
      e=boosting_sum(pack(w*err,positive))
      if(e>=.5_dp) then
        if(k==1) self%n_estimators_=1
        exit
      end if
      self%n_estimators_=k
      self%estimator_errors_(k)=e
      if(e<=0) then
        self%estimator_weights_(k)=1
        exit
      end if
      beta=e/(1-e); self%estimator_weights_(k)=self%learning_rate*log(1/beta)
      if(k<self%n_estimators) then
        where(positive) w=w*beta**((1-err)*self%learning_rate)
      end if
      total=boosting_sum(w)
      if(.not.ieee_is_finite(total).or.total<=0) exit
      if(k<self%n_estimators) w=w/total
    end do
    self%estimators_=self%estimators_(:self%n_estimators_)
    total=boosting_sum(self%estimator_weights_)
    if(total>0) then
      do k=1,self%n_estimators_
        self%feature_importances_=self%feature_importances_+ &
          self%estimator_weights_(k)*self%estimators_(k)%feature_importances_
      end do
      self%feature_importances_=self%feature_importances_/total
    end if
  end subroutine
  function staged_predict(self,x) result(out)
    class(adaboost_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: out(:,:),values(:,:),z(:,:)
    real(dp) :: cumulative,total
    integer, allocatable :: order(:)
    integer :: i,j,k
    call require(allocated(self%estimators_),'AdaBoostRegressor is not fitted')
    call require(size(x,2)==self%n_features_in_,'AdaBoost feature count mismatch')
    z=float_features(x)
    allocate(values(size(x,1),self%n_estimators_),out(size(x,1),self%n_estimators_))
    do k=1,self%n_estimators_
      values(:,k)=self%estimators_(k)%predict(z)
      do i=1,size(x,1)
        order=argsort(values(i,:k)); total=0; cumulative=0
        do j=1,k;total=total+self%estimator_weights_(order(j));end do
        do j=1,k
          cumulative=cumulative+self%estimator_weights_(order(j))
          if(cumulative>=.5_dp*total) exit
        end do
        out(i,k)=values(i,order(j))
      end do
    end do
  end function
  function predict(self,x) result(out)
    class(adaboost_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: out(size(x,1))
    real(dp), allocatable :: stages(:,:)
    stages=self%staged_predict(x); out=stages(:,self%n_estimators_)
  end function
  real(dp) function score(self,x,y,sample_weight) result(v)
    class(adaboost_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    v=r2_score(y,self%predict(x),sample_weight=sample_weight)
  end function
  function staged_score(self,x,y,sample_weight) result(v)
    class(adaboost_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    real(dp), allocatable :: v(:),stages(:,:)
    integer :: k
    stages=self%staged_predict(x); allocate(v(self%n_estimators_))
    do k=1,size(v)
      v(k)=r2_score(y,stages(:,k),sample_weight=sample_weight)
    end do
  end function
end module
