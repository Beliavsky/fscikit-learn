! sklearn 1.6.1 ensemble/_gb.py and _loss/: retained BSD notices apply.
! Native typed interfaces and integration are original MIT contributions.
module sklearn_classic_gradient_boosting
  use sklearn_common
  use sklearn_tree,only:decision_tree_regressor
  use sklearn_tree_features,only:feature_count_control,sample_count_control
  use sklearn_legacy_rng,only:legacy_random_state
  use sklearn_boosting_loss,only:boosting_sum,boosting_mean,boosting_quantile,boosting_regression_loss,boosting_logit
  use sklearn_ieee_math,only:ieee_difference,ieee_product,ieee_quotient
  use sklearn_metrics,only:accuracy_score,r2_score
  use,intrinsic::ieee_arithmetic,only:ieee_is_finite,ieee_value,ieee_quiet_nan,ieee_positive_inf,ieee_negative_inf
  use,intrinsic::iso_fortran_env,only:real32
  implicit none
  private
  public::gradient_boosting_regressor,gradient_boosting_classifier
  public::reset_boosting_fit
  type :: gradient_boosting_classifier
    integer :: n_estimators=100,max_depth=3,min_samples_leaf=1,min_samples_split=2,max_features=0,random_state=0
    integer :: n_estimators_=0,n_trees_per_iteration_=0,n_features_in_=0,n_classes_=0,max_leaf_nodes=0
    real(dp) :: learning_rate=.1_dp,min_impurity_decrease=0,ccp_alpha=0,min_weight_fraction_leaf=0
    real(dp)::subsample=1,oob_score_=0
    real(dp),allocatable::oob_scores_(:),oob_improvement_(:)
    integer::max_features_=0
    type(feature_count_control),private::feature_control_
    type(sample_count_control),private::leaf_control_,split_control_
    character(16)::criterion='friedman_mse'
    character(16) :: loss='log_loss'
    logical :: init_zero=.false.,warm_start=.false.
    logical,private::fitted_init_zero_=.false.
    type(legacy_random_state),private::rng_
    integer, allocatable :: classes_(:)
    real(dp), allocatable :: init_(:),class_prior_(:),train_score_(:),feature_importances_(:)
    type(decision_tree_regressor), allocatable :: estimators_(:,:)
    real(dp), private :: fitted_rate=.1_dp
    character(16), private :: fitted_loss='log_loss'
  contains
    procedure,private::leaf_integer=>gbc_leaf_integer,leaf_real=>gbc_leaf_real
    procedure,private::split_integer=>gbc_split_integer,split_real=>gbc_split_real
    generic::set_min_samples_leaf=>leaf_integer,leaf_real
    generic::set_min_samples_split=>split_integer,split_real
    procedure,private::features_integer=>gbc_features_integer,features_real=>gbc_features_real,features_name=>gbc_features_name
    generic::set_max_features=>features_integer,features_real,features_name
    procedure :: fit=>gbc_fit
    procedure :: decision_function=>gbc_decision
    procedure :: predict_proba=>gbc_proba
    procedure :: predict_log_proba=>gbc_log_proba
    procedure :: predict=>gbc_predict
    procedure :: score=>gbc_score
    procedure :: staged_decision_function=>gbc_staged_decision
    procedure :: staged_predict_proba=>gbc_staged_proba
    procedure :: staged_predict=>gbc_staged_predict
    procedure :: apply=>gbc_apply
  end type
  type :: gradient_boosting_regressor
    integer :: n_estimators=100,max_depth=3,min_samples_leaf=1,min_samples_split=2,random_state=0
    integer::max_features=0,max_leaf_nodes=0,n_features_in_=0,n_estimators_=0,n_trees_per_iteration_=1,max_features_=0
    real(dp) :: learning_rate=0.1_dp,init_=0,alpha=.9_dp,min_impurity_decrease=0,min_weight_fraction_leaf=0,ccp_alpha=0
    real(dp)::subsample=1,oob_score_=0
    real(dp),allocatable::oob_scores_(:),oob_improvement_(:)
    character(16)::loss='squared_error',criterion='friedman_mse'
    logical::init_zero=.false.,warm_start=.false.
    type(legacy_random_state),private::rng_
    type(feature_count_control),private::feature_control_
    type(sample_count_control),private::leaf_control_,split_control_
    type(decision_tree_regressor), allocatable :: estimators_(:)
    real(dp), allocatable :: train_score_(:),feature_importances_(:),huber_delta_(:)
  contains
    procedure,private::leaf_integer=>gbr_leaf_integer,leaf_real=>gbr_leaf_real
    procedure,private::split_integer=>gbr_split_integer,split_real=>gbr_split_real
    generic::set_min_samples_leaf=>leaf_integer,leaf_real
    generic::set_min_samples_split=>split_integer,split_real
    procedure :: fit=>boost_fit
    procedure :: predict=>boost_predict
    procedure :: score=>boost_score
    procedure::staged_predict=>boost_staged_predict,apply=>boost_apply
    procedure,private::features_integer=>gbr_features_integer,features_real=>gbr_features_real,features_name=>gbr_features_name
    generic::set_max_features=>features_integer,features_real,features_name
  end type
contains
  subroutine gbr_leaf_integer(self,value)
    class(gradient_boosting_regressor),intent(inout)::self
    integer,intent(in)::value
    self%min_samples_leaf=value;call self%leaf_control_%reset()
  end subroutine
  subroutine gbr_leaf_real(self,value)
    class(gradient_boosting_regressor),intent(inout)::self
    real(dp),intent(in)::value
    self%min_samples_leaf=0;call self%leaf_control_%set(value)
  end subroutine
  subroutine gbr_split_integer(self,value)
    class(gradient_boosting_regressor),intent(inout)::self
    integer,intent(in)::value
    self%min_samples_split=value;call self%split_control_%reset()
  end subroutine
  subroutine gbr_split_real(self,value)
    class(gradient_boosting_regressor),intent(inout)::self
    real(dp),intent(in)::value
    self%min_samples_split=0;call self%split_control_%set(value)
  end subroutine
  subroutine gbc_leaf_integer(self,value)
    class(gradient_boosting_classifier),intent(inout)::self
    integer,intent(in)::value
    self%min_samples_leaf=value;call self%leaf_control_%reset()
  end subroutine
  subroutine gbc_leaf_real(self,value)
    class(gradient_boosting_classifier),intent(inout)::self
    real(dp),intent(in)::value
    self%min_samples_leaf=0;call self%leaf_control_%set(value)
  end subroutine
  subroutine gbc_split_integer(self,value)
    class(gradient_boosting_classifier),intent(inout)::self
    integer,intent(in)::value
    self%min_samples_split=value;call self%split_control_%reset()
  end subroutine
  subroutine gbc_split_real(self,value)
    class(gradient_boosting_classifier),intent(inout)::self
    real(dp),intent(in)::value
    self%min_samples_split=0;call self%split_control_%set(value)
  end subroutine
  ! Native clone sites keep constructor parameters and discard fitted state.
  subroutine reset_boosting_fit(model)
    class(*),intent(inout)::model
    select type(model)
    type is(gradient_boosting_regressor)
      if(allocated(model%estimators_))deallocate(model%estimators_)
      if(allocated(model%train_score_))deallocate(model%train_score_)
      if(allocated(model%feature_importances_))deallocate(model%feature_importances_)
      if(allocated(model%huber_delta_))deallocate(model%huber_delta_)
      if(allocated(model%oob_scores_))deallocate(model%oob_scores_)
      if(allocated(model%oob_improvement_))deallocate(model%oob_improvement_)
      model%n_estimators_=0;model%n_features_in_=0;model%max_features_=0
      model%n_trees_per_iteration_=1;model%init_=0;model%oob_score_=0
    type is(gradient_boosting_classifier)
      if(allocated(model%estimators_))deallocate(model%estimators_)
      if(allocated(model%train_score_))deallocate(model%train_score_)
      if(allocated(model%feature_importances_))deallocate(model%feature_importances_)
      if(allocated(model%init_))deallocate(model%init_)
      if(allocated(model%classes_))deallocate(model%classes_)
      if(allocated(model%class_prior_))deallocate(model%class_prior_)
      if(allocated(model%oob_scores_))deallocate(model%oob_scores_)
      if(allocated(model%oob_improvement_))deallocate(model%oob_improvement_)
      model%n_estimators_=0;model%n_features_in_=0;model%max_features_=0
      model%n_trees_per_iteration_=0;model%n_classes_=0;model%oob_score_=0
      model%fitted_init_zero_=.false.;model%fitted_loss='log_loss';model%fitted_rate=.1_dp
    end select
  end subroutine
  ! np.resize repeats the old values when enlarging warm-start histories.
  subroutine resize_boosting_values(values,n)
    real(dp),allocatable,intent(inout)::values(:)
    integer,intent(in)::n
    real(dp),allocatable::copy(:)
    integer::i,m
    allocate(copy(n));copy=0
    if(allocated(values))then
      m=size(values)
      if(m>0)then
        do i=1,n;copy(i)=values(mod(i-1,m)+1);end do
      end if
    end if
    call move_alloc(copy,values)
  end subroutine
  ! sklearn ensemble/_gradient_boosting.pyx _random_sample_mask (BSD-3-Clause).
  function boosting_sample_mask(rng,n,n_inbag)result(mask)
    type(legacy_random_state),intent(inout)::rng
    integer,intent(in)::n,n_inbag
    logical::mask(n)
    integer::i,bagged
    bagged=0
    do i=1,n
      mask(i)=ieee_product(rng%uniform(),real(n-i+1,dp))<real(n_inbag-bagged,dp)
      if(mask(i))bagged=bagged+1
    end do
  end function
  subroutine check_boosting_subsample(value)
    real(dp),intent(in)::value
    call require(ieee_is_finite(value),'gradient boosting subsample must be finite')
    call require(value>0.and.value<=1,'gradient boosting subsample must be in (0,1]')
  end subroutine
  subroutine check_boosting_controls(depth,split,leaf,budget,features,seed,decrease,alpha,fraction,p)
    integer,intent(in)::depth,split,leaf,budget,features,seed,p
    real(dp),intent(in)::decrease,alpha,fraction
    call require(depth>0.and.split>=2.and.leaf>=1,'invalid boosting tree size controls')
    call require(budget>=0.and.budget/=1,'boosting max_leaf_nodes must be zero or at least two')
    call require(features>=0,'invalid boosting max_features')
    call require(seed>=0,'boosting random_state must be nonnegative')
    call require(ieee_is_finite(decrease),'boosting impurity decrease must be finite')
    call require(decrease>=0,'boosting impurity decrease must be nonnegative')
    call require(ieee_is_finite(alpha),'boosting ccp_alpha must be finite')
    call require(alpha>=0,'boosting ccp_alpha must be nonnegative')
    call require(ieee_is_finite(fraction),'boosting minimum weighted leaf fraction must be finite')
    call require(fraction>=0.and.fraction<=.5_dp,'boosting minimum weighted leaf fraction must be in [0,0.5]')
  end subroutine
  subroutine gbr_features_integer(self,value)
    class(gradient_boosting_regressor),intent(inout)::self
    integer,intent(in)::value
    self%max_features=value;call self%feature_control_%reset()
  end subroutine
  subroutine gbr_features_real(self,value)
    class(gradient_boosting_regressor),intent(inout)::self
    real(dp),intent(in)::value
    self%max_features=0;call self%feature_control_%set(value)
  end subroutine
  subroutine gbr_features_name(self,value)
    class(gradient_boosting_regressor),intent(inout)::self
    character(*),intent(in)::value
    self%max_features=0;call self%feature_control_%set(value)
  end subroutine
  subroutine gbc_features_integer(self,value)
    class(gradient_boosting_classifier),intent(inout)::self
    integer,intent(in)::value
    self%max_features=value;call self%feature_control_%reset()
  end subroutine
  subroutine gbc_features_real(self,value)
    class(gradient_boosting_classifier),intent(inout)::self
    real(dp),intent(in)::value
    self%max_features=0;call self%feature_control_%set(value)
  end subroutine
  subroutine gbc_features_name(self,value)
    class(gradient_boosting_classifier),intent(inout)::self
    character(*),intent(in)::value
    self%max_features=0;call self%feature_control_%set(value)
  end subroutine
  subroutine boost_fit(self,x,y,sample_weight)
    class(gradient_boosting_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    type(legacy_random_state)::rng
    type(decision_tree_regressor),allocatable::grown(:)
    real(dp)::residual(size(y)),prediction(size(y)),gradient(size(y)),w(size(y)),terms(size(y)),delta,median,value,total
    real(dp)::fit_w(size(y)),previous_loss
    logical::sample_mask(size(y)),do_oob,continuing
    integer::i,j,leaf,nontrivial,n_inbag,begin
    integer,allocatable::leaves(:),indices(:)
    call check_xy(x,size(y));call check_x(reshape(y,[size(y),1]))
    call check_boosting_subsample(self%subsample)
    call require(self%n_estimators>0,'GradientBoostingRegressor needs positive n_estimators')
    call require(ieee_is_finite(self%learning_rate),'GradientBoostingRegressor learning_rate must be finite')
    call require(self%learning_rate>=0,'GradientBoostingRegressor learning_rate must be nonnegative')
    call require(ieee_is_finite(self%alpha),'GradientBoostingRegressor alpha must be finite')
    call require(self%alpha>0.and.self%alpha<1,'GradientBoostingRegressor alpha must be in (0,1)')
    call require(any(self%loss==[character(16)::'squared_error','absolute_error','quantile','huber']),'unknown boosting regression loss')
    call require(any(self%criterion==[character(16)::'friedman_mse','squared_error']),'invalid boosting tree criterion')
    call check_boosting_controls(self%max_depth, &
      self%split_control_%resolve(self%min_samples_split,size(x,1),2,.true.), &
      self%leaf_control_%resolve(self%min_samples_leaf,size(x,1),1,.false.),self%max_leaf_nodes, &
      self%max_features,self%random_state,self%min_impurity_decrease,self%ccp_alpha,self%min_weight_fraction_leaf,size(x,2))
    call require(self%max_depth>0,'gradient boosting max_depth must be positive')
    self%max_features_=self%feature_control_%resolve(self%max_features,size(x,2),.false.)
    self%n_features_in_=size(x,2);self%n_estimators_=self%n_estimators;self%n_trees_per_iteration_=1
    w=1
    if(present(sample_weight))then
      call require(size(sample_weight)==size(y),'gradient boosting sample weight length mismatch')
      call require(all(ieee_is_finite(sample_weight)),'gradient boosting sample weights must be finite')
      w=sample_weight
    end if
    continuing=self%warm_start.and.allocated(self%estimators_);begin=0
    if(continuing)then
      begin=size(self%estimators_)
      call require(self%n_estimators>=begin,'gradient boosting warm start cannot reduce n_estimators')
      prediction=self%predict(x);rng=self%rng_
      allocate(grown(self%n_estimators));grown(:begin)=self%estimators_
      call move_alloc(grown,self%estimators_)
      call resize_boosting_values(self%train_score_,self%n_estimators)
      call resize_boosting_values(self%huber_delta_,self%n_estimators)
    else
      self%init_=0
      if(.not.self%init_zero)then
      select case(self%loss)
      case('squared_error');self%init_=boosting_mean(y,w)
      case('absolute_error','huber');self%init_=boosting_quantile(y,w,.5_dp,present(sample_weight))
      case('quantile');self%init_=boosting_quantile(y,w,self%alpha,present(sample_weight))
      end select
      end if
      prediction=self%init_;call rng%seed(self%random_state)
      if(allocated(self%estimators_)) deallocate(self%estimators_)
      allocate(self%estimators_(self%n_estimators)); self%train_score_=spread(0._dp,1,self%n_estimators)
      self%huber_delta_=spread(0._dp,1,self%n_estimators)
    end if
    self%feature_importances_=spread(0._dp,1,size(x,2))
    nontrivial=0
    do_oob=self%subsample<1;n_inbag=max(1,int(self%subsample*size(y)))
    sample_mask=.true.
    if(.not.continuing)then
      self%oob_score_=0
      if(allocated(self%oob_scores_))deallocate(self%oob_scores_,self%oob_improvement_)
    end if
    if(do_oob.or.allocated(self%oob_scores_))then
      call resize_boosting_values(self%oob_scores_,self%n_estimators)
      call resize_boosting_values(self%oob_improvement_,self%n_estimators)
      if(continuing)self%oob_score_=ieee_value(0._dp,ieee_quiet_nan)
    end if
    if(begin<self%n_estimators.and.self%subsample==1) &
      call require(boosting_sum(w)/=0,'gradient boosting weights must have nonzero total')
    if(do_oob.and.begin>0)previous_loss=self%oob_scores_(begin)
    do i=begin+1,self%n_estimators
      if(do_oob)then
        sample_mask=boosting_sample_mask(rng,size(y),n_inbag)
        if(i==1)previous_loss=boosting_regression_loss(self%loss,pack(y,.not.sample_mask), &
          pack(prediction,.not.sample_mask),pack(w,.not.sample_mask),self%alpha,.5_dp)
      end if
      fit_w=merge(w,0._dp,sample_mask)
      do j=1,size(y);residual(j)=ieee_difference(y(j),prediction(j));end do
      delta=0
      select case(self%loss)
      case('squared_error');gradient=residual
      case('absolute_error');gradient=merge(1._dp,-1._dp,residual>=0)
      case('quantile');gradient=merge(self%alpha,self%alpha-1,residual>=0)
      case('huber')
        delta=boosting_quantile(abs(residual),w,self%alpha,.true.)
        gradient=sign(min(abs(residual),delta),residual)
      end select
      self%huber_delta_(i)=delta
      self%estimators_(i)%max_depth=self%max_depth
      self%estimators_(i)%min_samples_leaf=self%min_samples_leaf
      if(self%leaf_control_%fractional(self%min_samples_leaf)) &
        call self%estimators_(i)%set_min_samples_leaf(self%leaf_control_%value())
      self%estimators_(i)%min_samples_split=self%min_samples_split
      if(self%split_control_%fractional(self%min_samples_split)) &
        call self%estimators_(i)%set_min_samples_split(self%split_control_%value())
      self%estimators_(i)%criterion=self%criterion;self%estimators_(i)%random_state=self%random_state
      self%estimators_(i)%max_leaf_nodes=self%max_leaf_nodes;self%estimators_(i)%ccp_alpha=self%ccp_alpha
      self%estimators_(i)%min_weight_fraction_leaf=self%min_weight_fraction_leaf
      self%estimators_(i)%min_impurity_decrease=self%min_impurity_decrease
      call self%estimators_(i)%set_max_features(self%max_features_)
      call self%estimators_(i)%fit_with_rng(x,gradient,rng,fit_w)
      leaves=self%estimators_(i)%apply(x)
      if(self%loss/='squared_error')then
        do leaf=1,self%estimators_(i)%node_count_
          if(self%estimators_(i)%nodes_(leaf)%left/=0)cycle
          indices=pack([(j,j=1,size(y))],leaves==leaf.and.sample_mask)
          if(self%loss=='quantile')then
            value=boosting_quantile(residual(indices),w(indices),self%alpha,.true.)
          else
            value=boosting_quantile(residual(indices),w(indices),.5_dp,.true.)
            if(self%loss=='huber')then
              median=value
              do j=1,size(indices)
                total=ieee_difference(residual(indices(j)),median)
                terms(j)=sign(min(delta,abs(total)),total)
              end do
              value=ieee_difference(median,-boosting_mean(terms(:size(indices)),w(indices)))
            end if
          end if
          self%estimators_(i)%nodes_(leaf)%value=value;self%estimators_(i)%nodes_(leaf)%values(1)=value
        end do
      end if
      terms=self%estimators_(i)%predict(x)
      do j=1,size(y);prediction(j)=ieee_difference(prediction(j),-ieee_product(self%learning_rate,terms(j)));end do
      if(do_oob)then
        self%train_score_(i)=boosting_regression_loss(self%loss,pack(y,sample_mask), &
          pack(prediction,sample_mask),pack(w,sample_mask),self%alpha,delta)
        self%oob_scores_(i)=boosting_regression_loss(self%loss,pack(y,.not.sample_mask), &
          pack(prediction,.not.sample_mask),pack(w,.not.sample_mask),self%alpha,delta)
        self%oob_improvement_(i)=ieee_difference(previous_loss,self%oob_scores_(i))
        previous_loss=self%oob_scores_(i);self%oob_score_=previous_loss
      else
        self%train_score_(i)=boosting_regression_loss(self%loss,y,prediction,w,self%alpha,delta)
      end if
    end do
    do i=1,size(self%estimators_)
      if(self%estimators_(i)%node_count_>1)then
        self%feature_importances_=self%feature_importances_+boost_tree_importances(self%estimators_(i))
        nontrivial=nontrivial+1
      end if
    end do
    if(nontrivial>0)then
      self%feature_importances_=self%feature_importances_/nontrivial;total=boosting_sum(self%feature_importances_)
      do j=1,size(x,2);self%feature_importances_(j)=ieee_quotient(self%feature_importances_(j),total);end do
    end if
    self%rng_=rng;self%n_estimators_=self%n_estimators
    if(begin==self%n_estimators)then
      self%n_estimators_=begin+1
      if(allocated(self%oob_scores_))self%oob_score_=self%oob_scores_(self%n_estimators)
    end if
  end subroutine
  function boost_tree_importances(tree)result(values)
    type(decision_tree_regressor),intent(in)::tree
    real(dp)::values(tree%n_features_in_),decrease
    integer::j,left,right,feature
    values=0
    do j=1,tree%node_count_
      left=tree%nodes_(j)%left;if(left==0)cycle
      right=tree%nodes_(j)%right;feature=tree%nodes_(j)%feature
      decrease=ieee_difference(ieee_product(tree%nodes_(j)%weighted_n_samples,tree%nodes_(j)%impurity), &
        ieee_product(tree%nodes_(left)%weighted_n_samples,tree%nodes_(left)%impurity))
      decrease=ieee_difference(decrease,ieee_product(tree%nodes_(right)%weighted_n_samples,tree%nodes_(right)%impurity))
      values(feature)=ieee_difference(values(feature),-decrease)
    end do
    do j=1,size(values);values(j)=ieee_quotient(values(j),tree%nodes_(1)%weighted_n_samples);end do
  end function
  function boost_predict(self,x) result(y)
    class(gradient_boosting_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: y(size(x,1))
    integer :: i
    call require(allocated(self%estimators_),'gradient boosting is not fitted'); y=self%init_
    call check_x(x)
    call require(size(x,2)==self%n_features_in_,'gradient boosting query width mismatch')
    do i=1,size(self%estimators_)
      y=y+self%learning_rate*self%estimators_(i)%predict(x)
    end do
  end function
  real(dp) function boost_score(self,x,y,sample_weight) result(v)
    class(gradient_boosting_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    v=r2_score(y,self%predict(x),sample_weight=sample_weight)
  end function
  function boost_staged_predict(self,x)result(stages)
    class(gradient_boosting_regressor),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),allocatable::stages(:,:),prediction(:)
    integer::i
    call require(allocated(self%estimators_),'gradient boosting is not fitted')
    call check_x(x)
    call require(size(x,2)==self%n_features_in_,'gradient boosting query width mismatch')
    allocate(stages(size(x,1),size(self%estimators_)),prediction(size(x,1)));prediction=self%init_
    do i=1,size(self%estimators_)
      prediction=prediction+self%learning_rate*self%estimators_(i)%predict(x);stages(:,i)=prediction
    end do
  end function
  function boost_apply(self,x)result(leaves)
    class(gradient_boosting_regressor),intent(in)::self
    real(dp),intent(in)::x(:,:)
    integer,allocatable::leaves(:,:)
    integer::i
    call require(allocated(self%estimators_),'gradient boosting is not fitted')
    call check_x(x)
    call require(size(x,2)==self%n_features_in_,'gradient boosting query width mismatch')
    allocate(leaves(size(x,1),size(self%estimators_)))
    do i=1,size(self%estimators_);leaves(:,i)=self%estimators_(i)%apply(x);end do
  end function
  real(dp) function gbc_softplus(x) result(v)
    real(dp), intent(in) :: x
    v=max(x,0._dp)+log(1+exp(-abs(x)))
  end function

  function gbc_probabilities(self,raw) result(prob)
    class(gradient_boosting_classifier), intent(in) :: self
    real(dp), intent(in) :: raw(:,:)
    real(dp), allocatable :: prob(:,:)
    real(dp) :: scale
    integer :: i
    allocate(prob(size(raw,1),size(self%classes_)))
    if(size(self%classes_)==2) then
      scale=1
      if(self%fitted_loss=='exponential') scale=2
      prob(:,2)=sigmoid(scale*raw(:,1)); prob(:,1)=1-prob(:,2)
    else
      do i=1,size(raw,1)
        prob(i,:)=exp(raw(i,:)-maxval(raw(i,:)))
        prob(i,:)=prob(i,:)/sum(prob(i,:))
      end do
    end if
  end function

  subroutine gbc_fit(self,x,y,sample_weight)
    class(gradient_boosting_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    real(dp)::w(size(y)),loss_values(size(y)),fit_w(size(y)),previous_loss
    real(dp), allocatable :: raw(:,:),prob(:,:),gradient(:,:),indicator(:),hessian(:),logs(:),class_mass(:)
    integer, allocatable :: encoded(:),leaves(:)
    type(legacy_random_state)::rng
    type(decision_tree_regressor),allocatable::grown(:,:)
    logical, allocatable :: mask(:),sample_mask(:)
    logical::do_oob,continuing
    real(dp) :: floor_prior,numerator,denominator,factor,v
    integer :: n,p,c,k,t,i,leaf,left,nontrivial,n_inbag,begin
    call check_xy(x,size(y)); n=size(y); p=size(x,2)
    call check_boosting_subsample(self%subsample)
    call require(self%n_estimators>=1,'GradientBoostingClassifier n_estimators must be positive')
    call require(ieee_is_finite(self%learning_rate),'nonfinite GradientBoostingClassifier learning_rate')
    call require(self%learning_rate>=0,'GradientBoostingClassifier learning_rate must be nonnegative')
    call check_boosting_controls(self%max_depth, &
      self%split_control_%resolve(self%min_samples_split,size(x,1),2,.true.), &
      self%leaf_control_%resolve(self%min_samples_leaf,size(x,1),1,.false.),self%max_leaf_nodes, &
      self%max_features,self%random_state,self%min_impurity_decrease,self%ccp_alpha,self%min_weight_fraction_leaf,p)
    self%max_features_=self%feature_control_%resolve(self%max_features,p,.false.)
    call require(any(self%criterion==[character(16)::'friedman_mse','squared_error']),'invalid boosting tree criterion')
    call require(self%loss=='log_loss'.or.self%loss=='exponential','invalid GradientBoostingClassifier loss')
    self%classes_=unique_labels(y); c=size(self%classes_)
    call require(c>=2,'GradientBoostingClassifier requires at least two classes')
    w=1
    if(present(sample_weight))then
      call require(size(sample_weight)==n,'GradientBoostingClassifier sample weight length mismatch')
      call require(all(ieee_is_finite(sample_weight)),'GradientBoostingClassifier sample weights must be finite')
      w=sample_weight
    end if
    encoded=[(label_index(self%classes_,y(i)),i=1,n)];allocate(class_mass(c));class_mass=0
    ! Match DummyClassifier class_distribution's sequential weighted bincount.
    do i=1,n;class_mass(encoded(i))=ieee_difference(class_mass(encoded(i)),-w(i));end do
    call require(count(class_mass/=0)>=2,'GradientBoostingClassifier weights must retain at least two classes')
    continuing=self%warm_start.and.allocated(self%estimators_);begin=0
    if(continuing)then
      begin=size(self%estimators_,1)
      call require(self%n_estimators>=begin,'gradient boosting warm start cannot reduce n_estimators')
      call require(size(self%estimators_,2)==merge(1,c,c==2),'warm boosting class channel count changed')
    end if
    if(begin<self%n_estimators.and.self%subsample==1) &
      call require(boosting_sum(w)/=0,'gradient boosting weights must have nonzero total')
    if(self%loss=='exponential') call require(c==2,'exponential boosting requires binary labels')
    self%n_features_in_=p; self%n_estimators_=self%n_estimators;self%n_classes_=c
    self%n_trees_per_iteration_=c
    if(c==2) self%n_trees_per_iteration_=1
    self%fitted_loss=self%loss; self%fitted_rate=self%learning_rate
    if(.not.continuing)then
      self%fitted_init_zero_=self%init_zero
      self%class_prior_=class_mass
      v=boosting_sum(class_mass)
      if(v/=0)then
        self%class_prior_=class_mass/v
      else
        do k=1,c
          if(class_mass(k)>0)then
            self%class_prior_(k)=ieee_value(0._dp,ieee_positive_inf)
          else if(class_mass(k)<0)then
            self%class_prior_(k)=ieee_value(0._dp,ieee_negative_inf)
          else
            self%class_prior_(k)=ieee_value(0._dp,ieee_quiet_nan)
          end if
        end do
      end if
    end if
    floor_prior=real(epsilon(1._real32),dp)
    if(allocated(self%init_)) deallocate(self%init_)
    allocate(self%init_(self%n_trees_per_iteration_)); self%init_=0
    if(.not.self%fitted_init_zero_) then
      logs=log(max(floor_prior,min(1-floor_prior,self%class_prior_)))
      if(c==2) then
        v=max(floor_prior,min(1-floor_prior,self%class_prior_(2)))
        self%init_(1)=boosting_logit(v)
        if(self%loss=='exponential') self%init_=self%init_/2
      else
        ! sklearn MultinomialLogit.link uses SciPy gmean then log(p/gmean).
        v=exp(boosting_sum(logs)/c)
        self%init_=log(max(floor_prior,min(1-floor_prior,self%class_prior_))/v)
      end if
    end if
    if(continuing)then
      raw=self%decision_function(x);rng=self%rng_
      allocate(grown(self%n_estimators,self%n_trees_per_iteration_));grown(:begin,:)=self%estimators_
      call move_alloc(grown,self%estimators_)
      call resize_boosting_values(self%train_score_,self%n_estimators)
    else
      raw=spread(self%init_,1,n);call rng%seed(self%random_state)
      if(allocated(self%estimators_))deallocate(self%estimators_)
      allocate(self%estimators_(self%n_estimators,self%n_trees_per_iteration_))
      self%train_score_=spread(0._dp,1,self%n_estimators)
    end if
    allocate(gradient(n,self%n_trees_per_iteration_),indicator(n),hessian(n),mask(n))
    self%feature_importances_=spread(0._dp,1,p)
    do_oob=self%subsample<1;n_inbag=max(1,int(self%subsample*n))
    allocate(sample_mask(n));sample_mask=.true.
    if(.not.continuing)then
      self%oob_score_=0
      if(allocated(self%oob_scores_))deallocate(self%oob_scores_,self%oob_improvement_)
    end if
    if(do_oob.or.allocated(self%oob_scores_))then
      call resize_boosting_values(self%oob_scores_,self%n_estimators)
      call resize_boosting_values(self%oob_improvement_,self%n_estimators)
      if(continuing)self%oob_score_=ieee_value(0._dp,ieee_quiet_nan)
    end if
    if(do_oob.and.begin>0)previous_loss=self%oob_scores_(begin)
    do t=begin+1,self%n_estimators
      if(do_oob)then
        sample_mask=boosting_sample_mask(rng,n,n_inbag)
        if(t==1)then
          loss_values=gbc_loss_values(raw,encoded,self%loss)
          previous_loss=boosting_mean(pack(loss_values,.not.sample_mask),pack(w,.not.sample_mask))
        end if
      end if
      fit_w=merge(w,0._dp,sample_mask)
      prob=gbc_probabilities(self,raw)
      do k=1,self%n_trees_per_iteration_
        if(c==2) then
          indicator=merge(1._dp,0._dp,encoded==2)
          if(self%loss=='exponential') then
            ! CyExponentialLoss uses a reciprocal of exp(raw), rather than
            ! exp(-raw); the one-ulp distinction can change later split ties.
            do i=1,n
              v=exp(raw(i,1))
              gradient(i,k)=ieee_difference(ieee_quotient(indicator(i),v),ieee_product(1-indicator(i),v))
            end do
          else
            ! Match CyHalfBinomialLoss's stable gradient, including rounding
            ! before the terminal-region probability is recovered below.
            do i=1,n
              if(raw(i,1)>-37._dp)then
                v=exp(-raw(i,1))
                gradient(i,k)=-((1-indicator(i))-indicator(i)*v)/(1+v)
              else
                gradient(i,k)=indicator(i)-exp(raw(i,1))
              end if
            end do
          end if
        else
          gradient(:,k)=merge(1._dp,0._dp,encoded==k)-prob(:,k)
        end if
      end do
      do k=1,self%n_trees_per_iteration_
        self%estimators_(t,k)%max_depth=self%max_depth
        self%estimators_(t,k)%min_samples_leaf=self%min_samples_leaf
        if(self%leaf_control_%fractional(self%min_samples_leaf)) &
          call self%estimators_(t,k)%set_min_samples_leaf(self%leaf_control_%value())
        self%estimators_(t,k)%min_samples_split=self%min_samples_split
        if(self%split_control_%fractional(self%min_samples_split)) &
          call self%estimators_(t,k)%set_min_samples_split(self%split_control_%value())
        self%estimators_(t,k)%max_leaf_nodes=self%max_leaf_nodes;self%estimators_(t,k)%ccp_alpha=self%ccp_alpha
        self%estimators_(t,k)%min_weight_fraction_leaf=self%min_weight_fraction_leaf
        call self%estimators_(t,k)%set_max_features(self%max_features_)
        self%estimators_(t,k)%criterion=self%criterion
        self%estimators_(t,k)%min_impurity_decrease=self%min_impurity_decrease
        self%estimators_(t,k)%random_state=self%random_state
        call self%estimators_(t,k)%fit_with_rng(x,gradient(:,k),rng,fit_w)
        leaves=self%estimators_(t,k)%apply(x)
        if(c==2) then
          if(self%loss=='exponential') then
            hessian=abs(gradient(:,k))
          else
            do i=1,n
              v=merge(1._dp,0._dp,encoded(i)==2)-gradient(i,k)
              hessian(i)=v*(1-v)
            end do
          end if
        else
          do i=1,n
            v=merge(1._dp,0._dp,encoded(i)==k)-gradient(i,k)
            hessian(i)=v*(1-v)
          end do
        end if
        factor=1
        if(c>2) factor=real(c-1,dp)/c
        do leaf=1,self%estimators_(t,k)%node_count_
          left=self%estimators_(t,k)%nodes_(leaf)%left
          if(left>0)cycle
          mask=leaves==leaf.and.sample_mask
          numerator=boosting_mean(pack(gradient(:,k),mask),pack(w,mask))
          numerator=factor*numerator
          denominator=boosting_mean(pack(hessian,mask),pack(w,mask))
          v=0
          if(abs(denominator)>=1.e-150_dp) v=numerator/denominator
          self%estimators_(t,k)%nodes_(leaf)%value=v
          self%estimators_(t,k)%nodes_(leaf)%values(1)=v
        end do
        raw(:,k)=raw(:,k)+self%learning_rate*self%estimators_(t,k)%predict(x)
      end do
      loss_values=gbc_loss_values(raw,encoded,self%loss)
      if(do_oob)then
        self%train_score_(t)=boosting_mean(pack(loss_values,sample_mask),pack(w,sample_mask))
        self%oob_scores_(t)=boosting_mean(pack(loss_values,.not.sample_mask),pack(w,.not.sample_mask))
        self%oob_improvement_(t)=ieee_difference(previous_loss,self%oob_scores_(t))
        previous_loss=self%oob_scores_(t);self%oob_score_=previous_loss
      else
        self%train_score_(t)=boosting_mean(loss_values,w)
      end if
    end do
    self%feature_importances_=0;nontrivial=0
    do t=1,size(self%estimators_,1)
      do k=1,self%n_trees_per_iteration_
        if(self%estimators_(t,k)%node_count_<=1)cycle
        self%feature_importances_=self%feature_importances_+boost_tree_importances(self%estimators_(t,k))
        nontrivial=nontrivial+1
      end do
    end do
    if(nontrivial>0)then
      self%feature_importances_=self%feature_importances_/nontrivial;v=boosting_sum(self%feature_importances_)
      do k=1,p;self%feature_importances_(k)=ieee_quotient(self%feature_importances_(k),v);end do
    end if
    self%rng_=rng;self%n_estimators_=self%n_estimators
    if(begin==self%n_estimators)then
      self%n_estimators_=begin+1
      if(allocated(self%oob_scores_))self%oob_score_=self%oob_scores_(self%n_estimators)
    end if
  end subroutine

  function gbc_loss_values(raw,encoded,loss)result(values)
    real(dp),intent(in)::raw(:,:)
    integer,intent(in)::encoded(:)
    character(*),intent(in)::loss
    real(dp)::values(size(encoded)),v
    integer::i
    do i=1,size(encoded)
      if(size(raw,2)>1)then
        values(i)=logsumexp(raw(i,:))-raw(i,encoded(i))
      else if(loss=='exponential')then
        v=exp(raw(i,1))
        values(i)=ieee_difference(ieee_quotient(merge(1._dp,0._dp,encoded(i)==2),v), &
          -ieee_product(merge(1._dp,0._dp,encoded(i)/=2),v))
      else
        values(i)=2*gbc_softplus(merge(-raw(i,1),raw(i,1),encoded(i)==2))
      end if
    end do
  end function

  function gbc_decision(self,x) result(raw)
    class(gradient_boosting_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: raw(:,:)
    integer :: t,k
    call require(allocated(self%estimators_),'GradientBoostingClassifier is not fitted')
    call check_xy(transpose(x),self%n_features_in_)
    raw=spread(self%init_,1,size(x,1))
    do t=1,size(self%estimators_,1)
      do k=1,self%n_trees_per_iteration_
        raw(:,k)=raw(:,k)+self%learning_rate*self%estimators_(t,k)%predict(x)
      end do
    end do
  end function

  function gbc_proba(self,x) result(prob)
    class(gradient_boosting_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: prob(:,:)
    prob=gbc_probabilities(self,self%decision_function(x))
  end function

  function gbc_log_proba(self,x) result(prob)
    class(gradient_boosting_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: prob(:,:),raw(:,:)
    real(dp) :: v
    integer :: i
    raw=self%decision_function(x); allocate(prob(size(x,1),size(self%classes_)))
    do i=1,size(x,1)
      if(size(self%classes_)==2) then
        v=raw(i,1)
        if(self%fitted_loss=='exponential') v=2*v
        prob(i,:)=[-gbc_softplus(v),-gbc_softplus(-v)]
      else
        prob(i,:)=raw(i,:)-logsumexp(raw(i,:))
      end if
    end do
  end function

  function gbc_predict(self,x) result(labels)
    class(gradient_boosting_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: labels(:)
    real(dp), allocatable :: raw(:,:)
    raw=self%decision_function(x)
    if(size(self%classes_)==2) then
      labels=self%classes_(merge(2,1,raw(:,1)>=0))
    else
      labels=self%classes_(maxloc(raw,dim=2))
    end if
  end function

  real(dp) function gbc_score(self,x,y,sample_weight) result(v)
    class(gradient_boosting_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    v=accuracy_score(y,self%predict(x),sample_weight=sample_weight)
  end function

  function gbc_staged_decision(self,x) result(stages)
    class(gradient_boosting_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: stages(:,:,:),raw(:,:)
    integer :: t,k
    call require(allocated(self%estimators_),'GradientBoostingClassifier is not fitted')
    call check_xy(transpose(x),self%n_features_in_)
    allocate(stages(size(x,1),self%n_trees_per_iteration_,size(self%estimators_,1)))
    raw=spread(self%init_,1,size(x,1))
    do t=1,size(self%estimators_,1)
      do k=1,self%n_trees_per_iteration_
        raw(:,k)=raw(:,k)+self%learning_rate*self%estimators_(t,k)%predict(x)
      end do
      stages(:,:,t)=raw
    end do
  end function

  function gbc_staged_proba(self,x) result(stages)
    class(gradient_boosting_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: stages(:,:,:),raw(:,:,:)
    integer :: t
    raw=self%staged_decision_function(x)
    allocate(stages(size(x,1),size(self%classes_),size(self%estimators_,1)))
    do t=1,size(self%estimators_,1)
      stages(:,:,t)=gbc_probabilities(self,raw(:,:,t))
    end do
  end function

  function gbc_staged_predict(self,x) result(stages)
    class(gradient_boosting_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: stages(:,:)
    real(dp), allocatable :: raw(:,:,:)
    integer :: t
    raw=self%staged_decision_function(x); allocate(stages(size(x,1),size(self%estimators_,1)))
    do t=1,size(self%estimators_,1)
      if(size(self%classes_)==2) then
        stages(:,t)=self%classes_(merge(2,1,raw(:,1,t)>=0))
      else
        stages(:,t)=self%classes_(maxloc(raw(:,:,t),dim=2))
      end if
    end do
  end function

  function gbc_apply(self,x) result(leaves)
    class(gradient_boosting_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: leaves(:,:,:)
    integer :: t,k
    call require(allocated(self%estimators_),'GradientBoostingClassifier is not fitted')
    call check_xy(transpose(x),self%n_features_in_)
    allocate(leaves(size(x,1),size(self%estimators_,1),self%n_trees_per_iteration_))
    do k=1,self%n_trees_per_iteration_
      do t=1,size(self%estimators_,1)
        leaves(:,t,k)=self%estimators_(t,k)%apply(x)
      end do
    end do
  end function
end module
