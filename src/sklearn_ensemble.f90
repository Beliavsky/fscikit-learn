module sklearn_ensemble
  use sklearn_sparse, only: csr_matrix
  use sklearn_class_weight, only: class_weight_map,compute_sample_weight
  use sklearn_label_binarization,only:label_set
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_ieee_math, only: ieee_difference,ieee_product,ieee_quotient,ieee_total,ieee_mean,ieee_argmax
  use sklearn_hist_gradient_boosting, only: hist_gradient_boosting_regressor,hist_gradient_boosting_classifier
  use sklearn_tree_embedding, only: random_trees_embedding
  use sklearn_stacking, only: stacking_classifier,stacking_regressor,stacking_estimator
  use sklearn_bagging, only: bagging_classifier,bagging_regressor
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value,ieee_quiet_nan,ieee_negative_inf
  use, intrinsic :: iso_fortran_env, only: real32,int64
  use sklearn_voting, only: voting_classifier,voting_regressor,voting_estimator
  use sklearn_isolation_forest, only: isolation_forest
  use sklearn_adaboost_regression, only: adaboost_regressor
  use sklearn_adaboost, only: adaboost_classifier
  use sklearn_common
  use sklearn_tree
  use sklearn_metrics, only: accuracy_score,r2_score
  implicit none
  private
  public :: hist_gradient_boosting_regressor,hist_gradient_boosting_classifier
  public :: random_trees_embedding
  public :: voting_classifier,voting_regressor,voting_estimator
  public :: bagging_classifier,bagging_regressor
  public :: stacking_classifier,stacking_regressor,stacking_estimator
  public :: isolation_forest
  public :: adaboost_classifier,adaboost_regressor
  public :: random_forest_classifier,random_forest_regressor,extra_trees_classifier,extra_trees_regressor
  public :: reset_forest_fit
  public :: gradient_boosting_regressor,gradient_boosting_classifier
  ! sklearn 1.6.1 ensemble/_gb.py and _loss/: gradient trees and Newton leaves.
  type :: gradient_boosting_classifier
    integer :: n_estimators=100,max_depth=3,min_samples_leaf=1,min_samples_split=2,max_features=0,random_state=0
    integer :: n_estimators_=0,n_trees_per_iteration_=0,n_features_in_=0
    real(dp) :: learning_rate=.1_dp,min_impurity_decrease=0
    character(16) :: loss='log_loss'
    logical :: init_zero=.false.
    integer, allocatable :: classes_(:)
    real(dp), allocatable :: init_(:),class_prior_(:),train_score_(:),feature_importances_(:)
    type(decision_tree_regressor), allocatable :: estimators_(:,:)
    real(dp), private :: fitted_rate=.1_dp
    character(16), private :: fitted_loss='log_loss'
  contains
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
  type :: random_forest_classifier
    integer :: n_estimators=100,max_depth=100,min_samples_leaf=1,min_samples_split=2
    integer :: max_features=0,random_state=0
    integer :: max_leaf_nodes=0
    integer :: max_samples=0,n_features_in_=0,n_classes_=0,n_outputs_=1
    integer,private::n_samples_fit_=0,n_samples_bootstrap_=0
    real(dp),private::max_samples_fraction_=0
    logical,private::fractional_samples_=.false.
    integer,allocatable::n_classes_multi_(:)
    type(label_set),allocatable::classes_multi_(:)
    character(16) :: criterion='gini'
    character(20) :: class_weight='none'
    type(class_weight_map), allocatable :: class_weight_values
    type(class_weight_map),allocatable::class_weight_values_multi(:)
    real(dp) :: min_impurity_decrease=0,ccp_alpha=0,min_weight_fraction_leaf=0
    logical :: bootstrap=.true.,random_thresholds=.false.,warm_start=.false.
    logical::oob_score=.false.
    logical,private::bootstrap_set_=.false.
    real(dp)::oob_score_=0
    real(dp),allocatable::oob_decision_function_(:,:)
    type(tree_probability),allocatable::oob_decision_function_multi_(:)
    integer,allocatable::oob_counts_(:)
    integer, allocatable :: classes_(:)
    type(decision_tree_classifier), allocatable :: estimators_(:)
    real(dp), allocatable :: feature_importances_(:)
  contains
    procedure,private::fit_single=>forest_classifier_fit,fit_multi=>forest_classifier_fit_multi
    generic::fit=>fit_single,fit_multi
    procedure :: decision_path=>forest_classifier_path
    procedure :: apply=>forest_classifier_apply
    procedure :: predict_log_proba=>forest_log_proba
    procedure :: predict_proba=>forest_proba
    procedure :: predict=>forest_classifier_predict
    procedure,private::score_single=>forest_classifier_score,score_multi=>forest_classifier_score_multi
    generic::score=>score_single,score_multi
    procedure::predict_multi=>forest_classifier_predict_multi
    procedure::predict_proba_multi=>forest_proba_multi,predict_log_proba_multi=>forest_log_proba_multi
    procedure::set_bootstrap=>forest_classifier_set_bootstrap,estimators_samples=>forest_classifier_samples
    procedure,private::samples_integer=>classifier_samples_integer,samples_real=>classifier_samples_real
    generic::set_max_samples=>samples_integer,samples_real
  end type
  type, extends(random_forest_classifier) :: extra_trees_classifier
  contains
    procedure,private::fit_single=>extra_classifier_fit,fit_multi=>extra_classifier_fit_multi
  end type
  type :: random_forest_regressor
    integer :: n_estimators=100,max_depth=100,min_samples_leaf=1,min_samples_split=2
    integer :: max_features=0,random_state=0
    integer :: max_leaf_nodes=0
    integer :: max_samples=0,n_features_in_=0,n_outputs_=1
    integer,private::n_samples_fit_=0,n_samples_bootstrap_=0
    real(dp),private::max_samples_fraction_=0
    logical,private::fractional_samples_=.false.
    character(16) :: criterion='squared_error'
    real(dp) :: min_impurity_decrease=0,ccp_alpha=0,min_weight_fraction_leaf=0
    logical :: bootstrap=.true.,random_thresholds=.false.,warm_start=.false.
    logical :: oob_score=.false.
    logical,private::bootstrap_set_=.false.
    real(dp)::oob_score_=0
    real(dp),allocatable::oob_prediction_(:),oob_prediction_multi_(:,:)
    integer,allocatable::oob_counts_(:)
    type(decision_tree_regressor), allocatable :: estimators_(:)
    real(dp), allocatable :: feature_importances_(:)
  contains
    procedure,private::fit_single=>forest_regressor_fit,fit_multi=>forest_regressor_fit_multi
    generic::fit=>fit_single,fit_multi
    procedure :: decision_path=>forest_regressor_path
    procedure :: apply=>forest_regressor_apply
    procedure :: predict=>forest_regressor_predict
    procedure :: predict_multi=>forest_regressor_predict_multi
    procedure,private::score_single=>forest_regressor_score,score_multi=>forest_regressor_score_multi
    generic::score=>score_single,score_multi
    procedure::set_bootstrap=>forest_regressor_set_bootstrap
    procedure::estimators_samples=>forest_regressor_samples
    procedure,private::samples_integer=>regressor_samples_integer,samples_real=>regressor_samples_real
    generic::set_max_samples=>samples_integer,samples_real
  end type
  type, extends(random_forest_regressor) :: extra_trees_regressor
  contains
    procedure,private::fit_single=>extra_regressor_fit,fit_multi=>extra_regressor_fit_multi
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
  subroutine classifier_samples_integer(self,value)
    class(random_forest_classifier),intent(inout)::self
    integer,intent(in)::value
    self%max_samples=value;self%fractional_samples_=.false.
  end subroutine
  subroutine classifier_samples_real(self,value)
    class(random_forest_classifier),intent(inout)::self
    real(dp),intent(in)::value
    self%max_samples=0;self%fractional_samples_=.true.;self%max_samples_fraction_=value
  end subroutine
  subroutine regressor_samples_integer(self,value)
    class(random_forest_regressor),intent(inout)::self
    integer,intent(in)::value
    self%max_samples=value;self%fractional_samples_=.false.
  end subroutine
  subroutine regressor_samples_real(self,value)
    class(random_forest_regressor),intent(inout)::self
    real(dp),intent(in)::value
    self%max_samples=0;self%fractional_samples_=.true.;self%max_samples_fraction_=value
  end subroutine
  ! sklearn ensemble/_forest.py: Python round uses nearest, ties to even.
  integer function forest_sample_count(n,count,fractional,fraction,bootstrap) result(ns)
    integer,intent(in)::n,count
    logical,intent(in)::fractional,bootstrap
    real(dp),intent(in)::fraction
    real(dp)::draws,remainder
    call require(count>=0.and.count<=n,'invalid forest max_samples')
    call require(bootstrap.or.(count==0.and..not.fractional),'forest max_samples requires bootstrap')
    ns=n
    if(fractional.and.count==0)then
      call require(ieee_is_finite(fraction),'forest max_samples fraction must be finite')
      call require(fraction>0.and.fraction<=1,'forest max_samples fraction must be in (0,1]')
      draws=real(n,dp)*fraction;ns=floor(draws);remainder=draws-real(ns,dp)
      if(remainder>.5_dp.or.(remainder==.5_dp.and.mod(ns,2)==1))ns=ns+1
      ns=max(1,ns)
    end if
    if(count>0)ns=count
  end function
  subroutine reset_forest_fit(model)
    class(*),intent(inout)::model
    select type(model)
    class is(random_forest_classifier)
      if(allocated(model%estimators_))deallocate(model%estimators_)
      if(allocated(model%feature_importances_))deallocate(model%feature_importances_)
      if(allocated(model%classes_))deallocate(model%classes_)
      if(allocated(model%classes_multi_))deallocate(model%classes_multi_)
      if(allocated(model%n_classes_multi_))deallocate(model%n_classes_multi_)
      if(allocated(model%oob_decision_function_))deallocate(model%oob_decision_function_)
      if(allocated(model%oob_decision_function_multi_))deallocate(model%oob_decision_function_multi_)
      if(allocated(model%oob_counts_))deallocate(model%oob_counts_)
      model%n_features_in_=0;model%n_outputs_=1;model%n_classes_=0;model%oob_score_=0
      model%n_samples_fit_=0;model%n_samples_bootstrap_=0
    class is(random_forest_regressor)
      if(allocated(model%estimators_))deallocate(model%estimators_)
      if(allocated(model%feature_importances_))deallocate(model%feature_importances_)
      if(allocated(model%oob_prediction_))deallocate(model%oob_prediction_)
      if(allocated(model%oob_prediction_multi_))deallocate(model%oob_prediction_multi_)
      if(allocated(model%oob_counts_))deallocate(model%oob_counts_)
      model%n_features_in_=0;model%n_outputs_=1;model%oob_score_=0
      model%n_samples_fit_=0;model%n_samples_bootstrap_=0
    end select
  end subroutine
  subroutine check_forest_controls(depth,split,leaf,budget,features,seed,decrease,alpha,fraction,p)
    integer,intent(in)::depth,split,leaf,budget,features,seed,p
    real(dp),intent(in)::decrease,alpha,fraction
    call require(depth>=0.and.split>=2.and.leaf>=1,'invalid forest tree size controls')
    call require(budget>=0.and.budget/=1,'forest max_leaf_nodes must be zero or at least two')
    call require(features>=0.and.features<=p,'invalid forest max_features')
    call require(seed>=0,'forest random_state must be nonnegative')
    call require(ieee_is_finite(decrease),'forest impurity decrease must be finite')
    call require(decrease>=0,'forest impurity decrease must be nonnegative')
    call require(ieee_is_finite(alpha),'forest ccp_alpha must be finite')
    call require(alpha>=0,'forest ccp_alpha must be nonnegative')
    call require(ieee_is_finite(fraction),'forest minimum weighted leaf fraction must be finite')
    call require(fraction>=0.and.fraction<=.5_dp,'forest minimum weighted leaf fraction must be in [0,0.5]')
  end subroutine
  subroutine forest_classifier_fit(self,x,y,sample_weight)
    class(random_forest_classifier),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    integer,intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    call forest_classifier_fit_body(self,x,reshape(y,[size(y),1]),sample_weight)
  end subroutine
  subroutine forest_classifier_fit_multi(self,x,y,sample_weight)
    class(random_forest_classifier),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    integer,intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    call forest_classifier_fit_body(self,x,y,sample_weight)
  end subroutine
  subroutine forest_classifier_fit_body(self,x,y,sample_weight)
    class(random_forest_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    type(decision_tree_classifier),allocatable::grown(:)
    logical::recompute_oob
    integer::old
    type(legacy_random_state) :: seeds,draws
    real(dp) :: w(size(y,1)),tree_weights(size(y,1)),normalizer
    real(dp), allocatable :: factors(:)
    integer :: i,j,k,c,n,p,q,features,seed,ns,nontrivial,at,counts(size(y,1)),encoded(size(y,1),size(y,2))
    integer, allocatable :: indices(:),unsampled(:),oob_labels(:,:)
    type(tree_probability),allocatable::oob(:),prob(:)
    real(dp),allocatable::query(:,:)
    call check_xy(x,size(y,1));call require(self%n_estimators>0,'forest needs positive n_estimators')
    ns=forest_sample_count(size(y,1),self%max_samples,self%fractional_samples_,self%max_samples_fraction_,self%bootstrap)
    call require(any(self%class_weight==[character(20)::'none','balanced','balanced_subsample']),'invalid forest class weight mode')
    call require(.not.(allocated(self%class_weight_values).and.self%class_weight/='none'), &
      'forest class weight map conflicts with preset')
    q=size(y,2);call require(q>0,'forest targets require at least one output')
    old=0
    if(self%warm_start.and.allocated(self%estimators_))old=size(self%estimators_)
    call require(self%n_estimators>=old,'warm-start n_estimators cannot decrease')
    if(old>0)then
      call require(size(x,2)==self%n_features_in_,'warm-start forest feature count must remain unchanged')
      call require(q==self%n_outputs_,'warm-start forest output count must remain unchanged')
    end if
    recompute_oob=self%oob_score.and.(self%n_estimators>old.or..not.allocated(self%oob_counts_))
    call check_forest_controls(self%max_depth,self%min_samples_split,self%min_samples_leaf,self%max_leaf_nodes, &
      self%max_features,self%random_state,self%min_impurity_decrease,self%ccp_alpha,self%min_weight_fraction_leaf,size(x,2))
    call require(maxval(abs(x))<=real(huge(0._real32),dp),'forest training features exceed float32 range')
    call require(any(self%criterion==['gini    ','entropy ','log_loss']),'unknown forest classification criterion')
    call require(self%bootstrap.or..not.self%oob_score,'forest OOB scoring requires bootstrap')
    call require(.not.(allocated(self%class_weight_values).and.allocated(self%class_weight_values_multi)), &
      'forest scalar and per-output class weight maps conflict')
    call require(.not.(allocated(self%class_weight_values_multi).and.self%class_weight/='none'), &
      'forest per-output class weight maps conflict with preset')
    if(allocated(self%class_weight_values))call require(q==1,'joint forest class weights require per-output maps')
    if(allocated(self%classes_multi_))deallocate(self%classes_multi_,self%n_classes_multi_)
    allocate(self%classes_multi_(q),self%n_classes_multi_(q));self%n_outputs_=q
    do k=1,q
      self%classes_multi_(k)%values=unique_labels(y(:,k));self%n_classes_multi_(k)=size(self%classes_multi_(k)%values)
      do j=1,size(y,1);encoded(j,k)=label_index(self%classes_multi_(k)%values,y(j,k))-1;end do
    end do
    self%classes_=self%classes_multi_(1)%values;self%n_classes_=self%n_classes_multi_(1)
    if(recompute_oob)then
      self%oob_score_=0
    if(allocated(self%oob_decision_function_))deallocate(self%oob_decision_function_)
    if(allocated(self%oob_decision_function_multi_))deallocate(self%oob_decision_function_multi_)
    if(allocated(self%oob_counts_))deallocate(self%oob_counts_)
    end if
    if(recompute_oob.and.q>1)then
      call require(all(self%n_classes_multi_<=2),'multiclass-multioutput forest OOB scoring is unsupported')
      call require(all(self%n_classes_multi_==self%n_classes_),'forest OOB class widths must agree')
    end if
    w=1
    if(present(sample_weight)) then
      call require(size(sample_weight)==size(y,1),'forest sample weight length mismatch')
      call require(all(ieee_is_finite(sample_weight)),'forest sample weights must be finite')
      w=sample_weight
    end if
    if(allocated(self%class_weight_values)) then
      factors=compute_sample_weight(y(:,1),class_weight=self%class_weight_values)
    else if(allocated(self%class_weight_values_multi)) then
      factors=compute_sample_weight(y,class_weight=self%class_weight_values_multi)
    else if(self%class_weight=='balanced'.or.(self%class_weight=='balanced_subsample'.and..not.self%bootstrap)) then
      factors=compute_sample_weight(y,mode='balanced')
    end if
    if(allocated(factors)) then
      do j=1,size(y,1)
        w(j)=ieee_product(w(j),factors(j))
      end do
    end if
    n=size(y,1);p=size(x,2);self%n_features_in_=p
    self%n_samples_fit_=n;self%n_samples_bootstrap_=ns
    if(recompute_oob)then
      allocate(oob(q),self%oob_counts_(n));self%oob_counts_=0
      do k=1,q;allocate(oob(k)%values(n,self%n_classes_multi_(k)));oob(k)%values=0;end do
    end if
    allocate(indices(ns));call seeds%seed(self%random_state)
    features=self%max_features;if(features==0) features=max(1,int(sqrt(real(p,dp))))
    if(old==0)then
      if(allocated(self%estimators_))deallocate(self%estimators_)
      allocate(self%estimators_(self%n_estimators))
    else if(self%n_estimators>old)then
      allocate(grown(self%n_estimators));grown(:old)=self%estimators_
      call move_alloc(grown,self%estimators_)
    end if
    self%feature_importances_=spread(0._dp,1,p);nontrivial=0
    do i=1,self%n_estimators
      seed=seeds%integer_index(2147483647)-1
      if(i>old)then
      tree_weights=w
      if(self%bootstrap) then
        call draws%seed(seed);counts=0
        do j=1,ns
          at=draws%integer_index(n);counts(at)=counts(at)+1;indices(j)=at
        end do
        do j=1,n
          tree_weights(j)=ieee_product(w(j),real(counts(j),dp))
        end do
        if(self%class_weight=='balanced_subsample') then
          factors=compute_sample_weight(encoded,mode='balanced',indices=indices)
          do j=1,n
            tree_weights(j)=ieee_product(tree_weights(j),factors(j))
          end do
        end if
      end if
      self%estimators_(i)%criterion=self%criterion
      self%estimators_(i)%max_depth=self%max_depth
      self%estimators_(i)%min_samples_leaf=self%min_samples_leaf
      self%estimators_(i)%min_samples_split=self%min_samples_split
      self%estimators_(i)%max_features=features
      self%estimators_(i)%max_leaf_nodes=self%max_leaf_nodes
      self%estimators_(i)%min_weight_fraction_leaf=self%min_weight_fraction_leaf
      self%estimators_(i)%ccp_alpha=self%ccp_alpha
      self%estimators_(i)%min_impurity_decrease=self%min_impurity_decrease
      self%estimators_(i)%random_thresholds=self%random_thresholds
      self%estimators_(i)%random_state=seed
      if(q==1)then
        call self%estimators_(i)%fit(x,encoded(:,1),sample_weight=tree_weights)
      else
        call self%estimators_(i)%fit(x,encoded,sample_weight=tree_weights)
      end if
      end if
      if(recompute_oob)then
        if(i<=old)then
          call draws%seed(self%estimators_(i)%random_state);counts=0
          do j=1,ns
            at=draws%integer_index(n);counts(at)=counts(at)+1
          end do
        end if
        unsampled=pack([(j,j=1,n)],counts==0)
        if(size(unsampled)>0)then
          query=x(unsampled,:);prob=self%estimators_(i)%predict_proba_multi(query)
          do j=1,size(unsampled)
            at=unsampled(j);self%oob_counts_(at)=self%oob_counts_(at)+1
            do k=1,q
              call require(size(prob(k)%values,2)==1.or.size(prob(k)%values,2)==self%n_classes_multi_(k), &
                'warm-start forest class probabilities cannot broadcast')
              do c=1,self%n_classes_multi_(k)
                oob(k)%values(at,c)=ieee_difference(oob(k)%values(at,c),-prob(k)%values(j,min(c,size(prob(k)%values,2))))
              end do
            end do
          end do
        end if
      end if
      if(self%estimators_(i)%node_count_>1) then
        nontrivial=nontrivial+1
        do j=1,p
          self%feature_importances_(j)=ieee_difference(self%feature_importances_(j), &
            -self%estimators_(i)%feature_importances_(j))
        end do
      end if
    end do
    if(nontrivial>0)then
    do j=1,p
      self%feature_importances_(j)=ieee_quotient(self%feature_importances_(j),real(nontrivial,dp))
    end do
    normalizer=ieee_total(self%feature_importances_)
    do j=1,p
      self%feature_importances_(j)=ieee_quotient(self%feature_importances_(j),normalizer)
    end do
    end if
    if(recompute_oob)then
      allocate(oob_labels(n,q))
      do k=1,q
        do j=1,n
          do c=1,self%n_classes_multi_(k)
            oob(k)%values(j,c)=ieee_quotient(oob(k)%values(j,c),real(max(1,self%oob_counts_(j)),dp))
          end do
          oob_labels(j,k)=ieee_argmax(oob(k)%values(j,:))-1
        end do
      end do
      if(q==1)then
        self%oob_decision_function_=oob(1)%values
        self%oob_score_=accuracy_score(encoded(:,1),oob_labels(:,1))
      else
        self%oob_decision_function_multi_=oob
        self%oob_score_=accuracy_score(encoded,oob_labels)
      end if
    end if
  end subroutine
  subroutine extra_classifier_fit(self,x,y,sample_weight)
    class(extra_trees_classifier),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    integer,intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    if(.not.self%bootstrap_set_)self%bootstrap=.false.
    self%random_thresholds=.true.
    call forest_classifier_fit(self,x,y,sample_weight)
  end subroutine
  subroutine extra_classifier_fit_multi(self,x,y,sample_weight)
    class(extra_trees_classifier),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    integer,intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    if(.not.self%bootstrap_set_)self%bootstrap=.false.
    self%random_thresholds=.true.
    call forest_classifier_fit_multi(self,x,y,sample_weight)
  end subroutine
  subroutine forest_classifier_set_bootstrap(self,value)
    class(random_forest_classifier),intent(inout)::self
    logical,intent(in)::value
    self%bootstrap=value;self%bootstrap_set_=.true.
  end subroutine
  function forest_classifier_samples(self)result(indices)
    class(random_forest_classifier),intent(in)::self
    integer,allocatable::indices(:,:)
    type(legacy_random_state)::draws
    integer::i,j,n
    call require(allocated(self%estimators_),'forest is not fitted')
    n=self%n_samples_fit_
    if(self%bootstrap)then
      allocate(indices(self%n_samples_bootstrap_,size(self%estimators_)))
      do i=1,size(self%estimators_)
        call draws%seed(self%estimators_(i)%random_state)
        do j=1,self%n_samples_bootstrap_;indices(j,i)=draws%integer_index(n);end do
      end do
    else
      allocate(indices(n,size(self%estimators_)))
      do i=1,size(self%estimators_);indices(:,i)=[(j,j=1,n)];end do
    end if
  end function
  function forest_proba_multi(self,x)result(prob)
    class(random_forest_classifier),intent(in)::self
    real(dp),intent(in)::x(:,:)
    type(tree_probability),allocatable::prob(:),part(:)
    integer::t,k,i,c
    call require(allocated(self%estimators_),'forest is not fitted')
    allocate(prob(self%n_outputs_))
    do k=1,self%n_outputs_
      allocate(prob(k)%values(size(x,1),self%n_classes_multi_(k)));prob(k)%values=0
    end do
    do t=1,size(self%estimators_)
      part=self%estimators_(t)%predict_proba_multi(x)
      do k=1,self%n_outputs_
        call require(size(part(k)%values,2)==1.or.size(part(k)%values,2)==self%n_classes_multi_(k), &
          'warm-start forest class probabilities cannot broadcast')
        do c=1,self%n_classes_multi_(k)
          do i=1,size(x,1)
            prob(k)%values(i,c)=ieee_difference(prob(k)%values(i,c),-part(k)%values(i,min(c,size(part(k)%values,2))))
          end do
        end do
      end do
    end do
    do k=1,self%n_outputs_
      do c=1,self%n_classes_multi_(k)
        do i=1,size(x,1)
          prob(k)%values(i,c)=ieee_quotient(prob(k)%values(i,c),real(size(self%estimators_),dp))
        end do
      end do
    end do
  end function
  function forest_log_proba_multi(self,x)result(prob)
    class(random_forest_classifier),intent(in)::self
    real(dp),intent(in)::x(:,:)
    type(tree_probability),allocatable::prob(:)
    integer::k,i,c
    prob=self%predict_proba_multi(x)
    do k=1,size(prob)
      do c=1,size(prob(k)%values,2)
        do i=1,size(x,1)
          if(ieee_is_nan(prob(k)%values(i,c)))cycle
          if(prob(k)%values(i,c)==0)then
            prob(k)%values(i,c)=ieee_value(0._dp,ieee_negative_inf)
          else if(prob(k)%values(i,c)>0)then
            prob(k)%values(i,c)=log(prob(k)%values(i,c))
          else
            prob(k)%values(i,c)=ieee_value(0._dp,ieee_quiet_nan)
          end if
        end do
      end do
    end do
  end function
  function forest_classifier_predict_multi(self,x)result(y)
    class(random_forest_classifier),intent(in)::self
    real(dp),intent(in)::x(:,:)
    integer,allocatable::y(:,:)
    type(tree_probability),allocatable::prob(:)
    integer::i,k
    prob=self%predict_proba_multi(x);allocate(y(size(x,1),self%n_outputs_))
    do k=1,self%n_outputs_
      do i=1,size(x,1);y(i,k)=self%classes_multi_(k)%values(ieee_argmax(prob(k)%values(i,:)));end do
    end do
  end function
  real(dp)function forest_classifier_score_multi(self,x,y,sample_weight)result(value)
    class(random_forest_classifier),intent(in)::self
    real(dp),intent(in)::x(:,:)
    integer,intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    value=accuracy_score(y,self%predict_multi(x),sample_weight=sample_weight)
  end function
  function forest_proba(self,x) result(prob)
    class(random_forest_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: prob(:,:),tree_prob(:,:)
    integer :: i,j,c
    call require(self%n_outputs_==1,'joint forest probabilities require predict_proba_multi')
    call require(allocated(self%estimators_),'forest is not fitted')
    allocate(prob(size(x,1),self%n_classes_));prob=0
    do i=1,size(self%estimators_)
      tree_prob=self%estimators_(i)%predict_proba(x)
      call require(size(tree_prob,2)==1.or.size(tree_prob,2)==self%n_classes_, &
        'warm-start forest class probabilities cannot broadcast')
      do c=1,self%n_classes_
        do j=1,size(x,1)
          prob(j,c)=ieee_difference(prob(j,c),-tree_prob(j,min(c,size(tree_prob,2))))
        end do
      end do
    end do
    do c=1,self%n_classes_
      do j=1,size(x,1)
        prob(j,c)=ieee_quotient(prob(j,c),real(size(self%estimators_),dp))
      end do
    end do
  end function
  function forest_log_proba(self,x) result(prob)
    class(random_forest_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: prob(:,:)
    integer :: i,j
    prob=self%predict_proba(x)
    do j=1,size(prob,2)
      do i=1,size(prob,1)
        if(ieee_is_nan(prob(i,j))) cycle
        if(prob(i,j)==0) then
          prob(i,j)=ieee_value(0._dp,ieee_negative_inf)
        else if(prob(i,j)>0) then
          prob(i,j)=log(prob(i,j))
        else
          prob(i,j)=ieee_value(0._dp,ieee_quiet_nan)
        end if
      end do
    end do
  end function
  function stack_decision_paths(parts,n_nodes_ptr) result(indicator)
    type(csr_matrix), intent(in) :: parts(:)
    integer, allocatable, optional, intent(out) :: n_nodes_ptr(:)
    type(csr_matrix) :: indicator
    integer :: offsets(size(parts)+1),i,t,j,k
    integer(int64) :: columns,stored
    columns=0;stored=0;offsets(1)=0
    do t=1,size(parts)
      columns=columns+int(parts(t)%n_cols,int64);stored=stored+size(parts(t)%data,kind=int64)
      call require(columns<=int(huge(1),int64),'forest decision-path columns exceed integer range')
      call require(stored<int(huge(1),int64),'forest decision-path stored count exceeds integer range')
      offsets(t+1)=int(columns)
    end do
    indicator%n_rows=parts(1)%n_rows;indicator%n_cols=int(columns)
    allocate(indicator%indptr(indicator%n_rows+1),indicator%indices(int(stored)),indicator%data(int(stored)))
    indicator%data=1;indicator%indptr(1)=1;k=1
    do i=1,indicator%n_rows
      do t=1,size(parts)
        do j=parts(t)%indptr(i),parts(t)%indptr(i+1)-1
          indicator%indices(k)=offsets(t)+parts(t)%indices(j);k=k+1
        end do
      end do
      indicator%indptr(i+1)=k
    end do
    if(present(n_nodes_ptr)) n_nodes_ptr=offsets
    call indicator%validate()
  end function
  function forest_classifier_path(self,x,n_nodes_ptr) result(indicator)
    class(random_forest_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable, optional, intent(out) :: n_nodes_ptr(:)
    type(csr_matrix) :: indicator
    type(csr_matrix), allocatable :: parts(:)
    integer :: t
    call require(allocated(self%estimators_),'forest is not fitted')
    allocate(parts(size(self%estimators_)))
    do t=1,size(parts)
      parts(t)=self%estimators_(t)%decision_path(x)
    end do
    indicator=stack_decision_paths(parts,n_nodes_ptr)
  end function
  function forest_regressor_path(self,x,n_nodes_ptr) result(indicator)
    class(random_forest_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable, optional, intent(out) :: n_nodes_ptr(:)
    type(csr_matrix) :: indicator
    type(csr_matrix), allocatable :: parts(:)
    integer :: t
    call require(allocated(self%estimators_),'forest is not fitted')
    allocate(parts(size(self%estimators_)))
    do t=1,size(parts)
      parts(t)=self%estimators_(t)%decision_path(x)
    end do
    indicator=stack_decision_paths(parts,n_nodes_ptr)
  end function
  function forest_classifier_apply(self,x) result(leaves)
    class(random_forest_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: leaves(:,:)
    integer :: i
    call require(allocated(self%estimators_),'forest is not fitted')
    allocate(leaves(size(x,1),size(self%estimators_)))
    do i=1,size(self%estimators_)
      leaves(:,i)=self%estimators_(i)%apply(x)
    end do
  end function
  function forest_classifier_predict(self,x) result(y)
    class(random_forest_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: y(size(x,1)),i
    real(dp), allocatable :: prob(:,:)
    call require(self%n_outputs_==1,'joint forest prediction requires predict_multi')
    prob=self%predict_proba(x)
    do i=1,size(y)
      y(i)=self%classes_(ieee_argmax(prob(i,:)))
    end do
  end function
  real(dp) function forest_classifier_score(self,x,y,sample_weight) result(v)
    class(random_forest_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer :: predicted(size(x,1))
    real(dp) :: mass
    predicted=self%predict(x)
    call require(size(y)==size(predicted),'forest score target length mismatch')
    if(present(sample_weight)) then
      call require(size(sample_weight)==size(y),'forest score weight length mismatch')
      call require(all(ieee_is_finite(sample_weight)),'forest score weights must be finite')
      mass=ieee_total(sample_weight)
      if(.not.ieee_is_nan(mass)) call require(mass/=0,'forest score weights must have a nonzero sum')
    end if
    v=ieee_mean(merge(1._dp,0._dp,y==predicted),sample_weight)
  end function
  subroutine forest_regressor_fit(self,x,y,sample_weight)
    class(random_forest_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    call forest_regressor_fit_body(self,x,y,sample_weight)
  end subroutine
  subroutine forest_regressor_fit_multi(self,x,y,sample_weight)
    class(random_forest_regressor),intent(inout)::self
    real(dp),intent(in)::x(:,:),y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    call check_xy(x,size(y,1));call require(size(y,2)>0,'forest targets require at least one output')
    call forest_regressor_fit_body(self,x,y(:,1),sample_weight,targets=y)
  end subroutine
  subroutine forest_regressor_fit_body(self,x,y,sample_weight,targets)
    class(random_forest_regressor),intent(inout)::self
    real(dp),intent(in)::x(:,:),y(:)
    real(dp),optional,intent(in)::sample_weight(:),targets(:,:)
    type(decision_tree_regressor),allocatable::grown(:)
    logical::recompute_oob
    integer::old
    type(legacy_random_state) :: seeds,draws
    real(dp) :: w(size(y)),tree_weights(size(y)),normalizer
    real(dp),allocatable::oob(:,:),prediction(:,:),query(:,:)
    integer,allocatable::unsampled(:)
    integer :: i,j,k,n,p,seed,ns,nontrivial,at,counts(size(y)),q
    call check_xy(x,size(y));call check_x(reshape(y,[size(y),1]))
    call require(self%n_estimators>0,'forest needs positive n_estimators')
    ns=forest_sample_count(size(y),self%max_samples,self%fractional_samples_,self%max_samples_fraction_,self%bootstrap)
    call require(self%bootstrap.or..not.self%oob_score,'forest OOB scoring requires bootstrap')
    q=1
    if(present(targets))then
      call check_x(targets);call require(size(targets,1)==size(y),'forest target row mismatch')
      q=size(targets,2)
    end if
    old=0
    if(self%warm_start.and.allocated(self%estimators_))old=size(self%estimators_)
    call require(self%n_estimators>=old,'warm-start n_estimators cannot decrease')
    if(old>0)then
      call require(size(x,2)==self%n_features_in_,'warm-start forest feature count must remain unchanged')
      call require(q==self%n_outputs_,'warm-start forest output count must remain unchanged')
    end if
    recompute_oob=self%oob_score.and.(self%n_estimators>old.or..not.allocated(self%oob_counts_))
    call check_forest_controls(self%max_depth,self%min_samples_split,self%min_samples_leaf,self%max_leaf_nodes, &
      self%max_features,self%random_state,self%min_impurity_decrease,self%ccp_alpha,self%min_weight_fraction_leaf,size(x,2))
    call require(maxval(abs(x))<=real(huge(0._real32),dp),'forest training features exceed float32 range')
    call require(any(self%criterion==[character(16)::'squared_error','friedman_mse','absolute_error','poisson']), &
      'unknown forest regression criterion')
    if(self%criterion=='poisson')then
      if(present(targets))then
        call require(all(targets>=0),'Poisson forest targets must be nonnegative')
        call require(ieee_total(reshape(targets,[size(targets)]))>0,'Poisson forest target sum must be positive')
      else
        call require(all(y>=0),'Poisson forest targets must be nonnegative')
        call require(ieee_total(y)>0,'Poisson forest target sum must be positive')
      end if
    end if
    self%n_outputs_=q
    if(recompute_oob)then
      self%oob_score_=0
    if(allocated(self%oob_prediction_))deallocate(self%oob_prediction_)
    if(allocated(self%oob_prediction_multi_))deallocate(self%oob_prediction_multi_)
    if(allocated(self%oob_counts_))deallocate(self%oob_counts_)
    end if
    w=1
    if(present(sample_weight)) then
      call require(size(sample_weight)==size(y),'forest sample weight length mismatch')
      call require(all(ieee_is_finite(sample_weight)),'forest sample weights must be finite')
      w=sample_weight
    end if
    n=size(y);p=size(x,2);self%n_features_in_=p
    self%n_samples_fit_=n;self%n_samples_bootstrap_=ns
    if(recompute_oob)then
      allocate(oob(n,q),self%oob_counts_(n));oob=0;self%oob_counts_=0
    end if
    call seeds%seed(self%random_state)
    if(old==0)then
      if(allocated(self%estimators_))deallocate(self%estimators_)
      allocate(self%estimators_(self%n_estimators))
    else if(self%n_estimators>old)then
      allocate(grown(self%n_estimators));grown(:old)=self%estimators_
      call move_alloc(grown,self%estimators_)
    end if
    self%feature_importances_=spread(0._dp,1,p);nontrivial=0
    do i=1,self%n_estimators
      seed=seeds%integer_index(2147483647)-1
      if(i>old)then
      tree_weights=w
      if(self%bootstrap) then
        call draws%seed(seed);counts=0
        do j=1,ns
          at=draws%integer_index(n);counts(at)=counts(at)+1
        end do
        do j=1,n
          tree_weights(j)=ieee_product(w(j),real(counts(j),dp))
        end do
      end if
      self%estimators_(i)%criterion=self%criterion
      self%estimators_(i)%max_depth=self%max_depth
      self%estimators_(i)%min_samples_leaf=self%min_samples_leaf
      self%estimators_(i)%min_samples_split=self%min_samples_split
      self%estimators_(i)%max_features=self%max_features
      self%estimators_(i)%max_leaf_nodes=self%max_leaf_nodes
      self%estimators_(i)%min_weight_fraction_leaf=self%min_weight_fraction_leaf
      self%estimators_(i)%ccp_alpha=self%ccp_alpha
      self%estimators_(i)%min_impurity_decrease=self%min_impurity_decrease
      self%estimators_(i)%random_thresholds=self%random_thresholds
      self%estimators_(i)%random_state=seed
      if(present(targets))then
        call self%estimators_(i)%fit(x,targets,sample_weight=tree_weights)
      else
        call self%estimators_(i)%fit(x,y,sample_weight=tree_weights)
      end if
      end if
      if(recompute_oob)then
        if(i<=old)then
          call draws%seed(self%estimators_(i)%random_state);counts=0
          do j=1,ns
            at=draws%integer_index(n);counts(at)=counts(at)+1
          end do
        end if
        unsampled=pack([(j,j=1,n)],counts==0)
        if(size(unsampled)>0)then
          query=x(unsampled,:);prediction=self%estimators_(i)%predict_multi(query)
          do j=1,size(unsampled)
            at=unsampled(j);self%oob_counts_(at)=self%oob_counts_(at)+1
            do k=1,q;oob(at,k)=ieee_difference(oob(at,k),-prediction(j,k));end do
          end do
        end if
      end if
      if(self%estimators_(i)%node_count_>1) then
        nontrivial=nontrivial+1
        do j=1,p
          self%feature_importances_(j)=ieee_difference(self%feature_importances_(j), &
            -self%estimators_(i)%feature_importances_(j))
        end do
      end if
    end do
    if(nontrivial>0)then
      do j=1,p
        self%feature_importances_(j)=ieee_quotient(self%feature_importances_(j),real(nontrivial,dp))
      end do
      normalizer=ieee_total(self%feature_importances_)
      do j=1,p
        self%feature_importances_(j)=ieee_quotient(self%feature_importances_(j),normalizer)
      end do
    end if
    if(recompute_oob)then
      do j=1,n
        do k=1,q;oob(j,k)=ieee_quotient(oob(j,k),real(max(1,self%oob_counts_(j)),dp));end do
      end do
      if(q==1)then
        self%oob_prediction_=oob(:,1);self%oob_score_=r2_score(y,self%oob_prediction_)
      else
        self%oob_prediction_multi_=oob;self%oob_score_=r2_score(targets,oob)
      end if
    end if
  end subroutine
  subroutine extra_regressor_fit(self,x,y,sample_weight)
    class(extra_trees_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    if(.not.self%bootstrap_set_)self%bootstrap=.false.
    self%random_thresholds=.true.
    call forest_regressor_fit(self,x,y,sample_weight)
  end subroutine
  subroutine extra_regressor_fit_multi(self,x,y,sample_weight)
    class(extra_trees_regressor),intent(inout)::self
    real(dp),intent(in)::x(:,:),y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    if(.not.self%bootstrap_set_)self%bootstrap=.false.
    self%random_thresholds=.true.
    call forest_regressor_fit_multi(self,x,y,sample_weight)
  end subroutine
  subroutine forest_regressor_set_bootstrap(self,value)
    class(random_forest_regressor),intent(inout)::self
    logical,intent(in)::value
    self%bootstrap=value;self%bootstrap_set_=.true.
  end subroutine
  function forest_regressor_samples(self)result(indices)
    class(random_forest_regressor),intent(in)::self
    integer,allocatable::indices(:,:)
    type(legacy_random_state)::draws
    integer::i,j,n
    call require(allocated(self%estimators_),'forest is not fitted')
    n=self%n_samples_fit_
    if(self%bootstrap)then
      allocate(indices(self%n_samples_bootstrap_,size(self%estimators_)))
      do i=1,size(self%estimators_)
        call draws%seed(self%estimators_(i)%random_state)
        do j=1,self%n_samples_bootstrap_;indices(j,i)=draws%integer_index(n);end do
      end do
    else
      allocate(indices(n,size(self%estimators_)))
      do i=1,size(self%estimators_);indices(:,i)=[(j,j=1,n)];end do
    end if
  end function
  function forest_regressor_predict(self,x) result(y)
    class(random_forest_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: y(size(x,1)),values(size(x,1))
    integer :: i,j
    call require(self%n_outputs_==1,'joint forest prediction requires predict_multi')
    call require(allocated(self%estimators_),'forest is not fitted');y=0
    do i=1,size(self%estimators_)
      values=self%estimators_(i)%predict(x)
      do j=1,size(y)
        y(j)=ieee_difference(y(j),-values(j))
      end do
    end do
    do j=1,size(y)
      y(j)=ieee_quotient(y(j),real(size(self%estimators_),dp))
    end do
  end function
  function forest_regressor_predict_multi(self,x)result(y)
    class(random_forest_regressor),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),allocatable::y(:,:),values(:,:)
    integer::i,j,k
    call require(allocated(self%estimators_),'forest is not fitted')
    allocate(y(size(x,1),self%n_outputs_));y=0
    do i=1,size(self%estimators_)
      values=self%estimators_(i)%predict_multi(x)
      do k=1,self%n_outputs_
        do j=1,size(x,1);y(j,k)=ieee_difference(y(j,k),-values(j,k));end do
      end do
    end do
    do k=1,self%n_outputs_
      do j=1,size(x,1);y(j,k)=ieee_quotient(y(j,k),real(size(self%estimators_),dp));end do
    end do
  end function
  real(dp)function forest_regressor_score_multi(self,x,y,sample_weight)result(value)
    class(random_forest_regressor),intent(in)::self
    real(dp),intent(in)::x(:,:),y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    value=r2_score(y,self%predict_multi(x),sample_weight=sample_weight)
  end function
  function forest_regressor_apply(self,x) result(leaves)
    class(random_forest_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: leaves(:,:)
    integer :: i
    call require(allocated(self%estimators_),'forest is not fitted')
    allocate(leaves(size(x,1),size(self%estimators_)))
    do i=1,size(self%estimators_)
      leaves(:,i)=self%estimators_(i)%apply(x)
    end do
  end function
  real(dp) function forest_regressor_score(self,x,y,sample_weight) result(v)
    class(random_forest_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    real(dp), allocatable :: predicted(:),numerators(:),denominators(:)
    real(dp) :: center,num,den,mass
    integer :: i
    predicted=self%predict(x)
    call require(size(y)==size(predicted),'forest score target length mismatch')
    call require(all(ieee_is_finite(y)),'forest regression score targets must be finite')
    call require(all(ieee_is_finite(predicted)),'forest regression score predictions must be finite')
    if(present(sample_weight)) then
      call require(size(sample_weight)==size(y),'forest score weight length mismatch')
      call require(all(ieee_is_finite(sample_weight)),'forest score weights must be finite')
    end if
    if(size(y)<2) then
      v=ieee_value(0._dp,ieee_quiet_nan); return
    end if
    if(present(sample_weight)) then
      mass=ieee_total(sample_weight)
      if(.not.ieee_is_nan(mass)) call require(mass/=0,'forest score weights must have a nonzero sum')
    end if
    center=ieee_mean(y,sample_weight); allocate(numerators(size(y)),denominators(size(y)))
    do i=1,size(y)
      num=ieee_difference(y(i),predicted(i)); numerators(i)=ieee_product(num,num)
      den=ieee_difference(y(i),center); denominators(i)=ieee_product(den,den)
    end do
    num=ieee_mean(numerators,sample_weight); den=ieee_mean(denominators,sample_weight)
    v=ieee_value(0._dp,ieee_quiet_nan)
    if(ieee_is_nan(num).or.ieee_is_nan(den)) return
    v=1
    if(num/=0) then
      v=0
      if(den/=0) v=ieee_difference(1._dp,ieee_quotient(num,den))
    end if
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

  subroutine gbc_fit(self,x,y)
    class(gradient_boosting_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), allocatable :: raw(:,:),prob(:,:),gradient(:,:),indicator(:),hessian(:),logs(:)
    integer, allocatable :: encoded(:),leaves(:)
    logical, allocatable :: mask(:)
    real(dp) :: floor_prior,numerator,denominator,factor,gain,stage_loss,v
    integer :: n,p,c,k,t,i,leaf,left,right,feature,mass
    call check_xy(x,size(y)); n=size(y); p=size(x,2)
    call require(self%n_estimators>=1,'GradientBoostingClassifier n_estimators must be positive')
    call require(ieee_is_finite(self%learning_rate),'nonfinite GradientBoostingClassifier learning_rate')
    call require(self%learning_rate>0,'GradientBoostingClassifier learning_rate must be positive')
    call require(ieee_is_finite(self%min_impurity_decrease),'nonfinite GradientBoostingClassifier impurity control')
    call require(self%min_impurity_decrease>=0,'negative GradientBoostingClassifier impurity decrease')
    call require(self%max_depth>=1.and.self%min_samples_leaf>=1.and.self%min_samples_split>=2, &
      'invalid GradientBoostingClassifier tree controls')
    call require(self%max_features>=0.and.self%max_features<=p,'invalid GradientBoostingClassifier max_features')
    call require(self%loss=='log_loss'.or.self%loss=='exponential','invalid GradientBoostingClassifier loss')
    self%classes_=unique_labels(y); c=size(self%classes_)
    call require(c>=2,'GradientBoostingClassifier requires at least two classes')
    if(self%loss=='exponential') call require(c==2,'exponential boosting requires binary labels')
    self%n_features_in_=p; self%n_estimators_=self%n_estimators
    self%n_trees_per_iteration_=c
    if(c==2) self%n_trees_per_iteration_=1
    self%fitted_loss=self%loss; self%fitted_rate=self%learning_rate
    encoded=[(label_index(self%classes_,y(i)),i=1,n)]
    self%class_prior_=[(real(count(encoded==k),dp)/n,k=1,c)]
    floor_prior=real(epsilon(1._real32),dp)
    logs=log(max(floor_prior,min(1-floor_prior,self%class_prior_)))
    if(allocated(self%init_)) deallocate(self%init_)
    allocate(self%init_(self%n_trees_per_iteration_)); self%init_=0
    if(.not.self%init_zero) then
      if(c==2) then
        v=max(floor_prior,min(1-floor_prior,self%class_prior_(2)))
        self%init_(1)=log(v/(1-v))
        if(self%loss=='exponential') self%init_=self%init_/2
      else
        self%init_=logs-sum(logs)/c
      end if
    end if
    raw=spread(self%init_,1,n)
    allocate(gradient(n,self%n_trees_per_iteration_),indicator(n),hessian(n),mask(n))
    if(allocated(self%estimators_)) deallocate(self%estimators_)
    allocate(self%estimators_(self%n_estimators_,self%n_trees_per_iteration_))
    self%train_score_=spread(0._dp,1,self%n_estimators_); self%feature_importances_=spread(0._dp,1,p)
    do t=1,self%n_estimators_
      prob=gbc_probabilities(self,raw)
      do k=1,self%n_trees_per_iteration_
        if(c==2) then
          indicator=merge(1._dp,0._dp,encoded==2)
          if(self%loss=='exponential') then
            gradient(:,k)=(2*indicator-1)*exp(-(2*indicator-1)*raw(:,1))
          else
            gradient(:,k)=indicator-prob(:,2)
          end if
        else
          gradient(:,k)=merge(1._dp,0._dp,encoded==k)-prob(:,k)
        end if
      end do
      do k=1,self%n_trees_per_iteration_
        self%estimators_(t,k)%max_depth=self%max_depth
        self%estimators_(t,k)%min_samples_leaf=self%min_samples_leaf
        self%estimators_(t,k)%min_samples_split=self%min_samples_split
        self%estimators_(t,k)%max_features=self%max_features
        ! Friedman improvement is the SSE reduction, rather than reduction / n.
        self%estimators_(t,k)%min_impurity_decrease=self%min_impurity_decrease/real(n,dp)
        self%estimators_(t,k)%random_state=self%random_state+t+7919*k
        call self%estimators_(t,k)%fit(x,gradient(:,k))
        leaves=self%estimators_(t,k)%apply(x)
        if(c==2) then
          if(self%loss=='exponential') then
            hessian=abs(gradient(:,k))
          else
            hessian=prob(:,2)*(1-prob(:,2))
          end if
        else
          hessian=prob(:,k)*(1-prob(:,k))
        end if
        factor=1
        if(c>2) factor=real(c-1,dp)/c
        do leaf=1,self%estimators_(t,k)%node_count_
          left=self%estimators_(t,k)%nodes_(leaf)%left
          if(left>0) then
            right=self%estimators_(t,k)%nodes_(leaf)%right
            feature=self%estimators_(t,k)%nodes_(leaf)%feature
            gain=self%estimators_(t,k)%nodes_(leaf)%n_samples*self%estimators_(t,k)%nodes_(leaf)%impurity- &
              self%estimators_(t,k)%nodes_(left)%n_samples*self%estimators_(t,k)%nodes_(left)%impurity- &
              self%estimators_(t,k)%nodes_(right)%n_samples*self%estimators_(t,k)%nodes_(right)%impurity
            self%feature_importances_(feature)=self%feature_importances_(feature)+max(0._dp,gain)
            cycle
          end if
          mask=leaves==leaf; mass=count(mask)
          numerator=sum(gradient(:,k),mask=mask)/mass
          numerator=factor*numerator
          denominator=sum(hessian,mask=mask)/mass
          v=0
          if(abs(denominator)>=1.e-150_dp) v=numerator/denominator
          self%estimators_(t,k)%nodes_(leaf)%value=v
        end do
        raw(:,k)=raw(:,k)+self%fitted_rate*self%estimators_(t,k)%predict(x)
      end do
      stage_loss=0
      do i=1,n
        if(c>2) then
          stage_loss=stage_loss+logsumexp(raw(i,:))-raw(i,encoded(i))
        else if(self%loss=='exponential') then
          stage_loss=stage_loss+exp(-merge(1._dp,-1._dp,encoded(i)==2)*raw(i,1))
        else
          stage_loss=stage_loss+2*gbc_softplus(merge(-raw(i,1),raw(i,1),encoded(i)==2))
        end if
      end do
      self%train_score_(t)=stage_loss/n
    end do
    if(sum(self%feature_importances_)>0) &
      self%feature_importances_=self%feature_importances_/sum(self%feature_importances_)
  end subroutine

  function gbc_decision(self,x) result(raw)
    class(gradient_boosting_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: raw(:,:)
    integer :: t,k
    call require(allocated(self%estimators_),'GradientBoostingClassifier is not fitted')
    call check_xy(transpose(x),self%n_features_in_)
    raw=spread(self%init_,1,size(x,1))
    do t=1,self%n_estimators_
      do k=1,self%n_trees_per_iteration_
        raw(:,k)=raw(:,k)+self%fitted_rate*self%estimators_(t,k)%predict(x)
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
      labels=self%classes_(merge(2,1,raw(:,1)>0))
    else
      labels=self%classes_(maxloc(raw,dim=2))
    end if
  end function

  real(dp) function gbc_score(self,x,y) result(v)
    class(gradient_boosting_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    v=accuracy_score(y,self%predict(x))
  end function

  function gbc_staged_decision(self,x) result(stages)
    class(gradient_boosting_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: stages(:,:,:),raw(:,:)
    integer :: t,k
    call require(allocated(self%estimators_),'GradientBoostingClassifier is not fitted')
    call check_xy(transpose(x),self%n_features_in_)
    allocate(stages(size(x,1),self%n_trees_per_iteration_,self%n_estimators_))
    raw=spread(self%init_,1,size(x,1))
    do t=1,self%n_estimators_
      do k=1,self%n_trees_per_iteration_
        raw(:,k)=raw(:,k)+self%fitted_rate*self%estimators_(t,k)%predict(x)
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
    allocate(stages(size(x,1),size(self%classes_),self%n_estimators_))
    do t=1,self%n_estimators_
      stages(:,:,t)=gbc_probabilities(self,raw(:,:,t))
    end do
  end function

  function gbc_staged_predict(self,x) result(stages)
    class(gradient_boosting_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: stages(:,:)
    real(dp), allocatable :: raw(:,:,:)
    integer :: t
    raw=self%staged_decision_function(x); allocate(stages(size(x,1),self%n_estimators_))
    do t=1,self%n_estimators_
      if(size(self%classes_)==2) then
        stages(:,t)=self%classes_(merge(2,1,raw(:,1,t)>0))
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
    allocate(leaves(size(x,1),self%n_estimators_,self%n_trees_per_iteration_))
    do k=1,self%n_trees_per_iteration_
      do t=1,self%n_estimators_
        leaves(:,t,k)=self%estimators_(t,k)%apply(x)
      end do
    end do
  end function
end module
