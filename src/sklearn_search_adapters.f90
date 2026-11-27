! Internal typed estimator adapters for sklearn-style parameter search (MIT).
module sklearn_search_adapters
  use sklearn_adaboost,only:adaboost_classifier
  use sklearn_adaboost_regression,only:adaboost_regressor
  use sklearn_classic_gradient_boosting,only:reset_boosting_fit
  use sklearn_common, only: dp,require,label_index,unique_labels
  use sklearn_parameter_space, only: parameter_set,parameter_value
  use sklearn_base_dispatch, only: base_supported,train_classifier,train_regressor,labels,regression, &
    probabilities,decisions,base_classes,base_decision
  use sklearn_linear_model, only: linear_regression,ridge,elastic_net,logistic_regression,tweedie_regressor,huber_regressor,ransac_regressor
  use sklearn_sgd, only: sgd_classifier,sgd_regressor
  use sklearn_sgd_clone, only: reset_sgd_fit
  use sklearn_naive_bayes, only: gaussian_nb,multinomial_nb
  use sklearn_neighbors, only: kneighbors_classifier,kneighbors_regressor
  use sklearn_tree, only: decision_tree_classifier,decision_tree_regressor
  use sklearn_ensemble, only: random_forest_classifier,random_forest_regressor, &
    gradient_boosting_classifier,gradient_boosting_regressor,reset_forest_fit
  use sklearn_preprocessing, only: standard_scaler,minmax_scaler
  use sklearn_decomposition, only: pca
  use sklearn_feature_selection, only: select_k_best,variance_threshold,select_from_model
  use sklearn_pipeline, only: pipeline
  use sklearn_compose, only: transformed_target_regressor
  use sklearn_svm, only: svc,linear_svc
  use sklearn_weighted_linear, only: weighted_linear
  use sklearn_composite_scores, only: regression_score
  use sklearn_metrics, only: accuracy_score,balanced_accuracy_score,f1_score,roc_auc_score
  use sklearn_ieee_math, only: ieee_mean,ieee_product,ieee_difference
  use,intrinsic::ieee_arithmetic,only:ieee_is_finite
  implicit none
  private
  public :: search_classifier,set_search_params,fit_search_native,search_regression,search_labels
  public :: reset_search_clone
  public :: search_probabilities,search_decisions,search_classes,native_search_score
contains
  recursive subroutine reset_search_clone(model)
    class(*), intent(inout) :: model
    integer :: i
    select type(model)
    type is(pipeline)
      if(allocated(model%steps)) then
        do i=1,size(model%steps)
          if(allocated(model%steps(i)%model)) call reset_search_clone(model%steps(i)%model)
        end do
      end if
    class default
      call reset_sgd_fit(model)
      call reset_boosting_fit(model)
      call reset_forest_fit(model)
    end select
  end subroutine
  recursive logical function search_classifier(model) result(classifier)
    class(*), intent(in) :: model
    classifier=base_supported(model,.true.)
    select type(model)
    type is(pipeline)
      call require(allocated(model%steps),'search Pipeline steps are unallocated')
      call require(size(model%steps)>0,'search Pipeline must have steps')
      call require(allocated(model%steps(size(model%steps))%model),'search Pipeline final model is unallocated')
      classifier=search_classifier(model%steps(size(model%steps))%model)
    class is(random_forest_classifier)
      classifier=.true.
    type is(gradient_boosting_classifier)
      classifier=.true.
    end select
  end function
  recursive subroutine set_parameter(model,name,value)
    class(*), intent(inout) :: model
    character(*), intent(in) :: name
    type(parameter_value), intent(in) :: value
    integer :: separator,step
    logical :: done
    separator=index(name,'__')
    if(separator>0) then
      call require(separator>1.and.separator+1<len(name),'search invalid nested parameter name')
      select type(model)
      type is(pipeline)
        step=model%named_step_index(name(:separator-1));call require(step>0,'search unknown Pipeline step')
        call require(model%steps(step)%action=='model','search cannot set nested parameter on passthrough')
        call require(allocated(model%steps(step)%model),'search Pipeline child is unallocated')
        call set_parameter(model%steps(step)%model,name(separator+2:),value);return
      type is(select_from_model)
        call require(name(:separator-1)=='estimator','search unknown SelectFromModel child')
        call require(allocated(model%estimator),'search selector estimator is unallocated')
        call set_parameter(model%estimator,name(separator+2:),value);return
      type is(ransac_regressor)
        call require(name(:separator-1)=='estimator','search unknown RANSAC child')
        call require(name(separator+2:)=='fit_intercept','search unsupported RANSAC OLS parameter')
        call set_parameter(model,'fit_intercept',value);return
      type is(adaboost_classifier)
        call require(name(:separator-1)=='estimator','search unknown AdaBoost child')
        if(name(separator+2:)=='random_state')then
          model%tree_random_state=value%as_integer();return
        end if
        call require(any(name(separator+2:)==[character(32)::'max_depth','min_samples_leaf','min_samples_split', &
          'max_features','criterion','splitter','min_impurity_decrease','max_leaf_nodes','ccp_alpha', &
          'min_weight_fraction_leaf']),'search unsupported AdaBoost tree parameter')
        call set_parameter(model,name(separator+2:),value);return
      type is(adaboost_regressor)
        call require(name(:separator-1)=='estimator','search unknown AdaBoost child')
        if(name(separator+2:)=='random_state')then
          model%tree_random_state=value%as_integer();return
        end if
        call require(any(name(separator+2:)==[character(32)::'max_depth','min_samples_leaf','min_samples_split', &
          'max_features','criterion','splitter','min_impurity_decrease','max_leaf_nodes','ccp_alpha', &
          'min_weight_fraction_leaf']),'search unsupported AdaBoost tree parameter')
        call set_parameter(model,name(separator+2:),value);return
      type is(transformed_target_regressor)
        call require(name(:separator-1)=='regressor','search unknown target-regressor child')
        call require(allocated(model%regressor),'search target regressor is unallocated')
        call set_parameter(model%regressor,name(separator+2:),value);return
      class default
        call require(.false.,'search estimator has no supported nested parameter')
      end select
    end if
    done=.true.
    select type(model)
    type is(pipeline)
      step=model%named_step_index(name);call require(step>0,'search unknown Pipeline parameter')
      call require(value%as_text()=='passthrough','search Pipeline replacement requires passthrough or a callback')
      model%steps(step)%action='passthrough'
    class is(logistic_regression)
      select case(name)
      case('C');model%c=value%as_real()
      case('tol');model%tol=value%as_real()
      case('max_iter');model%max_iter=value%as_integer()
      case('fit_intercept');model%fit_intercept=value%as_logical()
      case default;done=.false.
      end select
    type is(ransac_regressor)
      select case(name)
      case('min_samples')
        select case(value%kind)
        case(1);call model%set_min_samples(value%as_real())
        case(2);call model%set_min_samples(value%as_integer())
        case(4)
          call require(value%as_text()=='none','RANSAC min_samples text must be none')
          call model%clear_min_samples()
        case default;done=.false.
        end select
      case('residual_threshold')
        if(value%kind==4)then
          call require(value%as_text()=='none','RANSAC threshold text must be none')
          model%residual_threshold=-1
        else
          model%residual_threshold=value%as_real()
          call require(ieee_is_finite(model%residual_threshold),'RANSAC threshold must be finite')
          call require(model%residual_threshold>=0,'RANSAC threshold must be nonnegative')
        end if
      case('max_trials');model%max_trials=value%as_integer()
      case('max_skips');model%max_skips=value%as_integer()
      case('stop_n_inliers');model%stop_n_inliers=value%as_integer()
      case('stop_score');model%stop_score=value%as_real()
      case('stop_probability');model%stop_probability=value%as_real()
      case('random_state');model%random_state=value%as_integer()
      case('fit_intercept');model%fit_intercept=value%as_logical()
      case('loss');model%loss=value%as_text()
      case default;done=.false.
      end select
    type is(huber_regressor)
      select case(name)
      case('epsilon');model%epsilon=value%as_real()
      case('alpha');model%alpha=value%as_real()
      case('tol');model%tol=value%as_real()
      case('max_iter');model%max_iter=value%as_integer()
      case('fit_intercept');model%fit_intercept=value%as_logical()
      case('warm_start');model%warm_start=value%as_logical()
      case default;done=.false.
      end select
    class is(tweedie_regressor)
      select case(name)
      case('power','link')
        select type(model)
        type is(tweedie_regressor)
          if(name=='power')model%power=value%as_real()
          if(name=='link')model%link=value%as_text()
        class default;done=.false.
        end select
      case('alpha');model%alpha=value%as_real()
      case('tol');model%tol=value%as_real()
      case('max_iter');model%max_iter=value%as_integer()
      case('fit_intercept');model%fit_intercept=value%as_logical()
      case('warm_start');model%warm_start=value%as_logical()
      case default;done=.false.
      end select
    class is(linear_regression)
      select case(name)
      case('fit_intercept');model%fit_intercept=value%as_logical()
      case('alpha')
        select type(model)
        type is(ridge)
          model%alpha=value%as_real()
          if(allocated(model%alpha_per_target))deallocate(model%alpha_per_target)
        class is(elastic_net);model%alpha=value%as_real()
        class default;done=.false.
        end select
      case('l1_ratio','tol','max_iter','solver','random_state','positive')
        select type(model)
        type is(ridge)
          select case(name)
          case('tol');model%tol=value%as_real()
          case('max_iter');model%max_iter=value%as_integer()
          case('solver');model%solver=value%as_text()
          case('random_state');model%random_state=value%as_integer()
          case('positive');model%positive=value%as_logical()
          case default;done=.false.
          end select
        class is(elastic_net)
          select case(name)
          case('l1_ratio');model%l1_ratio=value%as_real()
          case('tol');model%tol=value%as_real()
          case('max_iter');model%max_iter=value%as_integer()
          case default;done=.false.
          end select
        class default;done=.false.
        end select
      case default;done=.false.
      end select
    type is(kneighbors_classifier)
      select case(name)
      case('n_neighbors');model%n_neighbors=value%as_integer()
      case('weights');model%weights=value%as_text()
      case('metric');model%metric=value%as_text()
      case('p');model%p=value%as_real()
      case default;done=.false.
      end select
    type is(kneighbors_regressor)
      select case(name)
      case('n_neighbors');model%n_neighbors=value%as_integer()
      case('weights');model%weights=value%as_text()
      case('metric');model%metric=value%as_text()
      case('p');model%p=value%as_real()
      case default;done=.false.
      end select
    class is(decision_tree_classifier)
      select case(name)
      case('max_leaf_nodes');model%max_leaf_nodes=value%as_integer()
      case('min_weight_fraction_leaf');model%min_weight_fraction_leaf=value%as_real()
      case('ccp_alpha');model%ccp_alpha=value%as_real()
      case('max_depth');model%max_depth=value%as_integer()
      case('min_samples_leaf')
        if(value%kind==1)then
          call model%set_min_samples_leaf(value%as_real())
        else
          call model%set_min_samples_leaf(value%as_integer())
        end if
      case('min_samples_split')
        if(value%kind==1)then
          call model%set_min_samples_split(value%as_real())
        else
          call model%set_min_samples_split(value%as_integer())
        end if
      case('max_features')
        select case(value%kind)
        case(1);call model%set_max_features(value%as_real())
        case(4);call model%set_max_features(value%as_text())
        case default;call model%set_max_features(value%as_integer())
        end select
      case('random_state');model%random_state=value%as_integer()
      case('criterion');model%criterion=value%as_text()
      case('min_impurity_decrease');model%min_impurity_decrease=value%as_real()
      case default;done=.false.
      end select
    class is(decision_tree_regressor)
      select case(name)
      case('max_leaf_nodes');model%max_leaf_nodes=value%as_integer()
      case('min_weight_fraction_leaf');model%min_weight_fraction_leaf=value%as_real()
      case('ccp_alpha');model%ccp_alpha=value%as_real()
      case('criterion');model%criterion=value%as_text()
      case('max_depth');model%max_depth=value%as_integer()
      case('min_samples_leaf')
        if(value%kind==1)then
          call model%set_min_samples_leaf(value%as_real())
        else
          call model%set_min_samples_leaf(value%as_integer())
        end if
      case('min_samples_split')
        if(value%kind==1)then
          call model%set_min_samples_split(value%as_real())
        else
          call model%set_min_samples_split(value%as_integer())
        end if
      case('max_features')
        select case(value%kind)
        case(1);call model%set_max_features(value%as_real())
        case(4);call model%set_max_features(value%as_text())
        case default;call model%set_max_features(value%as_integer())
        end select
      case('random_state');model%random_state=value%as_integer()
      case('min_impurity_decrease');model%min_impurity_decrease=value%as_real()
      case default;done=.false.
      end select
    type is(gaussian_nb)
      select case(name)
      case('var_smoothing');model%var_smoothing=value%as_real()
      case default;done=.false.
      end select
    class is(multinomial_nb)
      select case(name)
      case('alpha');model%alpha=value%as_real()
      case('fit_prior');model%fit_prior=value%as_logical()
      case default;done=.false.
      end select
    type is(sgd_regressor)
      select case(name)
      case('epsilon');model%epsilon=value%as_real()
      case('l1_ratio');model%l1_ratio=value%as_real()
      case('warm_start');model%warm_start=value%as_logical()
      case('average');model%average=value%as_integer()
      case('random_state');model%random_state=value%as_integer()
      case('n_iter_no_change');model%n_iter_no_change=value%as_integer()
      case('alpha');model%alpha=value%as_real()
      case('eta0');model%eta0=value%as_real()
      case('power_t');model%power_t=value%as_real()
      case('tol');model%tol=value%as_real()
      case('max_iter');model%max_iter=value%as_integer()
      case('loss');model%loss=value%as_text()
      case('penalty');model%penalty=value%as_text()
      case('learning_rate');model%learning_rate=value%as_text()
      case('fit_intercept');model%fit_intercept=value%as_logical()
      case('shuffle');model%shuffle=value%as_logical()
      case default;done=.false.
      end select
    type is(sgd_classifier)
      select case(name)
      case('epsilon');model%epsilon=value%as_real()
      case('l1_ratio');model%l1_ratio=value%as_real()
      case('warm_start');model%warm_start=value%as_logical()
      case('average');model%average=value%as_integer()
      case('random_state');model%random_state=value%as_integer()
      case('n_iter_no_change');model%n_iter_no_change=value%as_integer()
      case('alpha');model%alpha=value%as_real()
      case('eta0');model%eta0=value%as_real()
      case('power_t');model%power_t=value%as_real()
      case('tol');model%tol=value%as_real()
      case('max_iter');model%max_iter=value%as_integer()
      case('loss');model%loss=value%as_text()
      case('penalty');model%penalty=value%as_text()
      case('learning_rate');model%learning_rate=value%as_text()
      case('fit_intercept');model%fit_intercept=value%as_logical()
      case('shuffle');model%shuffle=value%as_logical()
      case default;done=.false.
      end select
    class is(random_forest_classifier)
      select case(name)
      case('warm_start');model%warm_start=value%as_logical()
      case('oob_score');model%oob_score=value%as_logical()
      case('bootstrap');call model%set_bootstrap(value%as_logical())
      case('max_leaf_nodes');model%max_leaf_nodes=value%as_integer()
      case('min_weight_fraction_leaf');model%min_weight_fraction_leaf=value%as_real()
      case('ccp_alpha');model%ccp_alpha=value%as_real()
      case('criterion');model%criterion=value%as_text()
      case('class_weight');model%class_weight=value%as_text()
      case('max_samples')
        if(value%kind==1)then
          call model%set_max_samples(value%as_real())
        else
          call model%set_max_samples(value%as_integer())
        end if
      case('min_impurity_decrease');model%min_impurity_decrease=value%as_real()
      case('n_estimators');model%n_estimators=value%as_integer()
      case('max_depth');model%max_depth=value%as_integer()
      case('min_samples_leaf')
        if(value%kind==1)then
          call model%set_min_samples_leaf(value%as_real())
        else
          call model%set_min_samples_leaf(value%as_integer())
        end if
      case('min_samples_split')
        if(value%kind==1)then
          call model%set_min_samples_split(value%as_real())
        else
          call model%set_min_samples_split(value%as_integer())
        end if
      case('max_features')
        select case(value%kind)
        case(1);call model%set_max_features(value%as_real())
        case(4);call model%set_max_features(value%as_text())
        case default;call model%set_max_features(value%as_integer())
        end select
      case('random_state');model%random_state=value%as_integer()
      case default;done=.false.
      end select
    class is(random_forest_regressor)
      select case(name)
      case('warm_start');model%warm_start=value%as_logical()
      case('oob_score');model%oob_score=value%as_logical()
      case('bootstrap');call model%set_bootstrap(value%as_logical())
      case('max_leaf_nodes');model%max_leaf_nodes=value%as_integer()
      case('min_weight_fraction_leaf');model%min_weight_fraction_leaf=value%as_real()
      case('ccp_alpha');model%ccp_alpha=value%as_real()
      case('criterion');model%criterion=value%as_text()
      case('max_samples')
        if(value%kind==1)then
          call model%set_max_samples(value%as_real())
        else
          call model%set_max_samples(value%as_integer())
        end if
      case('min_impurity_decrease');model%min_impurity_decrease=value%as_real()
      case('n_estimators');model%n_estimators=value%as_integer()
      case('max_depth');model%max_depth=value%as_integer()
      case('min_samples_leaf')
        if(value%kind==1)then
          call model%set_min_samples_leaf(value%as_real())
        else
          call model%set_min_samples_leaf(value%as_integer())
        end if
      case('min_samples_split')
        if(value%kind==1)then
          call model%set_min_samples_split(value%as_real())
        else
          call model%set_min_samples_split(value%as_integer())
        end if
      case('max_features')
        select case(value%kind)
        case(1);call model%set_max_features(value%as_real())
        case(4);call model%set_max_features(value%as_text())
        case default;call model%set_max_features(value%as_integer())
        end select
      case('random_state');model%random_state=value%as_integer()
      case default;done=.false.
      end select
    type is(adaboost_classifier)
      select case(name)
      case('n_estimators');model%n_estimators=value%as_integer()
      case('random_state');model%random_state=value%as_integer()
      case('tree_random_state');model%tree_random_state=value%as_integer()
      case('learning_rate');model%learning_rate=value%as_real()
      case('max_depth');model%max_depth=value%as_integer()
      case('criterion');model%criterion=value%as_text()
      case('splitter');model%splitter=value%as_text()
      case('min_impurity_decrease');model%min_impurity_decrease=value%as_real()
      case('max_leaf_nodes');model%max_leaf_nodes=value%as_integer()
      case('ccp_alpha');model%ccp_alpha=value%as_real()
      case('min_weight_fraction_leaf');model%min_weight_fraction_leaf=value%as_real()
      case('min_samples_leaf')
        if(value%kind==1)then
          call model%set_min_samples_leaf(value%as_real())
        else
          call model%set_min_samples_leaf(value%as_integer())
        end if
      case('min_samples_split')
        if(value%kind==1)then
          call model%set_min_samples_split(value%as_real())
        else
          call model%set_min_samples_split(value%as_integer())
        end if
      case('max_features')
        select case(value%kind)
        case(1);call model%set_max_features(value%as_real())
        case(3);done=.false.
        case(4);call model%set_max_features(value%as_text())
        case default;call model%set_max_features(value%as_integer())
        end select
      case default;done=.false.
      end select
    type is(adaboost_regressor)
      select case(name)
      case('n_estimators');model%n_estimators=value%as_integer()
      case('random_state');model%random_state=value%as_integer()
      case('tree_random_state');model%tree_random_state=value%as_integer()
      case('learning_rate');model%learning_rate=value%as_real()
      case('max_depth');model%max_depth=value%as_integer()
      case('criterion');model%criterion=value%as_text()
      case('splitter');model%splitter=value%as_text()
      case('min_impurity_decrease');model%min_impurity_decrease=value%as_real()
      case('max_leaf_nodes');model%max_leaf_nodes=value%as_integer()
      case('ccp_alpha');model%ccp_alpha=value%as_real()
      case('min_weight_fraction_leaf');model%min_weight_fraction_leaf=value%as_real()
      case('min_samples_leaf')
        if(value%kind==1)then
          call model%set_min_samples_leaf(value%as_real())
        else
          call model%set_min_samples_leaf(value%as_integer())
        end if
      case('min_samples_split')
        if(value%kind==1)then
          call model%set_min_samples_split(value%as_real())
        else
          call model%set_min_samples_split(value%as_integer())
        end if
      case('max_features')
        select case(value%kind)
        case(1);call model%set_max_features(value%as_real())
        case(3);done=.false.
        case(4);call model%set_max_features(value%as_text())
        case default;call model%set_max_features(value%as_integer())
        end select
      case('loss');model%loss=value%as_text()
      case default;done=.false.
      end select
    type is(gradient_boosting_classifier)
      select case(name)
      case('n_iter_no_change');model%n_iter_no_change=value%as_integer()
      case('validation_fraction');model%validation_fraction=value%as_real()
      case('tol');model%tol=value%as_real()
      case('subsample');model%subsample=value%as_real()
      case('warm_start');model%warm_start=value%as_logical()
      case('criterion');model%criterion=value%as_text()
      case('max_leaf_nodes');model%max_leaf_nodes=value%as_integer()
      case('min_weight_fraction_leaf');model%min_weight_fraction_leaf=value%as_real()
      case('ccp_alpha');model%ccp_alpha=value%as_real()
      case('init_zero');model%init_zero=value%as_logical()
      case('min_impurity_decrease');model%min_impurity_decrease=value%as_real()
      case('n_estimators');model%n_estimators=value%as_integer()
      case('max_depth');model%max_depth=value%as_integer()
      case('min_samples_leaf')
        if(value%kind==1)then
          call model%set_min_samples_leaf(value%as_real())
        else
          call model%set_min_samples_leaf(value%as_integer())
        end if
      case('random_state');model%random_state=value%as_integer()
      case('learning_rate');model%learning_rate=value%as_real()
      case('min_samples_split')
        if(value%kind==1)then
          call model%set_min_samples_split(value%as_real())
        else
          call model%set_min_samples_split(value%as_integer())
        end if
      case('max_features')
        select case(value%kind)
        case(1);call model%set_max_features(value%as_real())
        case(4);call model%set_max_features(value%as_text())
        case default;call model%set_max_features(value%as_integer())
        end select
      case('loss');model%loss=value%as_text()
      case default;done=.false.
      end select
    type is(gradient_boosting_regressor)
      select case(name)
      case('n_iter_no_change');model%n_iter_no_change=value%as_integer()
      case('validation_fraction');model%validation_fraction=value%as_real()
      case('tol');model%tol=value%as_real()
      case('subsample');model%subsample=value%as_real()
      case('warm_start');model%warm_start=value%as_logical()
      case('n_estimators');model%n_estimators=value%as_integer()
      case('max_depth');model%max_depth=value%as_integer()
      case('min_samples_leaf')
        if(value%kind==1)then
          call model%set_min_samples_leaf(value%as_real())
        else
          call model%set_min_samples_leaf(value%as_integer())
        end if
      case('random_state');model%random_state=value%as_integer()
      case('learning_rate');model%learning_rate=value%as_real()
      case('loss');model%loss=value%as_text()
      case('criterion');model%criterion=value%as_text()
      case('alpha');model%alpha=value%as_real()
      case('init_zero');model%init_zero=value%as_logical()
      case('min_samples_split')
        if(value%kind==1)then
          call model%set_min_samples_split(value%as_real())
        else
          call model%set_min_samples_split(value%as_integer())
        end if
      case('max_leaf_nodes');model%max_leaf_nodes=value%as_integer()
      case('ccp_alpha');model%ccp_alpha=value%as_real()
      case('min_weight_fraction_leaf');model%min_weight_fraction_leaf=value%as_real()
      case('min_impurity_decrease');model%min_impurity_decrease=value%as_real()
      case('max_features')
        select case(value%kind)
        case(1);call model%set_max_features(value%as_real())
        case(4);call model%set_max_features(value%as_text())
        case default;call model%set_max_features(value%as_integer())
        end select
      case default;done=.false.
      end select
    type is(standard_scaler)
      select case(name)
      case('with_mean');model%with_mean=value%as_logical()
      case('with_std');model%with_std=value%as_logical()
      case default;done=.false.
      end select
    type is(minmax_scaler)
      select case(name)
      case('clip');model%clip=value%as_logical()
      case default;done=.false.
      end select
    type is(pca)
      select case(name)
      case('n_components');model%n_components=value%as_integer()
      case('whiten');model%whiten=value%as_logical()
      case default;done=.false.
      end select
    type is(select_k_best)
      select case(name)
      case('k');model%k=value%as_integer()
      case default;done=.false.
      end select
    type is(variance_threshold)
      select case(name)
      case('threshold');model%threshold=value%as_real()
      case default;done=.false.
      end select
    type is(select_from_model)
      select case(name)
      case('threshold')
        if(value%kind==4) then
          model%threshold=value%as_text()
        else
          model%threshold='value';model%numeric_threshold=value%as_real()
        end if
      case('max_features');model%max_features=value%as_integer()
      case('norm_order');model%norm_order=value%as_real()
      case default;done=.false.
      end select
    type is(linear_svc)
      select case(name)
      case('C');model%c=value%as_real()
      case('tol');model%tol=value%as_real()
      case('max_iter');model%max_iter=value%as_integer()
      case('fit_intercept');model%fit_intercept=value%as_logical()
      case('intercept_scaling');model%intercept_scaling=value%as_real()
      case default;done=.false.
      end select
    type is(svc)
      select case(name)
      case('C');model%c=value%as_real()
      case('gamma');model%gamma=value%as_real()
      case('coef0');model%coef0=value%as_real()
      case('degree');model%degree=value%as_integer()
      case('kernel');model%kernel=value%as_text()
      case('tol');model%tol=value%as_real()
      case('max_iter');model%max_iter=value%as_integer()
      case default;done=.false.
      end select
    class default
      done=.false.
    end select
    call require(done,'unsupported native search parameter: '//name)
  end subroutine
  subroutine set_search_params(model,parameters)
    class(*), intent(inout) :: model
    type(parameter_set), intent(in) :: parameters
    integer :: i
    if(.not.allocated(parameters%items)) return
    do i=1,size(parameters%items)
      call require(allocated(parameters%items(i)%name),'search parameter name is unallocated')
      call set_parameter(model,parameters%items(i)%name,parameters%items(i)%value)
    end do
  end subroutine
  subroutine fit_search_native(model,x,yr,yi,sample_weight)
    class(*), intent(inout) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: yr(:),sample_weight(:)
    integer, optional, intent(in) :: yi(:)
    real(dp) :: w(size(x,1))
    if(present(yi)) then
      select type(model)
      type is(pipeline)
        call model%fit(x,yi,sample_weight)
      class is(random_forest_classifier)
        call model%fit(x,yi,sample_weight)
      type is(gradient_boosting_classifier)
        call model%fit(x,yi,sample_weight)
      class default
        call train_classifier(model,x,yi,sample_weight)
      end select
    else
      call require(present(yr),'search fitting requires a target')
      select type(model)
      type is(pipeline)
        call model%fit(x,yr,sample_weight)
      type is(transformed_target_regressor)
        call model%fit(x,yr,sample_weight)
      type is(linear_regression)
        w=1;if(present(sample_weight)) w=sample_weight
        call weighted_linear(model,x,yr,w,0._dp)
      type is(ridge)
        w=1;if(present(sample_weight)) w=sample_weight
        call weighted_linear(model,x,yr,w,model%alpha)
      class is(random_forest_regressor)
        call model%fit(x,yr,sample_weight)
      type is(gradient_boosting_regressor)
        call model%fit(x,yr,sample_weight)
      class default
        call train_regressor(model,x,yr,sample_weight)
      end select
    end if
  end subroutine
  function search_regression(model,x) result(y)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:)
    select type(model)
    type is(pipeline);y=model%predict(x)
    type is(transformed_target_regressor);y=model%predict(x)
    class is(random_forest_regressor);y=model%predict(x)
    type is(gradient_boosting_regressor);y=model%predict(x)
    class default;y=regression(model,x)
    end select
  end function
  function search_labels(model,x) result(y)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: y(:)
    select type(model)
    type is(pipeline);y=model%predict_class(x)
    class is(random_forest_classifier);y=model%predict(x)
    type is(gradient_boosting_classifier);y=model%predict(x)
    class default;y=labels(model,x)
    end select
  end function
  function search_probabilities(model,x) result(y)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:,:)
    select type(model)
    type is(pipeline);y=model%predict_proba(x)
    class is(random_forest_classifier);y=model%predict_proba(x)
    type is(gradient_boosting_classifier);y=model%predict_proba(x)
    class default;y=probabilities(model,x)
    end select
  end function
  function search_decisions(model,x) result(y)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:,:)
    select type(model)
    type is(pipeline);y=model%decision_function(x)
    type is(gradient_boosting_classifier);y=model%decision_function(x)
    class default;y=decisions(model,x)
    end select
  end function
  recursive function search_classes(model) result(classes)
    class(*), intent(in) :: model
    integer, allocatable :: classes(:)
    select type(model)
    type is(pipeline);classes=search_classes(model%steps(size(model%steps))%model)
    class is(random_forest_classifier);classes=model%classes_
    type is(gradient_boosting_classifier);classes=model%classes_
    class default;classes=base_classes(model)
    end select
  end function
  recursive logical function has_decision(model) result(available)
    class(*), intent(in) :: model
    select type(model)
    type is(pipeline);available=has_decision(model%steps(size(model%steps))%model)
    type is(gradient_boosting_classifier);available=.true.
    class default;available=base_decision(model)
    end select
  end function
  real(dp) function native_search_score(model,x,scoring,yr,yi) result(value)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    character(*), intent(in) :: scoring
    real(dp), optional, intent(in) :: yr(:)
    integer, optional, intent(in) :: yi(:)
    real(dp), allocatable :: prediction(:),prob(:,:),response(:,:),loss(:)
    integer, allocatable :: predicted(:),classes(:)
    integer :: i,k
    if(present(yi)) then
      select case(scoring)
      case('auto','accuracy','balanced_accuracy','f1_macro','f1_weighted')
        predicted=search_labels(model,x)
        select case(scoring)
        case('auto','accuracy');value=accuracy_score(yi,predicted)
        case('balanced_accuracy');value=balanced_accuracy_score(yi,predicted)
        case('f1_macro');value=f1_score(yi,predicted,average='macro')
        case('f1_weighted');value=f1_score(yi,predicted,average='weighted')
        end select
      case('neg_log_loss')
        prob=search_probabilities(model,x);classes=unique_labels(yi)
        call require(size(prob,1)==size(yi).and.size(prob,2)==size(classes),'search log loss class count mismatch')
        call require(size(classes)>=2,'search log loss requires multiple classes')
        prob=max(epsilon(1._dp),min(1-epsilon(1._dp),prob));allocate(loss(size(yi)))
        do i=1,size(yi)
          k=label_index(classes,yi(i));call require(k>0,'search scoring label is unknown')
          loss(i)=log(prob(i,k))
        end do
        value=ieee_mean(loss)
      case('roc_auc')
        classes=search_classes(model);call require(size(classes)==2,'search roc_auc requires binary classification')
        if(has_decision(model)) then
          response=search_decisions(model,x)
        else
          prob=search_probabilities(model,x);response=prob(:,2:2)
        end if
        value=roc_auc_score(yi,response(:,1),pos_label=classes(2))
      case default
        call require(.false.,'unsupported native classification search scorer')
      end select
    else
      call require(present(yr),'search scoring requires a target')
      if(scoring=='auto') then
        select type(model)
        type is(pipeline);value=model%score(x,yr);return
        type is(transformed_target_regressor);value=model%score(x,yr);return
        class is(tweedie_regressor);value=model%score(x,yr);return
        end select
      end if
      prediction=search_regression(model,x)
      select case(scoring)
      case('auto','r2');value=regression_score(yr,prediction,context='search')
      case('neg_mean_squared_error','neg_root_mean_squared_error')
        allocate(loss(size(yr)))
        do i=1,size(yr)
          loss(i)=ieee_difference(yr(i),prediction(i));loss(i)=ieee_product(loss(i),loss(i))
        end do
        value=-ieee_mean(loss)
        if(scoring=='neg_root_mean_squared_error') value=-sqrt(-value)
      case('neg_mean_absolute_error');value=-ieee_mean(abs(yr-prediction))
      case default;call require(.false.,'unsupported native regression search scorer')
      end select
    end if
  end function
end module
