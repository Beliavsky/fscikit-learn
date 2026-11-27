! Internal native base dispatch shared by voting and bagging ensembles.
! Existing voting dispatch semantics and diagnostics are retained.
module sklearn_base_dispatch
  use sklearn_classic_gradient_boosting,only:gradient_boosting_regressor,gradient_boosting_classifier
  use sklearn_model_selection, only: ridge_cv
  use sklearn_weighted_linear, only: weighted_ridge_loo
  use sklearn_common, only: dp,require,check_x,check_xy,unique_labels,label_index
  use sklearn_linear_model, only: linear_regression,elastic_net,logistic_regression,perceptron, &
    passive_aggressive_classifier,passive_aggressive_regressor,ransac_regressor,theil_sen_regressor,huber_regressor,tweedie_regressor,bayesian_ridge
  use sklearn_ridge_classification, only: ridge_classifier
  use sklearn_sgd, only: sgd_classifier,sgd_regressor
  use sklearn_naive_bayes, only: gaussian_nb,multinomial_nb
  use sklearn_neighbors, only: kneighbors_classifier,kneighbors_regressor
  use sklearn_tree, only: decision_tree_classifier,decision_tree_regressor
  use sklearn_svm, only: svc,linear_svc
  use sklearn_dummy, only: dummy_classifier,dummy_regressor
  use sklearn_kernel_ridge, only: kernel_ridge
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value,ieee_quiet_nan
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_adaboost_regression,only:adaboost_regressor
  use sklearn_adaboost, only: adaboost_classifier
  implicit none
  private
  public :: train_classifier,train_regressor,labels,regression,probabilities
  public :: base_weighted,base_supported,base_proba,base_classes,seed_base,base_decision,decisions
  public :: base_fitted
contains
  logical function base_fitted(model) result(ok)
    class(*), intent(in) :: model
    ok=.false.
    select type(model)
    class is(logistic_regression)
      ok=allocated(model%classes_)
    type is(gaussian_nb)
      ok=allocated(model%classes_)
    type is(sgd_classifier)
      ok=allocated(model%classes_)
    type is(kneighbors_classifier)
      ok=allocated(model%classes_)
    type is(gradient_boosting_classifier)
      ok=allocated(model%estimators_)
    class is(decision_tree_classifier)
      ok=allocated(model%classes_)
    class is(multinomial_nb)
      ok=allocated(model%classes_)
    type is(perceptron)
      ok=allocated(model%classes_)
    type is(passive_aggressive_classifier)
      ok=allocated(model%classes_)
    class is(ridge_classifier)
      ok=allocated(model%classes_)
    type is(linear_svc)
      ok=allocated(model%classes_)
    type is(svc)
      ok=allocated(model%classes_)
    type is(dummy_classifier)
      ok=allocated(model%classes_)
    type is(gradient_boosting_regressor)
      ok=allocated(model%estimators_)
    type is(adaboost_regressor)
      ok=allocated(model%estimators_)
    type is(adaboost_classifier)
      ok=model%n_estimators_>0
    end select
  end function
  subroutine train_classifier(model,x,y,weights)
    class(*), intent(inout) :: model
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(in), optional :: weights(:)
    select type(model)
    class is(logistic_regression)
      call model%fit(x,y,weights)
    type is(gaussian_nb)
      call model%fit(x,y,weights)
    type is(sgd_classifier)
      call model%fit(x,y,weights)
    type is(kneighbors_classifier)
      call require(.not.present(weights),'voting base has no native weighted fit')
      call model%fit(x,y)
    type is(gradient_boosting_classifier)
      call model%fit(x,y,weights)
    class is(decision_tree_classifier)
      call model%fit(x,y,weights)
    class is(multinomial_nb)
      call model%fit(x,y,weights)
    type is(perceptron)
      call require(.not.present(weights),'voting base has no native weighted fit')
      call model%fit(x,y)
    type is(passive_aggressive_classifier)
      call require(.not.present(weights),'voting base has no native weighted fit')
      call model%fit(x,y)
    class is(ridge_classifier)
      call model%fit(x,y,weights)
    type is(linear_svc)
      call model%fit(x,y,weights)
    type is(svc)
      call require(.not.present(weights),'voting base has no native weighted fit')
      call model%fit(x,y)
    type is(dummy_classifier)
      call model%fit(x,y,weights)
    type is(adaboost_classifier)
      call model%fit(x,y,weights)
    class default
      call require(.false.,'unsupported native voting classifier base')
    end select
  end subroutine

  subroutine train_regressor(model,x,y,weights)
    class(*), intent(inout) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(in) :: y(:)
    real(dp), intent(in), optional :: weights(:)
    select type(model)
    type is(sgd_regressor)
      call model%fit(x,y,weights)
    type is(ridge_cv)
      if(model%cv==0) then
        if(.not.allocated(model%alphas)) model%alphas=[.1_dp,1._dp,10._dp]
        call weighted_ridge_loo(model,x,y,model%alphas,model%alpha_,model%best_score_,model%cv_scores_,weights)
        model%alpha=model%alpha_
      else
        call require(.not.present(weights),'voting base has no native weighted fit')
        call model%fit(x,y)
      end if
    type is(ransac_regressor)
      if(present(weights))then
        call model%fit(x,y,weights)
      else
        call model%fit(x,y)
      end if
    type is(huber_regressor)
      if(present(weights))then
        call model%fit(x,y,weights)
      else
        call model%fit(x,y)
      end if
    type is(bayesian_ridge)
      if(present(weights))then
        call model%fit(x,y,weights)
      else
        call model%fit(x,y)
      end if
    class is(tweedie_regressor)
      if(present(weights))then
        call model%fit(x,y,weights)
      else
        call model%fit(x,y)
      end if
    class is(linear_regression)
      call require(.not.present(weights),'voting base has no native weighted fit')
      call model%fit(x,y)
    class is(elastic_net)
      call require(.not.present(weights),'voting base has no native weighted fit')
      call model%fit(x,y)
    type is(kneighbors_regressor)
      call require(.not.present(weights),'voting base has no native weighted fit')
      call model%fit(x,y)
    type is(gradient_boosting_regressor)
      call model%fit(x,y,weights)
    type is(adaboost_regressor)
      call model%fit(x,y,weights)
    class is(decision_tree_regressor)
      call model%fit(x,y,weights)
    type is(passive_aggressive_regressor)
      call require(.not.present(weights),'voting base has no native weighted fit')
      call model%fit(x,y)
    type is(kernel_ridge)
      call model%fit(x,y,weights)
    type is(dummy_regressor)
      call model%fit(x,y,weights)
    class default
      call require(.false.,'unsupported native voting regressor base')
    end select
  end subroutine

  function labels(model,x) result(values)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: values(:)
    select type(model)
    class is(logistic_regression)
      values=model%predict(x)
    type is(gaussian_nb)
      values=model%predict(x)
    type is(sgd_classifier)
      values=model%predict(x)
    type is(kneighbors_classifier)
      values=model%predict(x)
    type is(gradient_boosting_classifier)
      values=model%predict(x)
    class is(decision_tree_classifier)
      values=model%predict(x)
    class is(multinomial_nb)
      values=model%predict(x)
    type is(perceptron)
      values=model%predict(x)
    type is(passive_aggressive_classifier)
      values=model%predict(x)
    class is(ridge_classifier)
      values=model%predict(x)
    type is(linear_svc)
      values=model%predict(x)
    type is(svc)
      values=model%predict(x)
    type is(dummy_classifier)
      values=model%predict(x)
    type is(adaboost_classifier)
      values=model%predict(x)
    class default
      call require(.false.,'voting base does not support labels')
    end select
  end function

  function regression(model,x) result(values)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:)
    select type(model)
    type is(sgd_regressor)
      values=model%predict(x)
    class is(linear_regression)
      values=model%predict(x)
    class is(elastic_net)
      values=model%predict(x)
    type is(kneighbors_regressor)
      values=model%predict(x)
    type is(gradient_boosting_regressor)
      values=model%predict(x)
    type is(adaboost_regressor)
      values=model%predict(x)
    class is(decision_tree_regressor)
      values=model%predict(x)
    type is(passive_aggressive_regressor)
      values=model%predict(x)
    type is(kernel_ridge)
      values=model%predict(x)
    type is(dummy_regressor)
      values=model%predict(x)
    class default
      call require(.false.,'voting base does not support regression')
    end select
  end function

  function probabilities(model,x) result(values)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:,:)
    select type(model)
    class is(logistic_regression)
      values=model%predict_proba(x)
    type is(gaussian_nb)
      values=model%predict_proba(x)
    type is(sgd_classifier)
      call require(model%loss=='log_loss'.or.model%loss=='modified_huber','voting SGD loss has no probabilities')
      values=model%predict_proba(x)
    type is(kneighbors_classifier)
      values=model%predict_proba(x)
    type is(gradient_boosting_classifier)
      values=model%predict_proba(x)
    class is(decision_tree_classifier)
      values=model%predict_proba(x)
    class is(multinomial_nb)
      values=model%predict_proba(x)
    type is(dummy_classifier)
      values=model%predict_proba(x)
    type is(adaboost_classifier)
      values=model%predict_proba(x)
    class default
      call require(.false.,'voting base does not support probabilities')
    end select
  end function

  logical function base_supported(model,classifier) result(ok)
    class(*), intent(in) :: model
    logical, intent(in) :: classifier
    ok=.true.
    if(classifier) then
      select type(model)
      type is(adaboost_classifier)
      class is(logistic_regression)
      type is(gaussian_nb)
      type is(sgd_classifier)
      type is(kneighbors_classifier)
      type is(gradient_boosting_classifier)
      class is(decision_tree_classifier)
      class is(multinomial_nb)
      type is(perceptron)
      type is(passive_aggressive_classifier)
      class is(ridge_classifier)
      type is(linear_svc)
      type is(svc)
      type is(dummy_classifier)
      class default
        ok=.false.
      end select
    else
      select type(model)
      type is(sgd_regressor)
      class is(linear_regression)
      type is(kneighbors_regressor)
      type is(gradient_boosting_regressor)
      type is(adaboost_regressor)
      class is(decision_tree_regressor)
      type is(kernel_ridge)
      type is(dummy_regressor)
      class default
        ok=.false.
      end select
    end if
  end function
  logical function base_weighted(model) result(ok)
    class(*), intent(in) :: model
    ok=.true.
    select type(model)
    class is(logistic_regression)
    type is(gaussian_nb)
    class is(multinomial_nb)
    type is(sgd_classifier)
    type is(ransac_regressor)
    type is(huber_regressor)
    type is(bayesian_ridge)
    class is(tweedie_regressor)
    type is(sgd_regressor)
    type is(kernel_ridge)
    type is(dummy_classifier)
    type is(dummy_regressor)
    type is(ridge_cv)
      ok=model%cv==0
    type is(gradient_boosting_classifier)
    class is(decision_tree_classifier)
    type is(gradient_boosting_regressor)
    type is(adaboost_regressor)
    class is(decision_tree_regressor)
    class is(ridge_classifier)
    type is(linear_svc)
    type is(adaboost_classifier)
    class default
      ok=.false.
    end select
  end function
  logical function base_proba(model) result(ok)
    class(*), intent(in) :: model
    ok=.true.
    select type(model)
    class is(logistic_regression)
    type is(gaussian_nb)
    type is(sgd_classifier)
      ok=model%loss=='log_loss'.or.model%loss=='modified_huber'
    type is(kneighbors_classifier)
    type is(gradient_boosting_classifier)
    class is(decision_tree_classifier)
    class is(multinomial_nb)
    type is(dummy_classifier)
    type is(adaboost_classifier)
    class default
      ok=.false.
    end select
  end function
  function base_classes(model) result(values)
    class(*), intent(in) :: model
    integer, allocatable :: values(:)
    select type(model)
    class is(logistic_regression)
      values=model%classes_
    type is(gaussian_nb)
      values=model%classes_
    type is(sgd_classifier)
      values=model%classes_
    type is(kneighbors_classifier)
      values=model%classes_
    type is(gradient_boosting_classifier)
      values=model%classes_
    class is(decision_tree_classifier)
      values=model%classes_
    class is(multinomial_nb)
      values=model%classes_
    type is(dummy_classifier)
      values=model%classes_
    type is(perceptron)
      values=model%classes_
    type is(passive_aggressive_classifier)
      values=model%classes_
    class is(ridge_classifier)
      values=model%classes_
    type is(linear_svc)
      values=model%classes_
    type is(svc)
      values=model%classes_
    type is(adaboost_classifier)
      values=model%classes_
    class default
      call require(.false.,'native base has no class probability metadata')
    end select
  end function
  logical function base_decision(model) result(ok)
    class(*), intent(in) :: model
    ok=.true.
    select type(model)
    class is(logistic_regression)
    type is(sgd_classifier)
    type is(perceptron)
    type is(passive_aggressive_classifier)
    class is(ridge_classifier)
    type is(linear_svc)
    type is(svc)
    type is(gradient_boosting_classifier)
    type is(adaboost_classifier)
    class default
      ok=.false.
    end select
  end function
  function decisions(model,x) result(value)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: value(:,:)
    select type(model)
    class is(logistic_regression)
      value=model%decision_function(x)
    type is(sgd_classifier)
      value=model%decision_function(x)
    type is(perceptron)
      value=model%decision_function(x)
    type is(passive_aggressive_classifier)
      value=model%decision_function(x)
    class is(ridge_classifier)
      value=model%decision_function(x)
    type is(linear_svc)
      value=model%decision_function(x)
    type is(svc)
      value=model%decision_function(x)
    type is(gradient_boosting_classifier)
      value=model%decision_function(x)
    type is(adaboost_classifier)
      value=model%decision_function(x)
    class default
      call require(.false.,'native base has no decision_function')
    end select
  end function
  subroutine seed_base(model,seed)
    class(*), intent(inout) :: model
    integer, intent(in) :: seed
    type(legacy_random_state) :: rng
    integer :: value
    call rng%seed(seed); value=rng%integer_index(2147483647)-1
    select type(model)
    type is(gradient_boosting_classifier)
      model%random_state=value
    class is(decision_tree_classifier)
      model%random_state=value
    class is(decision_tree_regressor)
      model%random_state=value
    type is(gradient_boosting_regressor)
      model%random_state=value
    type is(adaboost_regressor)
      ! Explicit CART base: nested tree first, then the AdaBoost RNG.
      model%tree_random_state=value
      model%random_state=rng%integer_index(2147483647)-1
    type is(sgd_classifier)
      model%random_state=value
    type is(sgd_regressor)
      model%random_state=value
    type is(passive_aggressive_classifier)
      model%random_state=value
    type is(passive_aggressive_regressor)
      model%random_state=value
    type is(dummy_classifier)
      model%random_state=value
    type is(ransac_regressor)
      model%random_state=value
    type is(theil_sen_regressor)
      model%random_state=value
    type is(adaboost_classifier)
      ! Explicit CART base: nested tree first, then the AdaBoost RNG.
      model%tree_random_state=value
      model%random_state=rng%integer_index(2147483647)-1
    end select
  end subroutine
end module
