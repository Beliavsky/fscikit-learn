! Parameter-only copies at native sklearn-style clone sites (MIT).
! The historical reset_sgd_fit entry point also resets Huber warm-fit state.
! Preserve constructor controls and discard learned state, as sklearn.base.clone.
module sklearn_sgd_clone
  use sklearn_sgd, only: sgd_regressor,sgd_classifier
  use sklearn_linear_model,only:passive_aggressive_regressor,passive_aggressive_classifier
  use sklearn_quantile,only:quantile_regressor
  use sklearn_linear_model,only:ard_regression,bayesian_ridge,huber_regressor,tweedie_regressor,poisson_regressor,gamma_regressor
  implicit none
  private
  public :: clone_sgd_parameters,reset_sgd_fit
  public :: clone_quantile_parameters,clone_pa_parameters
  interface clone_sgd_parameters
    module procedure clone_sgdr_parameters,clone_sgdc_parameters
  end interface
  interface clone_pa_parameters
    module procedure clone_par_parameters,clone_pac_parameters
  end interface
contains
  function clone_par_parameters(source)result(model)
    type(passive_aggressive_regressor),intent(in)::source
    type(passive_aggressive_regressor)::model
    model%c=source%c;model%tol=source%tol;model%epsilon=source%epsilon;model%loss=source%loss
    model%max_iter=source%max_iter;model%n_iter_no_change=source%n_iter_no_change
    model%average=source%average;model%random_state=source%random_state;model%shuffle=source%shuffle
    model%fit_intercept=source%fit_intercept;model%positive=source%positive;model%warm_start=source%warm_start
  end function
  function clone_pac_parameters(source)result(model)
    type(passive_aggressive_classifier),intent(in)::source
    type(passive_aggressive_classifier)::model
    model%c=source%c;model%tol=source%tol;model%loss=source%loss
    model%max_iter=source%max_iter;model%n_iter_no_change=source%n_iter_no_change
    model%average=source%average;model%random_state=source%random_state;model%shuffle=source%shuffle
    model%fit_intercept=source%fit_intercept;model%class_weight_balanced=source%class_weight_balanced
    model%warm_start=source%warm_start
  end function
  function clone_quantile_parameters(source)result(model)
    type(quantile_regressor),intent(in)::source
    type(quantile_regressor)::model
    model%quantile=source%quantile;model%alpha=source%alpha;model%fit_intercept=source%fit_intercept
    model%tol=source%tol;model%max_iter=source%max_iter
  end function
  function clone_sgdr_parameters(source) result(model)
    type(sgd_regressor), intent(in) :: source
    type(sgd_regressor) :: model
    model%loss=source%loss;model%penalty=source%penalty;model%learning_rate=source%learning_rate
    model%alpha=source%alpha;model%eta0=source%eta0;model%power_t=source%power_t
    model%epsilon=source%epsilon;model%tol=source%tol;model%l1_ratio=source%l1_ratio
    model%max_iter=source%max_iter;model%n_iter_no_change=source%n_iter_no_change;model%average=source%average
    model%random_state=source%random_state;model%fit_intercept=source%fit_intercept
    model%early_stopping=source%early_stopping;model%validation_fraction=source%validation_fraction
    model%shuffle=source%shuffle;model%warm_start=source%warm_start
  end function
  function clone_sgdc_parameters(source) result(model)
    type(sgd_classifier), intent(in) :: source
    type(sgd_classifier) :: model
    model%loss=source%loss;model%penalty=source%penalty;model%learning_rate=source%learning_rate
    model%alpha=source%alpha;model%eta0=source%eta0;model%power_t=source%power_t
    model%epsilon=source%epsilon;model%tol=source%tol;model%l1_ratio=source%l1_ratio
    model%max_iter=source%max_iter;model%n_iter_no_change=source%n_iter_no_change;model%average=source%average
    model%random_state=source%random_state;model%fit_intercept=source%fit_intercept
    model%early_stopping=source%early_stopping;model%validation_fraction=source%validation_fraction
    model%shuffle=source%shuffle;model%warm_start=source%warm_start;model%class_weight_balanced=source%class_weight_balanced
  end function
  subroutine reset_sgd_fit(model)
    class(*), intent(inout) :: model
    select type(model)
    type is(passive_aggressive_regressor)
      model=clone_pa_parameters(model)
    type is(passive_aggressive_classifier)
      model=clone_pa_parameters(model)
    type is(quantile_regressor)
      model=clone_quantile_parameters(model)
    type is(ard_regression)
      block
        type(ard_regression)::copy
        copy%fit_intercept=model%fit_intercept;copy%positive=model%positive;copy%compute_score=model%compute_score
        copy%max_iter=model%max_iter;copy%tol=model%tol;copy%alpha_1=model%alpha_1;copy%alpha_2=model%alpha_2
        copy%lambda_1=model%lambda_1;copy%lambda_2=model%lambda_2;copy%threshold_lambda=model%threshold_lambda
        model=copy
      end block
    type is(bayesian_ridge)
      block
        type(bayesian_ridge)::copy
        copy%fit_intercept=model%fit_intercept;copy%positive=model%positive;copy%compute_score=model%compute_score
        copy%max_iter=model%max_iter;copy%tol=model%tol;copy%alpha_1=model%alpha_1;copy%alpha_2=model%alpha_2
        copy%lambda_1=model%lambda_1;copy%lambda_2=model%lambda_2
        copy%alpha_init=model%alpha_init;copy%lambda_init=model%lambda_init;model=copy
      end block
    type is(huber_regressor)
      block
        type(huber_regressor)::copy
        copy%fit_intercept=model%fit_intercept;copy%positive=model%positive
        copy%epsilon=model%epsilon;copy%alpha=model%alpha;copy%tol=model%tol
        copy%max_iter=model%max_iter;copy%warm_start=model%warm_start
        model=copy
      end block
    type is(tweedie_regressor)
      block
        type(tweedie_regressor)::copy
        copy%fit_intercept=model%fit_intercept;copy%positive=model%positive
        copy%solver=model%solver;copy%power=model%power;copy%link=model%link;copy%alpha=model%alpha;copy%tol=model%tol
        copy%max_iter=model%max_iter;copy%warm_start=model%warm_start
        model=copy
      end block
    type is(poisson_regressor)
      block
        type(poisson_regressor)::copy
        copy%fit_intercept=model%fit_intercept;copy%positive=model%positive
        copy%solver=model%solver;copy%power=model%power;copy%link=model%link;copy%alpha=model%alpha;copy%tol=model%tol
        copy%max_iter=model%max_iter;copy%warm_start=model%warm_start
        model=copy
      end block
    type is(gamma_regressor)
      block
        type(gamma_regressor)::copy
        copy%fit_intercept=model%fit_intercept;copy%positive=model%positive
        copy%solver=model%solver;copy%power=model%power;copy%link=model%link;copy%alpha=model%alpha;copy%tol=model%tol
        copy%max_iter=model%max_iter;copy%warm_start=model%warm_start
        model=copy
      end block
    type is(sgd_regressor)
      model=clone_sgd_parameters(model)
    type is(sgd_classifier)
      model=clone_sgd_parameters(model)
    end select
  end subroutine
end module
