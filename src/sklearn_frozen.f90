! FrozenEstimator contracts: sklearn 1.6.1 frozen/_frozen.py (BSD-3-Clause).
! Typed live-reference/bridge adapters are original MIT contributions.
module sklearn_frozen
  use sklearn_common, only: dp,require
  use sklearn_transformer_protocol, only: dense_transformer
  use sklearn_feature_extraction, only: feature_string
  use sklearn_feature_composition, only: supported,transform_model,model_names,input_names,column_transformer,feature_union
  use sklearn_base_dispatch, only: base_supported,base_fitted,base_classes,base_proba,base_decision, &
    labels,regression,probabilities,decisions
  use sklearn_preprocessing, only: standard_scaler,minmax_scaler,maxabs_scaler,robust_scaler,simple_imputer, &
    one_hot_encoder,ordinal_encoder,knn_imputer,missing_indicator,power_transformer,quantile_transformer, &
    kbins_discretizer,spline_transformer
  use sklearn_function_transformer, only: function_transformer
  use sklearn_decomposition, only: pca,truncated_svd
  use sklearn_feature_selection, only: variance_threshold
  use sklearn_linear_model, only: linear_regression,logistic_regression,perceptron,passive_aggressive_classifier
  use sklearn_sgd, only: sgd_classifier,sgd_regressor
  use sklearn_neural_network, only: mlp_classifier,mlp_regressor
  use sklearn_neighbors, only: kneighbors_classifier,kneighbors_regressor
  use sklearn_tree, only: decision_tree_classifier,decision_tree_regressor
  use sklearn_naive_bayes, only: gaussian_nb,multinomial_nb
  use sklearn_kernel_ridge, only: kernel_ridge
  use sklearn_dummy, only: dummy_regressor
  use sklearn_composite_scores, only: regression_score,score_weights
  use sklearn_ieee_math, only: ieee_mean
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_negative_inf
  implicit none
  private
  public :: frozen_estimator,frozen_parameters
  ! Internal adapters used by pipeline; not facade algorithms.
  public :: frozen_model_fitted,frozen_model_features
  type :: frozen_parameters
    class(*), pointer :: estimator=>null()
  end type
  type, extends(dense_transformer) :: frozen_estimator
    class(*), pointer :: estimator=>null()
  contains
    procedure :: set_params=>frozen_set
    procedure :: get_params=>frozen_get
    procedure :: get_estimator=>frozen_target
    procedure :: clone=>frozen_clone
    procedure :: fit=>frozen_fit
    procedure :: fit_data=>frozen_fit_data
    procedure :: is_fitted=>frozen_fitted
    procedure :: has_method=>frozen_supports
    procedure :: bridge_supports=>frozen_supports
    procedure :: bridge_kind=>frozen_kind
    procedure :: bridge_features=>frozen_features
    procedure :: can_transform=>frozen_can_transform
    procedure :: map_data=>frozen_transform
    procedure :: transform=>frozen_transform
    procedure :: can_inverse=>frozen_can_inverse
    procedure :: inverse_data=>frozen_inverse
    procedure :: inverse_transform=>frozen_inverse
    procedure :: feature_names=>frozen_names
    procedure :: get_feature_names_out=>frozen_names
    procedure :: predict=>frozen_predict
    procedure :: bridge_predict=>frozen_predict
    procedure :: predict_class=>frozen_labels
    procedure :: bridge_labels=>frozen_labels
    procedure :: predict_matrix=>frozen_matrix
    procedure :: predict_multioutput=>frozen_matrix
    procedure :: bridge_matrix=>frozen_matrix
    procedure :: predict_indicator=>frozen_indicator
    procedure :: predict_proba=>frozen_proba
    procedure :: bridge_proba=>frozen_proba
    procedure :: predict_log_proba=>frozen_log_proba
    procedure :: bridge_log_proba=>frozen_log_proba
    procedure :: decision_function=>frozen_decision
    procedure :: bridge_decision=>frozen_decision
    procedure :: get_classes=>frozen_classes
    procedure :: bridge_classes=>frozen_classes
    procedure :: bridge_score_real=>frozen_score_real
    procedure :: bridge_score_integer=>frozen_score_integer
    procedure :: bridge_score_matrix=>frozen_score_matrix
    procedure :: bridge_score_unsupervised=>frozen_score_unsupervised
    procedure :: score_samples=>frozen_score_samples
    procedure :: bridge_score_samples=>frozen_score_samples
    procedure, private :: score_indicator=>frozen_score_indicator
    generic :: score=>bridge_score_real,bridge_score_integer,bridge_score_matrix,score_indicator,bridge_score_unsupervised
  end type
contains
  subroutine frozen_set(self,estimator)
    class(frozen_estimator), intent(inout) :: self
    class(*), target, optional, intent(inout) :: estimator
    if(present(estimator)) self%estimator=>estimator
  end subroutine
  function frozen_get(self,deep) result(params)
    class(frozen_estimator), intent(in) :: self
    logical, optional, intent(in) :: deep
    type(frozen_parameters) :: params
    params%estimator=>self%estimator
  end function
  function frozen_target(self) result(estimator)
    class(frozen_estimator), intent(in) :: self
    class(*), pointer :: estimator
    estimator=>self%estimator
  end function
  function frozen_clone(self) result(alias)
    class(frozen_estimator), target, intent(inout) :: self
    class(frozen_estimator), pointer :: alias
    alias=>self
  end function
  subroutine frozen_fit(self,x,y,sample_weight)
    class(frozen_estimator), intent(inout) :: self
    class(*), optional, intent(in) :: x(..),y(..),sample_weight(..)
    call require(self%is_fitted(),'FrozenEstimator wrapped estimator is not fitted')
  end subroutine
  subroutine frozen_fit_data(self,x,feature_names)
    class(frozen_estimator), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    type(feature_string), optional, intent(in) :: feature_names(:)
    call self%fit()
  end subroutine
  recursive logical function frozen_model_fitted(model) result(ok)
    class(*), intent(in) :: model
    ok=.false.
    select type(model)
    class is(dense_transformer);ok=model%is_fitted()
    type is(function_transformer);ok=.true.
    type is(mlp_classifier);ok=allocated(model%layers_)
    type is(mlp_regressor);ok=allocated(model%layers_)
    class is(linear_regression);ok=allocated(model%coef_)
    type is(sgd_regressor);ok=allocated(model%coef_)
    type is(kneighbors_regressor);ok=allocated(model%y_)
    class is(decision_tree_regressor);ok=allocated(model%nodes_)
    type is(kernel_ridge);ok=allocated(model%dual_coef_)
    type is(dummy_regressor);ok=model%fitted_
    type is(standard_scaler);ok=model%n_features_in_>0
    type is(minmax_scaler);ok=allocated(model%scale_)
    type is(maxabs_scaler);ok=allocated(model%scale_)
    type is(robust_scaler);ok=model%n_features_in_>0
    type is(simple_imputer);ok=allocated(model%statistics_)
    type is(one_hot_encoder);ok=allocated(model%categories_)
    type is(ordinal_encoder);ok=allocated(model%categories_)
    type is(knn_imputer);ok=allocated(model%valid_mask_)
    type is(missing_indicator);ok=allocated(model%features_)
    type is(power_transformer);ok=allocated(model%lambdas_)
    type is(quantile_transformer);ok=allocated(model%quantiles_)
    type is(kbins_discretizer);ok=allocated(model%bin_edges_)
    type is(spline_transformer);ok=allocated(model%knots_)
    type is(pca);ok=allocated(model%components_)
    type is(truncated_svd);ok=allocated(model%components_)
    type is(variance_threshold);ok=allocated(model%variances_)
    type is(column_transformer);ok=allocated(model%feature_names_in_)
    type is(feature_union);ok=allocated(model%feature_names_in_)
    class default;ok=base_fitted(model)
    end select
  end function
  recursive logical function frozen_fitted(self) result(ok)
    class(frozen_estimator), intent(in) :: self
    ok=.false.;if(associated(self%estimator)) ok=frozen_model_fitted(self%estimator)
  end function
  recursive integer function frozen_kind(self) result(kind)
    class(frozen_estimator), intent(in) :: self
    kind=0;if(.not.associated(self%estimator)) return
    select type(model=>self%estimator)
    class is(dense_transformer);kind=model%bridge_kind()
    type is(mlp_classifier);kind=1
    type is(mlp_regressor);kind=2
    class default
      if(base_supported(model,.true.)) kind=1
      if(base_supported(model,.false.)) kind=2
    end select
  end function
  recursive integer function frozen_model_features(model) result(n)
    class(*), intent(in) :: model
    n=0
    select type(model)
    class is(dense_transformer);n=model%bridge_features()
    type is(mlp_classifier);n=model%n_features_in_
    type is(mlp_regressor);n=model%n_features_in_
    class is(linear_regression);if(allocated(model%coef_)) n=size(model%coef_)
    class is(logistic_regression);if(allocated(model%coef_)) n=size(model%coef_,2)
    type is(perceptron);if(allocated(model%coef_)) n=size(model%coef_,2)
    type is(passive_aggressive_classifier);if(allocated(model%coef_)) n=size(model%coef_,2)
    type is(sgd_classifier);n=model%n_features_in_
    type is(sgd_regressor);n=model%n_features_in_
    class is(decision_tree_classifier);n=model%n_features_in_
    class is(decision_tree_regressor);n=model%n_features_in_
    type is(gaussian_nb);n=model%n_features_in_
    class is(multinomial_nb);if(allocated(model%feature_log_prob_)) n=size(model%feature_log_prob_,2)
    type is(kneighbors_classifier);if(allocated(model%search_%x_)) n=size(model%search_%x_,2)
    type is(kneighbors_regressor);if(allocated(model%search_%x_)) n=size(model%search_%x_,2)
    type is(kernel_ridge);if(allocated(model%x_fit_)) n=size(model%x_fit_,2)
    type is(function_transformer);n=model%n_features_in_
    type is(standard_scaler);n=model%n_features_in_
    type is(minmax_scaler);if(allocated(model%scale_)) n=size(model%scale_)
    type is(maxabs_scaler);if(allocated(model%scale_)) n=size(model%scale_)
    type is(robust_scaler);n=model%n_features_in_
    type is(simple_imputer);if(allocated(model%statistics_)) n=size(model%statistics_)
    type is(one_hot_encoder);if(allocated(model%categories_)) n=size(model%categories_)
    type is(ordinal_encoder);n=model%n_features_in_
    type is(knn_imputer);n=model%n_features_in_
    type is(missing_indicator);n=model%n_features_in_
    type is(power_transformer);if(allocated(model%lambdas_)) n=size(model%lambdas_)
    type is(quantile_transformer);n=model%n_features_in_
    type is(kbins_discretizer);if(allocated(model%n_bins_)) n=size(model%n_bins_)
    type is(spline_transformer);n=model%n_features_in_
    type is(pca);if(allocated(model%components_)) n=size(model%components_,2)
    type is(truncated_svd);if(allocated(model%components_)) n=size(model%components_,2)
    type is(variance_threshold);if(allocated(model%variances_)) n=size(model%variances_)
    type is(column_transformer);n=model%n_features_in_
    type is(feature_union);if(allocated(model%feature_names_in_)) n=size(model%feature_names_in_)
    end select
  end function
  recursive integer function frozen_features(self) result(n)
    class(frozen_estimator), intent(in) :: self
    n=0;if(associated(self%estimator)) n=frozen_model_features(self%estimator)
  end function
  recursive logical function frozen_can_transform(self) result(ok)
    class(frozen_estimator), intent(in) :: self
    ok=.false.;if(associated(self%estimator)) ok=supported(self%estimator)
  end function
  recursive logical function frozen_can_inverse(self) result(ok)
    class(frozen_estimator), intent(in) :: self
    ok=.false.;if(.not.associated(self%estimator)) return
    select type(model=>self%estimator)
    class is(dense_transformer);ok=model%can_inverse()
    type is(function_transformer);ok=.true.
    type is(standard_scaler);ok=.true.
    type is(minmax_scaler);ok=.true.
    type is(maxabs_scaler);ok=.true.
    type is(robust_scaler);ok=.true.
    type is(power_transformer);ok=.true.
    type is(quantile_transformer);ok=.true.
    type is(kbins_discretizer);ok=.true.
    type is(ordinal_encoder);ok=.true.
    type is(pca);ok=.true.
    type is(truncated_svd);ok=.true.
    type is(variance_threshold);ok=.true.
    end select
  end function
  recursive logical function frozen_supports(self,name) result(ok)
    class(frozen_estimator), intent(in) :: self
    character(*), intent(in) :: name
    ok=.false.
    if(name=='fit') then
      ok=.true.;return
    end if
    if(name=='fit_predict'.or.name=='fit_transform') return
    if(.not.associated(self%estimator)) return
    select type(model=>self%estimator)
    class is(dense_transformer)
      ok=model%bridge_supports(name);return
    end select
    select case(name)
    case('predict');ok=self%bridge_kind()>0
    case('score','score_samples')
      ok=self%bridge_kind()>0.and.name=='score'
      select type(model=>self%estimator)
      type is(pca);ok=.true.
      end select
    case('transform');ok=self%can_transform()
    case('get_feature_names_out')
      ok=self%can_transform()
      select type(model=>self%estimator)
      type is(function_transformer);ok=model%feature_names_out/='none'.or.associated(model%names_func)
      end select
    case('inverse_transform');ok=self%can_inverse()
    case('predict_proba','predict_log_proba')
      select type(model=>self%estimator)
      type is(mlp_classifier);ok=.true.
      type is(kneighbors_classifier);ok=name=='predict_proba'
      class default;ok=base_proba(model)
      end select
    case('decision_function');ok=base_decision(self%estimator)
    end select
  end function
  subroutine ready(self,name)
    class(frozen_estimator), intent(in) :: self
    character(*), intent(in) :: name
    call require(self%has_method(name),'FrozenEstimator wrapped estimator has no '//name)
    call require(self%is_fitted(),'FrozenEstimator wrapped estimator is not fitted')
  end subroutine
  recursive function frozen_transform(self,x) result(z)
    class(frozen_estimator), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call ready(self,'transform');z=transform_model(self%estimator,x)
  end function
  recursive function frozen_names(self,input_features) result(names)
    class(frozen_estimator), intent(in) :: self
    type(feature_string), optional, intent(in) :: input_features(:)
    type(feature_string), allocatable :: names(:)
    type(feature_string), allocatable :: input(:)
    integer :: p
    call ready(self,'get_feature_names_out');p=self%bridge_features()
    if(p==0) then
      call require(present(input_features),'frozen input feature names require explicit names')
      p=size(input_features)
    end if
    input=input_names(p,input_features);names=model_names(self%estimator,input)
  end function
  recursive function frozen_predict(self,x) result(y)
    class(frozen_estimator), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:)
    call ready(self,'predict');call require(self%bridge_kind()==2,'use predict_class for a frozen classifier')
    select type(model=>self%estimator)
    class is(dense_transformer);y=model%bridge_predict(x)
    type is(mlp_regressor);y=model%predict(x)
    class default;y=regression(model,x)
    end select
  end function
  recursive function frozen_labels(self,x) result(y)
    class(frozen_estimator), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: y(:)
    call ready(self,'predict');call require(self%bridge_kind()==1,'use predict for a frozen regressor')
    select type(model=>self%estimator)
    class is(dense_transformer);y=model%bridge_labels(x)
    type is(mlp_classifier);y=model%predict(x)
    class default;y=labels(model,x)
    end select
  end function
  recursive function frozen_matrix(self,x) result(y)
    class(frozen_estimator), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:,:),v(:)
    call ready(self,'predict');call require(self%bridge_kind()==2,'frozen matrix prediction requires a regressor')
    select type(model=>self%estimator)
    class is(dense_transformer);y=model%bridge_matrix(x)
    type is(mlp_regressor);y=model%predict_multioutput(x)
    class default;v=regression(model,x);y=reshape(v,[size(v),1])
    end select
  end function
  recursive function frozen_indicator(self,x) result(y)
    class(frozen_estimator), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: y(:,:)
    call ready(self,'predict')
    select type(model=>self%estimator)
    type is(frozen_estimator);y=model%predict_indicator(x)
    type is(mlp_classifier);y=model%predict_indicator(x)
    class default;call require(.false.,'frozen indicator prediction requires a multilabel MLP')
    end select
  end function
  recursive function frozen_proba(self,x) result(y)
    class(frozen_estimator), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:,:)
    call ready(self,'predict_proba')
    select type(model=>self%estimator)
    class is(dense_transformer);y=model%bridge_proba(x)
    type is(mlp_classifier);y=model%predict_proba(x)
    class default;y=probabilities(model,x)
    end select
  end function
  recursive function frozen_log_proba(self,x) result(y)
    class(frozen_estimator), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:,:)
    integer :: i,j
    call ready(self,'predict_log_proba')
    select type(model=>self%estimator)
    class is(dense_transformer);y=model%bridge_log_proba(x);return
    type is(gaussian_nb);y=model%predict_log_proba(x);return
    class is(multinomial_nb);y=model%predict_log_proba(x);return
    type is(sgd_classifier);y=model%predict_log_proba(x);return
    type is(mlp_classifier);y=model%predict_log_proba(x);return
    end select
    y=self%predict_proba(x)
    do j=1,size(y,2)
      do i=1,size(y,1)
        if(y(i,j)==0) then
          y(i,j)=ieee_value(0._dp,ieee_negative_inf)
        else
          y(i,j)=log(y(i,j))
        end if
      end do
    end do
  end function
  recursive function frozen_decision(self,x) result(y)
    class(frozen_estimator), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:,:)
    call ready(self,'decision_function')
    select type(model=>self%estimator)
    class is(dense_transformer);y=model%bridge_decision(x)
    class default;y=decisions(model,x)
    end select
  end function
  recursive function frozen_classes(self) result(values)
    class(frozen_estimator), intent(in) :: self
    integer, allocatable :: values(:)
    call require(self%is_fitted(),'FrozenEstimator wrapped estimator is not fitted')
    call require(self%bridge_kind()==1,'frozen estimator is not a classifier')
    select type(model=>self%estimator)
    class is(dense_transformer);values=model%bridge_classes()
    type is(mlp_classifier);values=model%classes_
    class default;values=base_classes(model)
    end select
  end function
  recursive real(dp) function frozen_score_real(self,x,y,sample_weight) result(value)
    class(frozen_estimator), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    call ready(self,'score')
    select type(model=>self%estimator)
    type is(pca)
      call require(.not.present(sample_weight),'PCA score does not accept sample weights')
      value=model%score(x);return
    end select
    if(.not.present(sample_weight)) then
      select type(model=>self%estimator)
      class is(linear_regression);value=model%score(x,y);return
      type is(sgd_regressor);value=model%score(x,y);return
      type is(kneighbors_regressor);value=model%score(x,y);return
      class is(decision_tree_regressor);value=model%score(x,y);return
      type is(kernel_ridge);value=model%score(x,y);return
      type is(dummy_regressor);value=model%score(x,y);return
      type is(mlp_regressor);value=model%score(x,y);return
      end select
    end if
    select type(model=>self%estimator)
    class is(dense_transformer);value=model%bridge_score_real(x,y,sample_weight)
    class default
      call require(self%bridge_kind()==2,'frozen classifier score requires integer labels')
      value=regression_score(y,self%predict(x),sample_weight,'FrozenEstimator')
    end select
  end function
  recursive real(dp) function frozen_score_integer(self,x,y,sample_weight) result(value)
    class(frozen_estimator), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    integer, allocatable :: predicted(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    call ready(self,'score')
    select type(model=>self%estimator)
    type is(pca)
      call require(.not.present(sample_weight),'PCA score does not accept sample weights')
      value=model%score(x);return
    end select
    if(.not.present(sample_weight)) then
      select type(model=>self%estimator)
      class is(logistic_regression);value=model%score(x,y);return
      type is(gaussian_nb);value=model%score(x,y);return
      class is(multinomial_nb);value=model%score(x,y);return
      class is(decision_tree_classifier);value=model%score(x,y);return
      type is(sgd_classifier);value=model%score(x,y);return
      type is(mlp_classifier);value=model%score(x,y);return
      end select
    end if
    select type(model=>self%estimator)
    class is(dense_transformer);value=model%bridge_score_integer(x,y,sample_weight)
    class default
      call require(self%bridge_kind()==1,'frozen regressor score requires real responses')
      call score_weights(sample_weight,size(y),'FrozenEstimator')
      predicted=self%predict_class(x)
      call require(size(predicted)==size(y),'FrozenEstimator score target length mismatch')
      value=ieee_mean(merge(1._dp,0._dp,predicted==y),sample_weight)
    end select
  end function
  recursive real(dp) function frozen_score_matrix(self,x,y,sample_weight) result(value)
    class(frozen_estimator), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    real(dp), allocatable :: pred(:,:)
    integer :: j
    call ready(self,'score')
    select type(model=>self%estimator)
    type is(pca)
      call require(.not.present(sample_weight),'PCA score does not accept sample weights')
      value=model%score(x);return
    end select
    if(.not.present(sample_weight)) then
      select type(model=>self%estimator)
      type is(mlp_regressor);value=model%score(x,y);return
      end select
    end if
    select type(model=>self%estimator)
    class is(dense_transformer);value=model%bridge_score_matrix(x,y,sample_weight)
    class default
      pred=self%predict_matrix(x);call require(size(pred,2)==size(y,2),'frozen response width mismatch')
      value=0
      do j=1,size(y,2)
        value=value+regression_score(y(:,j),pred(:,j),sample_weight,'FrozenEstimator')/size(y,2)
      end do
    end select
  end function
  recursive real(dp) function frozen_score_indicator(self,x,y,sample_weight) result(value)
    class(frozen_estimator), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, allocatable :: pred(:,:)
    if(size(y,2)==1) then
      value=frozen_score_integer(self,x,y(:,1),sample_weight);return
    end if
    pred=self%predict_indicator(x);call require(all(shape(pred)==shape(y)),'frozen indicator shape mismatch')
    call score_weights(sample_weight,size(y,1),'FrozenEstimator')
    value=ieee_mean(merge(1._dp,0._dp,all(pred==y,dim=2)),sample_weight)
  end function
  recursive function frozen_inverse(self,x) result(z)
    class(frozen_estimator), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call ready(self,'inverse_transform')
    select type(model=>self%estimator)
    class is(dense_transformer);z=model%inverse_data(x)
    type is(function_transformer);z=model%inverse_transform(x)
    type is(standard_scaler);z=model%inverse_transform(x)
    type is(minmax_scaler);z=model%inverse_transform(x)
    type is(maxabs_scaler);z=model%inverse_transform(x)
    type is(robust_scaler);z=model%inverse_transform(x)
    type is(power_transformer);z=model%inverse_transform(x)
    type is(quantile_transformer);z=model%inverse_transform(x)
    type is(kbins_discretizer);z=model%inverse_transform(x)
    type is(ordinal_encoder);z=model%inverse_transform(x)
    type is(pca);z=model%inverse_transform(x)
    type is(truncated_svd);z=model%inverse_transform(x)
    type is(variance_threshold);z=model%inverse_transform(x)
    end select
  end function
  recursive real(dp) function frozen_score_unsupervised(self,x) result(value)
    class(frozen_estimator), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    call ready(self,'score');value=0
    select type(model=>self%estimator)
    class is(dense_transformer);value=model%bridge_score_unsupervised(x)
    type is(pca);value=model%score(x)
    class default;call require(.false.,'frozen score requires a target')
    end select
  end function
  recursive function frozen_score_samples(self,x) result(values)
    class(frozen_estimator), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:)
    call ready(self,'score_samples')
    select type(model=>self%estimator)
    class is(dense_transformer);values=model%bridge_score_samples(x)
    type is(pca);values=model%score_samples(x)
    class default;call require(.false.,'frozen estimator has no score_samples')
    end select
  end function
end module
