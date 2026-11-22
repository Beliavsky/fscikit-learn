! Convenience facade; individual sklearn_* modules may also be imported.
module sklearn
  use sklearn_common, only: dp
  use sklearn_preprocessing
  use sklearn_metrics
  use sklearn_image
  use sklearn_compose
  use sklearn_pipeline
  use sklearn_inspection
  use sklearn_calibration
  use sklearn_linear_model
  use sklearn_naive_bayes
  use sklearn_neighbors
  use sklearn_cluster
  use sklearn_decomposition
  use sklearn_tree
  use sklearn_ensemble
  use sklearn_model_selection
  use sklearn_search
  use sklearn_validation, only: cross_validate,cross_val_predict,cross_val_predict_class, &
    cross_validation_options,cross_validation_result,cv_estimator,learning_curve,validation_curve, &
    permutation_test_score,learning_curve_options,learning_curve_result,validation_curve_result,permutation_score_result
  use sklearn_threshold
  use sklearn_feature_selection
  use sklearn_dummy
  use sklearn_kernel_ridge
  use sklearn_discriminant_analysis
  use sklearn_svm
  use sklearn_mixture
  use sklearn_neural_network
  use sklearn_datasets
  use sklearn_gaussian_process
  use sklearn_covariance
  use sklearn_feature_extraction
  use sklearn_feature_extraction_text
  use sklearn_kernel_pca
  use sklearn_manifold
  use sklearn_outlier
  use sklearn_semi_supervised
  use sklearn_random_projection
  use sklearn_isotonic
  use sklearn_kernel_approximation
  use sklearn_cross_decomposition
  use sklearn_multiclass
  use sklearn_multioutput
  implicit none
  public
end module
