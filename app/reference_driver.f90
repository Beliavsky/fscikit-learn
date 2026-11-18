program reference_driver
  use sklearn_common
  use sklearn_preprocessing
  use sklearn_metrics
  use sklearn_linear_model
  use sklearn_naive_bayes
  use sklearn_neighbors
  use sklearn_decomposition
  use sklearn_cluster
  use sklearn_tree
  use sklearn_ensemble
  use sklearn_feature_selection
  use sklearn_model_selection
  use sklearn_discriminant_analysis
  use sklearn_kernel_ridge
  use sklearn_svm
  use sklearn_mixture
  use sklearn_covariance
  use sklearn_gaussian_process
  use sklearn_feature_extraction_text
  implicit none
  type(standard_scaler) :: ss
  type(minmax_scaler) :: mm
  type(robust_scaler) :: rs
  type(maxabs_scaler) :: ma
  type(linear_regression) :: ols
  type(ridge) :: rr
  type(lasso) :: l1
  type(elastic_net) :: en
  type(logistic_regression) :: lr
  type(gaussian_nb) :: gn
  type(multinomial_nb) :: mn
  type(bernoulli_nb) :: bn
  type(complement_nb) :: cn
  type(kneighbors_classifier) :: knc
  type(kneighbors_regressor) :: knr
  type(pca) :: pc
  type(truncated_svd) :: sv
  type(kmeans) :: km
  type(dbscan) :: db
  type(agglomerative_clustering) :: ag
  type(decision_tree_classifier) :: tc
  type(decision_tree_regressor) :: tr
  type(random_forest_classifier) :: fc
  type(random_forest_regressor) :: fr
  type(gradient_boosting_regressor) :: gb
  type(linear_discriminant_analysis) :: lda
  type(quadratic_discriminant_analysis) :: qda
  type(kernel_ridge) :: kr
  type(svc) :: svm
  type(gaussian_mixture) :: gm
  type(ridge_cv) :: rc
  type(empirical_covariance) :: ec
  type(shrunk_covariance) :: sc
  type(ledoit_wolf) :: lw
  type(oas) :: oa
  type(gaussian_process_regressor) :: gp
  type(count_vectorizer) :: cvtext
  type(tfidf_vectorizer) :: tvtext
  character(40) :: documents(4)=[character(40):: &
    'This is the first document.','This document is the second document.', &
    'And this is the third one.','Is this the first document?']
  real(dp), allocatable :: x(:,:),y(:),positive(:,:),z(:,:),init(:,:)
  integer, allocatable :: labels(:),prediction(:)
  integer :: n,p,i,input,output
  character(1024) :: path
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p
  allocate(x(n,p),y(n),labels(n))
  do i=1,n
    read(input,*) x(i,:),y(i),labels(i)
  end do
  close(input)
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  call ss%fit(x); call matrix('standard',ss%transform(x))
  call mm%fit(x); call matrix('minmax',mm%transform(x))
  call ma%fit(x); call matrix('maxabs',ma%transform(x))
  call rs%fit(x); call matrix('robust',rs%transform(x))
  call matrix('normalize',normalize(x)); call matrix('polynomial',polynomial_features(x,3))
  call ols%fit(x,y); call vector('ols_coef',ols%coef_); call vector('ols_predict',ols%predict(x))
  rr%alpha=0.7_dp; call rr%fit(x,y); call vector('ridge',rr%predict(x))
  l1%alpha=0.07_dp; l1%tol=1.e-12_dp; call l1%fit(x,y); call vector('lasso',l1%predict(x))
  en%alpha=0.07_dp; en%tol=1.e-12_dp; call en%fit(x,y); call vector('elastic',en%predict(x))
  lr%c=1.3_dp; lr%tol=1.e-10_dp; call lr%fit(x,labels); call matrix('logistic',lr%predict_proba(x))
  call gn%fit(x,labels); call matrix('gaussian',gn%predict_proba(x))
  positive=abs(x)
  call mn%fit(positive,labels); call matrix('multinomial',mn%predict_proba(positive))
  bn%binarize_threshold=0.5_dp; call bn%fit(positive,labels); call matrix('bernoulli',bn%predict_proba(positive))
  call cn%fit(positive,labels); call matrix('complement',cn%predict_proba(positive))
  knc%n_neighbors=5; knc%weights='distance'; call knc%fit(x,labels)
  call matrix('knn_classifier',knc%predict_proba(x+0.013_dp))
  knr%n_neighbors=5; knr%weights='distance'; call knr%fit(x,y)
  call vector('knn_regressor',knr%predict(x+0.013_dp))
  pc%n_components=min(3,p); call pc%fit(x); call matrix('pca_components',pc%components_)
  call vector('pca_variance',pc%explained_variance_); call matrix('pca_transform',pc%transform(x))
  sv%n_components=min(3,p); call sv%fit(x); call vector('svd_values',sv%singular_values_)
  call matrix('euclidean',euclidean_distances(x(:7,:),x(:9,:)))
  call matrix('cosine',cosine_similarity(x(:7,:),x(:9,:)))
  call matrix('rbf',rbf_kernel(x(:7,:),x(:9,:)))
  init=x(:3,:); km%n_clusters=3; km%tol=0; call km%fit(x,init)
  call vector('kmeans_labels',real(km%labels_,dp)); call vector('kmeans_inertia',[km%inertia_])
  db%eps=1.4_dp; db%min_samples=3; call db%fit(x)
  call vector('dbscan',real(db%labels_,dp))
  ag%n_clusters=3; call ag%fit(x); call vector('ward',real(ag%labels_,dp))
  prediction=lr%predict(x)
  call vector('classification_metrics',[accuracy_score(labels,prediction), &
    f1_score(labels,prediction,'macro'),precision_score(labels,prediction,'weighted'), &
    recall_score(labels,prediction,'micro'),matthews_corrcoef(labels,prediction), &
    cohen_kappa_score(labels,prediction),balanced_accuracy_score(labels,prediction)])
  call vector('regression_metrics',[mean_squared_error(y,ols%predict(x)),mean_absolute_error(y,ols%predict(x)), &
    r2_score(y,ols%predict(x)),explained_variance_score(y,ols%predict(x)), &
    median_absolute_error(y,ols%predict(x))])
  call vector('cluster_metrics',[silhouette_score(x,km%labels_),calinski_harabasz_score(x,km%labels_), &
    davies_bouldin_score(x,km%labels_),adjusted_rand_score(labels,km%labels_), &
    rand_score(labels,km%labels_),homogeneity_score(labels,km%labels_), &
    completeness_score(labels,km%labels_),v_measure_score(labels,km%labels_)])
  tc%max_depth=4; call tc%fit(x(:,:1),labels)
  call matrix('tree_classifier',tc%predict_proba(x(:,:1)+0.013_dp))
  call vector('tree_class_importance_total',[sum(tc%feature_importances_)])
  tr%max_depth=4; call tr%fit(x,y); call vector('tree_regressor',tr%predict(x+0.013_dp))
  call vector('tree_reg_importance_total',[sum(tr%feature_importances_)])
  fc%n_estimators=3; fc%max_depth=4; fc%bootstrap=.false.; fc%max_features=1
  call fc%fit(x(:,:1),labels); call matrix('forest_classifier',fc%predict_proba(x(:,:1)+0.013_dp))
  fr%n_estimators=3; fr%max_depth=4; fr%bootstrap=.false.
  call fr%fit(x,y); call vector('forest_regressor',fr%predict(x+0.013_dp))
  gb%n_estimators=15; call gb%fit(x,y); call vector('boosting',gb%predict(x))
  call vector('f_classif',f_classif(x,labels)); call vector('f_regression',f_regression(x,y))
  call vector('chi2',chi2(positive,labels))
  call lda%fit(x,labels); call matrix('lda',lda%predict_proba(x+0.013_dp))
  qda%reg_param=0.1_dp; call qda%fit(x,labels); call matrix('qda',qda%predict_proba(x+0.013_dp))
  kr%kernel='rbf'; kr%alpha=0.3_dp; call kr%fit(x,y)
  call vector('kernel_ridge',kr%predict(x+0.013_dp))
  rc%alphas=[0.1_dp,1._dp,10._dp]; call rc%fit(x,y)
  call vector('ridge_cv',[rc%alpha_,rc%best_score_])
  svm%tol=1.e-7_dp; call svm%fit(x,labels)
  call vector('svc',real(svm%predict(x+0.013_dp),dp))
  call vector('svc_converged',[merge(1._dp,0._dp,svm%converged_)])
  gm%n_components=1; call gm%fit(x); call vector('mixture_full',gm%score_samples(x))
  gm%covariance_type='diag'; call gm%fit(x); call vector('mixture_diag',gm%score_samples(x))
  call ec%fit(x); call matrix('empirical_covariance',ec%covariance_)
  call vector('mahalanobis',ec%mahalanobis(x+0.013_dp))
  call vector('covariance_score',[ec%score(x+0.013_dp)])
  call sc%fit(x); call matrix('shrunk_covariance',sc%covariance_)
  call lw%fit(x); call matrix('ledoit_wolf',lw%covariance_); call vector('ledoit_shrinkage',[lw%shrinkage_])
  call oa%fit(x); call matrix('oas',oa%covariance_); call vector('oas_shrinkage',[oa%shrinkage_])
  gp%normalize_y=.true.; gp%alpha=0.01_dp; gp%noise_level=0.02_dp
  call gp%fit(x,y); call vector('gp_prediction',gp%predict(x+0.013_dp))
  call vector('gp_std',gp%predict_std(x+0.013_dp))
  call matrix('gp_cov',gp%predict_cov(x(:7,:)+0.013_dp))
  call vector('gp_likelihood',[gp%log_marginal_likelihood_])
  call matrix('text_counts',cvtext%fit_transform(documents))
  call matrix('text_tfidf',tvtext%fit_transform(documents))
  close(output)
contains
  subroutine vector(name,values)
    character(*), intent(in) :: name
    real(dp), intent(in) :: values(:)
    write(output,'(a,1x,*(es25.16e3,1x))') name,values
  end subroutine
  subroutine matrix(name,values)
    character(*), intent(in) :: name
    real(dp), intent(in) :: values(:,:)
    call vector(name,reshape(values,[size(values)]))
  end subroutine
end program
