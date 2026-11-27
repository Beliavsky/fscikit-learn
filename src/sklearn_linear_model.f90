module sklearn_linear_model
  use sklearn_ols_core,only:fit_ols_estimator,ols_solution
  use sklearn_ridge_estimator_core,only:fit_ridge_estimator
  use sklearn_legacy_rng,only:legacy_random_state
  use sklearn_bagging_sampling,only:sample_indices
  use sklearn_boosting_loss,only:numpy_sum=>boosting_sum
  use sklearn_huber_solver,only:huber_solve
  use sklearn_glm_lbfgs,only:glm_lbfgs_solve
  use sklearn_sparse,only:csr_matrix
  use sklearn_ridge_solver,only:ridge_regression,ridge_solver_options,ridge_solution
  use sklearn_coordinate_paths,only:enet_path,lasso_path,coordinate_path_options,coordinate_path_result
  use sklearn_omp_paths,only:orthogonal_mp,orthogonal_mp_gram,omp_result
  use sklearn_nnls, only: positive_least_squares
  use sklearn_resampling, only: index_split
  use sklearn_composite_scores, only: score_weights
  use sklearn_ieee_math, only: ieee_mean,ieee_quotient,ieee_product,ieee_total,ieee_difference
  use sklearn_omp_cv, only: orthogonal_matching_pursuit_cv
  use sklearn_least_angle, only: lars,lasso_lars,lars_path,lars_path_gram,lars_path_result, &
    lars_cv,lasso_lars_cv,lasso_lars_ic
  use sklearn_sgd_one_class, only: sgd_one_class_svm
  use sklearn_quantile, only: quantile_regressor
  use sklearn_sgd, only: sgd_regressor,sgd_classifier
  use sklearn_multitask, only: multi_task_elastic_net,multi_task_lasso,multi_task_elastic_net_cv,multi_task_lasso_cv
  use sklearn_ridge_classification, only: ridge_classifier,ridge_classifier_cv
  use sklearn_common
  use sklearn_metrics, only: r2_score,accuracy_score,roc_auc_score
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value,ieee_negative_inf
  implicit none
  private
  public::ridge_regression,ridge_solver_options,ridge_solution
  public :: orthogonal_matching_pursuit_cv
  public :: orthogonal_mp,orthogonal_mp_gram,omp_result
  public :: enet_path,lasso_path,coordinate_path_options,coordinate_path_result
  public :: lars,lasso_lars,lars_path,lars_path_gram,lars_path_result,lars_cv,lasso_lars_cv,lasso_lars_ic
  public :: sgd_one_class_svm
  public :: quantile_regressor
  public :: sgd_regressor,sgd_classifier
  public :: multi_task_elastic_net,multi_task_lasso,multi_task_elastic_net_cv,multi_task_lasso_cv
  public :: ridge_classifier,ridge_classifier_cv
  public :: elastic_net_cv,lasso_cv
  public :: linear_regression, ridge, lasso, elastic_net, logistic_regression, perceptron
  public :: logistic_regression_cv
  public :: bayesian_ridge, ard_regression, orthogonal_matching_pursuit
  public :: fit_huber_column_subset
  public :: huber_regressor,ransac_regressor,tweedie_regressor,poisson_regressor,gamma_regressor
  public :: theil_sen_regressor
  public :: passive_aggressive_classifier,passive_aggressive_regressor
  type :: pa_config
    real(dp) :: c=1,tol=1.e-3_dp,epsilon=0.1_dp
    integer :: max_iter=1000,n_iter_no_change=5,average=0
    logical :: shuffle=.true.,fit_intercept=.true.,classification=.false.,pa_two=.false.
  end type
  type :: linear_regression
    logical :: fit_intercept=.true.,positive=.false.
    real(dp), allocatable :: coef_(:)
    real(dp) :: intercept_=0
    integer :: rank_=0,n_features_in_=0,n_targets_=0
    real(dp),allocatable::coef_matrix_(:,:),intercepts_(:),singular_(:)
    integer,allocatable::lsqr_iterations_(:),lsqr_stop_code_(:)
    logical::matrix_target_=.false.
    character(8)::least_squares_solver_=''
  contains
    procedure :: fit_dense=>ols_fit
    procedure :: fit_dense_weighted=>ols_fit_dw,fit_sparse=>ols_fit_sv,fit_matrix=>ols_fit_dm
    procedure :: fit_sparse_matrix=>ols_fit_sm
    procedure :: fit_scalar_dw=>ols_fit_dws,fit_scalar_dm=>ols_fit_dms
    procedure :: fit_scalar_sv=>ols_fit_svs,fit_scalar_sm=>ols_fit_sms
    generic :: fit=>fit_dense,fit_dense_weighted,fit_sparse,fit_matrix,fit_sparse_matrix, &
      fit_scalar_dw,fit_scalar_dm,fit_scalar_sv,fit_scalar_sm
    procedure :: predict_dense=>linear_predict,predict_sparse=>ols_predict_sparse
    generic :: predict=>predict_dense,predict_sparse
    procedure :: predict_matrix_dense=>ols_predict_dm,predict_matrix_sparse=>ols_predict_sm
    generic :: predict_matrix=>predict_matrix_dense,predict_matrix_sparse
    procedure :: score_dense=>linear_score,score_dw=>ols_score_dw,score_sv=>ols_score_sv
    procedure :: score_dm=>ols_score_dm,score_sm=>ols_score_sm
    generic :: score=>score_dense,score_dw,score_sv,score_dm,score_sm
  end type
  type, extends(linear_regression) :: ridge
    real(dp) :: alpha=1,tol=1.e-4_dp
    integer :: max_iter=-1,random_state=0
    character(16) :: solver='auto',solver_=''
    real(dp),allocatable::alpha_per_target(:)
    integer,allocatable::n_iter_(:),native_iterations_(:),stop_code_(:)
    logical,allocatable::converged_per_target_(:)
    logical::converged_=.false.
  contains
    procedure :: fit_dense=>ridge_fit
    procedure :: fit_dense_weighted=>ridge_fit_dw,fit_sparse=>ridge_fit_sv,fit_matrix=>ridge_fit_dm
    procedure :: fit_sparse_matrix=>ridge_fit_sm
    procedure :: fit_scalar_dw=>ridge_fit_dws,fit_scalar_dm=>ridge_fit_dms
    procedure :: fit_scalar_sv=>ridge_fit_svs,fit_scalar_sm=>ridge_fit_sms
    procedure :: predict_dense=>ridge_predict_dense,predict_sparse=>ridge_predict_sparse
    procedure :: predict_matrix_dense=>ridge_predict_dm,predict_matrix_sparse=>ridge_predict_sm
    procedure :: score_dense=>ridge_score_dense,score_dw=>ridge_score_dw,score_sv=>ridge_score_sv
    procedure :: score_dm=>ridge_score_dm,score_sm=>ridge_score_sm
  end type
  type, extends(linear_regression) :: elastic_net
    real(dp) :: alpha=1, l1_ratio=0.5_dp,tol=1.e-8_dp
    integer :: max_iter=10000,n_iter_=0
    logical :: converged_=.false.
  contains
    procedure :: fit_dense=>elastic_fit
  end type
  type, extends(elastic_net) :: lasso
  contains
    procedure :: fit_dense=>lasso_fit
  end type
  ! CV contracts: sklearn 1.6.1 linear_model/_coordinate_descent.py (BSD).
  type, extends(elastic_net) :: elastic_net_cv
    integer :: cv=5,n_alphas=100
    real(dp) :: eps=1.e-3_dp,alpha_=0,l1_ratio_=0,best_mse_=0
    real(dp), allocatable :: alphas(:),l1_ratios(:),alphas_(:,:),mse_path_(:,:,:)
    logical :: all_cv_converged_=.false.
  contains
    procedure :: fit_dense=>elastic_cv_fit
  end type
  type, extends(elastic_net_cv) :: lasso_cv
  contains
    procedure :: fit_dense=>lasso_cv_fit
  end type
  type :: logistic_regression
    logical :: fit_intercept=.true., converged_=.false.
    real(dp) :: c=1, tol=1.e-8_dp
    integer :: max_iter=200,n_iter_=0
    integer, allocatable :: classes_(:)
    real(dp), allocatable :: coef_(:,:),intercept_(:)
  contains
    procedure :: fit=>logistic_fit
    procedure :: decision_function=>logistic_decision
    procedure :: predict_proba=>logistic_proba
    procedure :: predict=>logistic_predict
    procedure :: score=>logistic_score
  end type
  ! Dense L2 LogisticRegressionCV (sklearn 1.6.1 _logistic.py, BSD-3-Clause).
  type, extends(logistic_regression) :: logistic_regression_cv
    integer :: cv=5,n_cs=10,n_features_in_=0
    logical :: refit=.true.,class_weight_balanced=.false.,all_cv_converged_=.false.
    character(20) :: scoring='accuracy'
    real(dp), allocatable :: cs(:),cs_(:),c_(:),class_weight(:),class_weight_(:),scores_(:,:)
    real(dp), allocatable :: coefs_paths_(:,:,:,:),refit_init_(:,:)
    integer, allocatable :: path_classes_(:),best_c_indices_(:),n_iter_path_(:,:)
    logical, allocatable :: converged_path_(:,:)
    type(index_split), allocatable :: splits(:),cv_splits_(:)
  contains
    procedure :: fit=>logistic_cv_fit
    procedure :: score=>logistic_cv_score
  end type
  type :: perceptron
    logical :: fit_intercept=.true.
    integer :: max_iter=1000, n_iter_=0
    real(dp) :: eta0=1
    integer, allocatable :: classes_(:)
    real(dp), allocatable :: coef_(:,:),intercept_(:)
  contains
    procedure :: fit=>perceptron_fit
    procedure :: decision_function=>perceptron_decision
    procedure :: predict=>perceptron_predict
    procedure :: score=>perceptron_score
  end type
  type :: passive_aggressive_classifier
    real(dp) :: c=1,tol=1.e-3_dp,t_=1
    integer :: max_iter=1000,n_iter_no_change=5,average=0,random_state=0,n_iter_=0
    character(16) :: loss='hinge'
    logical :: fit_intercept=.true.,shuffle=.true.,class_weight_balanced=.false.,converged_=.false.
    integer, allocatable :: classes_(:),n_iter_per_class_(:)
    real(dp), allocatable :: coef_(:,:),intercept_(:),loss_(:)
  contains
    procedure :: fit=>pa_classifier_fit
    procedure :: decision_function=>pa_classifier_decision
    procedure :: predict=>pa_classifier_predict
    procedure :: score=>pa_classifier_score
  end type
  type, extends(linear_regression) :: passive_aggressive_regressor
    real(dp) :: c=1,tol=1.e-3_dp,epsilon=0.1_dp,t_=1,loss_=0
    integer :: max_iter=1000,n_iter_no_change=5,average=0,random_state=0,n_iter_=0
    character(32) :: loss='epsilon_insensitive'
    logical :: shuffle=.true.,converged_=.false.
  contains
    procedure :: fit_dense=>pa_regressor_fit
  end type
  type, extends(linear_regression) :: bayesian_ridge
    integer :: max_iter=300,n_iter_=0
    real(dp) :: tol=1.e-3_dp,alpha_1=1.e-6_dp,alpha_2=1.e-6_dp
    real(dp) :: lambda_1=1.e-6_dp,lambda_2=1.e-6_dp,alpha_init=-1,lambda_init=1
    real(dp) :: alpha_=0,lambda_=0
    logical :: converged_=.false.
    real(dp), allocatable :: sigma_(:,:)
  contains
    procedure :: fit_dense=>bayesian_fit
    procedure :: predict_std=>bayesian_std
  end type
  type, extends(linear_regression) :: ard_regression
    integer :: max_iter=300,n_iter_=0
    real(dp) :: tol=1.e-3_dp,alpha_1=1.e-6_dp,alpha_2=1.e-6_dp
    real(dp) :: lambda_1=1.e-6_dp,lambda_2=1.e-6_dp,threshold_lambda=1.e4_dp,alpha_=0
    logical :: converged_=.false.
    real(dp), allocatable :: lambda_(:),sigma_(:,:)
  contains
    procedure :: fit_dense=>ard_fit
    procedure :: predict_std=>ard_std
  end type
  type, extends(linear_regression) :: orthogonal_matching_pursuit
    integer :: n_nonzero_coefs=0,n_iter_=0
    real(dp) :: tol=-1
  contains
    procedure :: fit_dense=>omp_fit
  end type
  type, extends(linear_regression) :: huber_regressor
    real(dp) :: epsilon=1.35_dp,alpha=1.e-4_dp,tol=1.e-5_dp,scale_=1,loss_=0
    integer :: max_iter=100,n_iter_=0
    logical :: converged_=.false.,warm_start=.false.
    character(60)::optimization_message_=''
    logical, allocatable :: outliers_(:)
  contains
    procedure :: fit_dense=>huber_fit
    procedure :: fit_dense_weighted=>huber_fit_dw,fit_scalar_dw=>huber_fit_dws
    procedure :: fit_sparse=>huber_fit_sv,fit_scalar_sv=>huber_fit_svs
    procedure :: fit_matrix=>huber_fit_dm,fit_scalar_dm=>huber_fit_dms
    procedure :: fit_sparse_matrix=>huber_fit_sm,fit_scalar_sm=>huber_fit_sms
    procedure :: predict_sparse=>huber_predict_sparse
    procedure :: score_dw=>huber_score_dw,score_sv=>huber_score_sv
    procedure :: score_dm=>huber_score_dm,score_sm=>huber_score_sm
  end type
  type, extends(linear_regression) :: ransac_regressor
    integer :: min_samples=0,max_trials=100,random_state=0,stop_n_inliers=huge(1),max_skips=huge(1)
    integer :: n_trials_=0,n_skips_no_inliers_=0,n_skips_invalid_data_=0,n_skips_invalid_model_=0
    integer :: min_samples_=0
    real(dp) :: residual_threshold=-1,stop_probability=.99_dp,stop_score=huge(1._dp),residual_threshold_=0
    character(20) :: loss='absolute_error'
    logical, allocatable :: inlier_mask_(:)
    logical::skips_exceeded_=.false.
    type(linear_regression)::estimator_
    logical,private::fractional_min_samples_=.false.,explicit_min_samples_=.false.
    real(dp),private::min_samples_fraction_=0
    procedure(ransac_data_check),pointer,nopass::is_data_valid=>null()
    procedure(ransac_model_check),pointer,nopass::is_model_valid=>null()
    procedure(ransac_loss_callback),pointer,nopass::loss_callback=>null()
  contains
    procedure :: fit_dense=>ransac_fit
    procedure :: fit_dense_weighted=>ransac_fit_dw,fit_scalar_dw=>ransac_fit_dws
    procedure :: fit_sparse=>ransac_fit_sv,fit_scalar_sv=>ransac_fit_svs
    procedure :: fit_matrix=>ransac_fit_dm,fit_scalar_dm=>ransac_fit_dms
    procedure :: fit_sparse_matrix=>ransac_fit_sm,fit_scalar_sm=>ransac_fit_sms
    procedure :: predict_sparse=>ransac_predict_sv
    procedure :: predict_matrix_dense=>ransac_predict_dm,predict_matrix_sparse=>ransac_predict_sm
    procedure :: score_dw=>ransac_score_dw,score_sv=>ransac_score_sv
    procedure :: score_dm=>ransac_score_dm,score_sm=>ransac_score_sm
    procedure,private::ransac_set_min_integer,ransac_set_min_fraction
    generic::set_min_samples=>ransac_set_min_integer,ransac_set_min_fraction
    procedure::clear_min_samples=>ransac_clear_min
  end type
  type, extends(linear_regression) :: theil_sen_regressor
    integer :: n_subsamples=0,max_subpopulation=10000,max_iter=300,random_state=0
    integer :: n_subsamples_=0,n_subpopulation_=0,n_iter_=0
    real(dp) :: tol=1.e-3_dp,breakdown_=0
    logical :: exhaustive_=.false.,converged_=.false.
    integer, allocatable :: subsets_(:,:)
  contains
    procedure :: fit_dense=>theil_sen_fit
  end type
  type, extends(linear_regression) :: tweedie_regressor
    real(dp) :: power=0,power_=0,alpha=1,tol=1.e-4_dp
    integer :: max_iter=100,n_iter_=0
    character(12) :: link='auto',link_='identity'
    character(16)::solver='lbfgs'
    character(60)::optimization_message_=''
    logical :: converged_=.false.,warm_start=.false.
  contains
    procedure :: fit_dense=>tweedie_fit
    procedure :: fit_dense_weighted=>glm_fit_dw,fit_scalar_dw=>glm_fit_dws
    procedure :: fit_matrix=>glm_fit_dm,fit_scalar_dm=>glm_fit_dms
    procedure :: score_dw=>glm_score_dw,score_dm=>glm_score_dm
    procedure :: fit_sparse=>glm_fit_sv,fit_scalar_sv=>glm_fit_svs
    procedure :: fit_sparse_matrix=>glm_fit_sm,fit_scalar_sm=>glm_fit_sms
    procedure :: predict_sparse=>glm_predict_sparse,score_sv=>glm_score_sv,score_sm=>glm_score_sm
    procedure :: predict_dense=>glm_predict
    procedure :: score_dense=>glm_score
  end type
  type, extends(tweedie_regressor) :: poisson_regressor
  contains
    procedure :: fit_dense=>poisson_fit
  end type
  type, extends(tweedie_regressor) :: gamma_regressor
  contains
    procedure :: fit_dense=>gamma_fit
  end type
  abstract interface
    logical function ransac_data_check(x,y)
      import dp
      real(dp),intent(in)::x(:,:),y(:,:)
    end function
    logical function ransac_model_check(model,x,y)
      import dp,linear_regression
      type(linear_regression),intent(in)::model
      real(dp),intent(in)::x(:,:),y(:,:)
    end function
    function ransac_loss_callback(y_true,y_pred)result(residual)
      import dp
      real(dp),intent(in)::y_true(:,:),y_pred(:,:)
      real(dp)::residual(size(y_true,1))
    end function
  end interface

contains
  subroutine check_positive_regression(self)
    class(linear_regression), intent(in) :: self
    if(.not.self%positive) return
    select type(self)
    type is(linear_regression)
    type is(ridge)
    class is(elastic_net)
    class default
      call require(.false.,'positive coefficients are unsupported for this regressor')
    end select
  end subroutine
  subroutine positive_data_check(x,y)
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp) :: limit
    call check_xy(x,size(y));call check_x(reshape(y,[size(y),1]))
    limit=sqrt(huge(1._dp))/(256*real(max(size(x,1),size(x,2)),dp))
    call require(maxval(abs(x))<=limit.and.maxval(abs(y))<=limit,'positive linear data exceeds working range')
  end subroutine
  real(dp) function positive_dot(a,b) result(value)
    real(dp), intent(in) :: a(:),b(:)
    real(dp) :: products(size(a))
    integer :: j
    do j=1,size(a)
      products(j)=ieee_product(a(j),b(j))
    end do
    value=ieee_total(products)
    call require(ieee_is_finite(value),'positive linear product exceeds working range')
  end function
  ! sklearn 1.6.1 linear_model/_passive_aggressive.py, _stochastic_gradient.py
  ! and _sgd_fast.pyx.tp. Retained BSD terms: LICENSES/scikit-learn-COPYING.
  function pa_settings(c,tol,max_iter,patience,average,shuffle,fit_intercept) result(options)
    real(dp), intent(in) :: c,tol
    integer, intent(in) :: max_iter,patience,average
    logical, intent(in) :: shuffle,fit_intercept
    type(pa_config) :: options
    call require(all(ieee_is_finite([c,tol])),'nonfinite passive-aggressive controls')
    call require(c>0,'passive-aggressive C must be positive')
    call require(tol>=0.or.tol== -1,'passive-aggressive tol must be nonnegative or -1')
    call require(max_iter>0.and.patience>0,'invalid passive-aggressive iteration controls')
    call require(average>=0,'passive-aggressive average must be nonnegative')
    options%c=c; options%tol=tol; options%max_iter=max_iter; options%n_iter_no_change=patience
    options%average=average; options%shuffle=shuffle; options%fit_intercept=fit_intercept
  end function

  subroutine pa_run(x,y,options,weight_pos,weight_neg,coef,intercept,n_iter,converged,last_loss)
    real(dp), intent(in) :: x(:,:),y(:),weight_pos,weight_neg
    type(pa_config), intent(in) :: options
    real(dp), intent(out) :: coef(:),intercept,last_loss
    integer, intent(out) :: n_iter
    logical, intent(out) :: converged
    real(dp) :: current(size(coef)),average_coef(size(coef)),bias,average_bias,prediction,loss,update,norm
    real(dp) :: sumloss,best_loss,coin,seen,average_count
    integer :: order(size(y)),i,j,it,pick,tmp,stale,n
    n=size(y); current=0; average_coef=0; bias=0; average_bias=0
    best_loss=huge(1._dp); stale=0; seen=1; converged=.false.; order=[(i,i=1,n)]
    do it=1,options%max_iter
      if(options%shuffle) then
        do j=n,2,-1
          call random_number(coin); pick=min(j,1+int(coin*j))
          tmp=order(j); order(j)=order(pick); order(pick)=tmp
        end do
      end if
      sumloss=0
      do j=1,n
        i=order(j); prediction=dot_product(current,x(i,:))+bias
        if(options%classification) then
          loss=max(0._dp,1-y(i)*prediction)
        else
          loss=max(0._dp,abs(y(i)-prediction)-options%epsilon)
        end if
        sumloss=sumloss+loss; norm=sum(x(i,:)**2)
        if(options%pa_two) then
          update=loss/(norm+0.5_dp/options%c)
        else
          ! sklearn PA-I skips zero-norm rows, including intercept/averaging.
          if(norm==0) cycle
          update=min(options%c,loss/norm)
        end if
        if(options%classification) then
          update=update*y(i)
        else
          if(y(i)-prediction<0) update=-update
        end if
        if(y(i)>0) then
          update=update*weight_pos
        else
          update=update*weight_neg
        end if
        current=current+update*x(i,:)
        if(options%fit_intercept) bias=bias+update
        if(options%average>0.and.seen>=options%average) then
          average_count=seen-options%average+1
          average_coef=average_coef+(current-average_coef)/average_count
          average_bias=average_bias+(bias-average_bias)/average_count
        end if
        seen=seen+1
      end do
      call require(all(ieee_is_finite(current)).and.ieee_is_finite(bias), &
        'passive-aggressive fit overflow; scale the inputs')
      if(options%tol>=0) then
        if(sumloss>best_loss-options%tol*n) then
          stale=stale+1
        else
          stale=0
        end if
      end if
      best_loss=min(best_loss,sumloss)
      if(stale>=options%n_iter_no_change) then
        converged=.true.; exit
      end if
    end do
    n_iter=min(it,options%max_iter); last_loss=sumloss/n; coef=current; intercept=bias
    ! The public averaging selection uses epoch sample counts, as sklearn does.
    if(options%average>0.and.real(options%average,dp)<=real(n_iter,dp)*n) then
      coef=average_coef; intercept=average_bias
    end if
  end subroutine

  subroutine pa_regressor_fit(self,x,y)
    class(passive_aggressive_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    type(pa_config) :: options
    call check_positive_regression(self)
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    options=pa_settings(self%c,self%tol,self%max_iter,self%n_iter_no_change,self%average,self%shuffle,self%fit_intercept)
    call require(ieee_is_finite(self%epsilon),'nonfinite passive-aggressive epsilon')
    call require(self%epsilon>=0,'passive-aggressive epsilon must be nonnegative')
    call require(self%loss=='epsilon_insensitive'.or.self%loss=='squared_epsilon_insensitive', &
      'invalid passive-aggressive regression loss')
    options%epsilon=self%epsilon; options%pa_two=self%loss=='squared_epsilon_insensitive'
    if(self%shuffle) call seed_rng(self%random_state)
    self%coef_=spread(0._dp,1,size(x,2))
    call pa_run(x,y,options,1._dp,1._dp,self%coef_,self%intercept_,self%n_iter_,self%converged_,self%loss_)
    self%t_=1+real(self%n_iter_,dp)*size(y); self%rank_=0
  end subroutine

  subroutine pa_classifier_fit(self,x,y)
    class(passive_aggressive_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    type(pa_config) :: options
    real(dp) :: target(size(y)),positive_weight,negative_weight
    real(dp), allocatable :: weights(:)
    integer :: k,heads,j,positive
    logical :: converged
    call check_xy(x,size(y))
    options=pa_settings(self%c,self%tol,self%max_iter,self%n_iter_no_change,self%average,self%shuffle,self%fit_intercept)
    call require(self%loss=='hinge'.or.self%loss=='squared_hinge','invalid passive-aggressive classification loss')
    self%classes_=unique_labels(y); k=size(self%classes_)
    call require(k>=2,'passive-aggressive classification requires at least two classes')
    heads=k; if(k==2) heads=1
    if(allocated(self%coef_)) deallocate(self%coef_,self%intercept_,self%n_iter_per_class_,self%loss_)
    allocate(self%coef_(heads,size(x,2)),self%intercept_(heads),self%n_iter_per_class_(heads),self%loss_(heads))
    options%classification=.true.; options%pa_two=self%loss=='squared_hinge'
    weights=spread(1._dp,1,k)
    if(self%class_weight_balanced) then
      do j=1,k
        weights(j)=real(size(y),dp)/(real(k,dp)*count(y==self%classes_(j)))
      end do
    end if
    if(self%shuffle) call seed_rng(self%random_state)
    self%converged_=.true.
    do j=1,heads
      positive=j; if(k==2) positive=2
      target=merge(1._dp,-1._dp,y==self%classes_(positive))
      positive_weight=weights(positive); negative_weight=1
      if(k==2) negative_weight=weights(1)
      call pa_run(x,target,options,positive_weight,negative_weight,self%coef_(j,:),self%intercept_(j), &
        self%n_iter_per_class_(j),converged,self%loss_(j))
      self%converged_=self%converged_.and.converged
    end do
    self%n_iter_=maxval(self%n_iter_per_class_); self%t_=1+real(self%n_iter_,dp)*size(y)
    if(self%average>0.and.real(self%average,dp)<=self%t_-1) then
      do j=1,heads
        if(real(self%average,dp)>real(self%n_iter_per_class_(j),dp)*size(y)) then
          self%coef_(j,:)=0; self%intercept_(j)=0
        end if
      end do
    end if
  end subroutine

  function pa_classifier_decision(self,x) result(z)
    class(passive_aggressive_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call require(allocated(self%coef_),'passive-aggressive classifier is not fitted')
    call check_xy(transpose(x),size(self%coef_,2))
    z=matmul(x,transpose(self%coef_))+spread(self%intercept_,1,size(x,1))
  end function

  function pa_classifier_predict(self,x) result(y)
    class(passive_aggressive_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: y(size(x,1)),i
    real(dp), allocatable :: z(:,:)
    z=self%decision_function(x)
    if(size(self%classes_)==2) then
      y=merge(self%classes_(2),self%classes_(1),z(:,1)>0)
    else
      do i=1,size(y)
        y(i)=self%classes_(maxloc(z(i,:),dim=1))
      end do
    end if
  end function

  real(dp) function pa_classifier_score(self,x,y) result(value)
    class(passive_aggressive_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    value=accuracy_score(y,self%predict(x))
  end function

  ! sklearn 1.6.1 linear_model/_theil_sen.py, BSD-3-Clause.
  ! See LICENSES/scikit-learn-COPYING and THIRD_PARTY_NOTICES.md.
  subroutine theil_spatial_median(weights,max_iter,tol,center,n_iter,converged)
    real(dp), intent(in) :: weights(:,:),tol
    integer, intent(in) :: max_iter
    real(dp), intent(out) :: center(:)
    integer, intent(out) :: n_iter
    logical, intent(out) :: converged
    real(dp) :: old(size(center)),direction(size(center)),quotient(size(center)),diff(size(center))
    real(dp) :: norm,quotient_norm,denominator,mix
    integer :: i,it
    logical :: coincident
    if(size(center)==1) then
      center(1)=sample_median(weights(:,1)); n_iter=1; converged=.true.; return
    end if
    old=sum(weights,dim=1)/size(weights,1); converged=.false.
    do it=1,max_iter
      quotient=0; direction=0; denominator=0; coincident=.false.
      do i=1,size(weights,1)
        diff=weights(i,:)-old; norm=sqrt(sum(diff**2))
        if(norm>=epsilon(1._dp)) then
          quotient=quotient+diff/norm
          direction=direction+weights(i,:)/norm; denominator=denominator+1/norm
        else
          coincident=.true.
        end if
      end do
      quotient_norm=sqrt(sum(quotient**2))
      if(quotient_norm>epsilon(1._dp)) then
        direction=direction/denominator
      else
        ! Preserve sklearn 1.6.1's modified-Weiszfeld zero-quotient convention.
        direction=1; quotient_norm=1
      end if
      mix=0
      if(coincident) mix=min(1._dp,1/quotient_norm)
      center=(1-mix)*direction+mix*old
      n_iter=it-1 ! sklearn reports the zero-based last Weiszfeld iteration.
      if(sum((center-old)**2)<tol**2) then
        converged=.true.; exit
      end if
      old=center
    end do
  end subroutine

  subroutine theil_sen_fit(self,x,y)
    class(theil_sen_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), allocatable :: design(:,:),weights(:,:),s(:),u(:,:),v(:,:),center(:)
    real(dp) :: combinations,coin,cutoff
    integer :: n,p,q,m,k,i,j,h,pick,tmp,first
    integer, allocatable :: subset(:),pool(:)
    call check_positive_regression(self)
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    call require(ieee_is_finite(self%tol),'nonfinite TheilSen tolerance')
    call require(self%tol>0.and.self%max_iter>0,'invalid TheilSen iteration controls')
    call require(self%max_subpopulation>0,'TheilSen max_subpopulation must be positive')
    call require(self%n_subsamples>=0,'invalid TheilSen n_subsamples')
    n=size(x,1); p=size(x,2); q=p+merge(1,0,self%fit_intercept); m=self%n_subsamples
    if(m==0) then
      m=min(q,n)
    else
      call require(m<=n,'TheilSen n_subsamples exceeds sample count')
      if(n>=p) then
        call require(m>=q,'TheilSen n_subsamples is smaller than design dimension')
      else
        call require(m==n,'underdetermined TheilSen requires all samples')
      end if
    end if
    self%n_subsamples_=m
    self%breakdown_=1-(0.5_dp**(1._dp/m)*(n-m+1)+m-1)/n
    ! Bound the binomial count before integer conversion to avoid overflow.
    combinations=1
    do j=1,min(m,n-m)
      combinations=combinations*real(n-min(m,n-m)+j,dp)/j
      if(combinations>real(self%max_subpopulation,dp)+0.5_dp) exit
    end do
    self%exhaustive_=combinations<=real(self%max_subpopulation,dp)+0.5_dp
    self%n_subpopulation_=self%max_subpopulation
    if(self%exhaustive_) self%n_subpopulation_=max(1,nint(combinations))
    if(allocated(self%subsets_)) deallocate(self%subsets_)
    allocate(self%subsets_(self%n_subpopulation_,m),subset(m),pool(n))
    if(self%exhaustive_) then
      subset=[(i,i=1,m)]
      do i=1,self%n_subpopulation_
        self%subsets_(i,:)=subset
        if(i==self%n_subpopulation_) exit
        j=m
        do while(j>0)
          if(subset(j)<n-m+j) exit
          j=j-1
        end do
        call require(j>0,'TheilSen combination enumeration failed')
        subset(j)=subset(j)+1
        do h=j+1,m
          subset(h)=subset(h-1)+1
        end do
      end do
    else
      call seed_rng(self%random_state)
      do i=1,self%n_subpopulation_
        pool=[(j,j=1,n)]
        do j=1,m
          call random_number(coin); pick=min(n,j+int(coin*(n-j+1)))
          tmp=pool(j); pool(j)=pool(pick); pool(pick)=tmp
        end do
        self%subsets_(i,:)=pool(:m)
      end do
    end if
    allocate(design(m,q),weights(self%n_subpopulation_,q),s(q),u(m,q),v(q,q),center(q))
    first=merge(2,1,self%fit_intercept)
    do i=1,self%n_subpopulation_
      subset=self%subsets_(i,:); design(:,first:)=x(subset,:)
      if(self%fit_intercept) design(:,1)=1
      call dense_svd(design,s,u,v); weights(i,:)=0
      cutoff=epsilon(1._dp)*max(m,q)*maxval(s)
      do k=1,q
        if(s(k)>cutoff) weights(i,:)=weights(i,:)+v(:,k)*dot_product(u(:,k),y(subset))/s(k)
      end do
    end do
    call theil_spatial_median(weights,self%max_iter,self%tol,center,self%n_iter_,self%converged_)
    self%intercept_=0; self%coef_=center(first:); self%rank_=0
    if(self%fit_intercept) self%intercept_=center(1)
  end subroutine

  ! sklearn 1.6.1 _huber.py, _ransac.py and _glm/glm.py are the references
  ! for these objectives and estimator contracts. See THIRD_PARTY_NOTICES.md.
  subroutine huber_fit(self,x,y)
    class(huber_regressor),intent(inout)::self
    real(dp),intent(in)::x(:,:),y(:)
    call huber_fit_common(self,y,dense=x)
  end subroutine
  subroutine huber_fit_dw(self,x,y,sample_weight)
    class(huber_regressor),intent(inout)::self
    real(dp),intent(in)::x(:,:),y(:),sample_weight(:)
    call huber_fit_common(self,y,sample_weight,dense=x)
  end subroutine
  subroutine huber_fit_dws(self,x,y,sample_weight)
    class(huber_regressor),intent(inout)::self
    real(dp),intent(in)::x(:,:),y(:),sample_weight
    call huber_fit_common(self,y,spread(sample_weight,1,size(y)),dense=x)
  end subroutine
  subroutine huber_fit_sv(self,x,y,sample_weight)
    class(huber_regressor),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    call huber_fit_common(self,y,sample_weight,sparse=x)
  end subroutine
  subroutine huber_fit_svs(self,x,y,sample_weight)
    class(huber_regressor),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:),sample_weight
    call huber_fit_common(self,y,spread(sample_weight,1,size(y)),sparse=x)
  end subroutine
  subroutine huber_fit_dm(self,x,y,sample_weight)
    class(huber_regressor),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    call require(size(y,2)==1,'Huber requires one target column')
    call huber_fit_common(self,y(:,1),sample_weight,dense=x)
  end subroutine
  subroutine huber_fit_dms(self,x,y,sample_weight)
    class(huber_regressor),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:,:),sample_weight
    call require(size(y,2)==1,'Huber requires one target column')
    call huber_fit_common(self,y(:,1),spread(sample_weight,1,size(y,1)),dense=x)
  end subroutine
  subroutine huber_fit_sm(self,x,y,sample_weight)
    class(huber_regressor),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    call require(size(y,2)==1,'Huber requires one target column')
    call huber_fit_common(self,y(:,1),sample_weight,sparse=x)
  end subroutine
  subroutine huber_fit_sms(self,x,y,sample_weight)
    class(huber_regressor),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:,:),sample_weight
    call require(size(y,2)==1,'Huber requires one target column')
    call huber_fit_common(self,y(:,1),spread(sample_weight,1,size(y,1)),sparse=x)
  end subroutine
  ! Internal bagging adapter: NumPy advanced column indexing produces an
  ! F-contiguous dense design, which changes BLAS reduction order.
  subroutine fit_huber_column_subset(self,x,y,sample_weight)
    type(huber_regressor),intent(inout)::self
    real(dp),intent(in)::x(:,:),y(:),sample_weight(:)
    call huber_fit_common(self,y,sample_weight,dense=x,dense_fortran=.true.)
  end subroutine
  subroutine huber_fit_common(self,y,sample_weight,dense,sparse,dense_fortran)
    class(huber_regressor),intent(inout)::self
    real(dp),intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:),dense(:,:)
    type(csr_matrix),optional,intent(in)::sparse
    logical,optional,intent(in)::dense_fortran
    real(dp),allocatable::point(:),weight(:),pred(:,:)
    integer::p,q
    call check_positive_regression(self)
    p=0;if(present(dense))p=size(dense,2)
    if(present(sparse))p=sparse%n_cols
    if(self%warm_start.and.allocated(self%coef_))then
      ! Source always includes the previous intercept in a warm initial point,
      ! even when the public fit_intercept flag is false.
      point=[self%coef_,self%intercept_,self%scale_]
    else
      q=p+1;if(self%fit_intercept)q=q+1
      allocate(point(q));point=0;point(q)=1
    end if
    allocate(weight(size(y)));weight=1
    if(present(sample_weight))then
      call require(size(sample_weight)==size(y),'Huber sample_weight length mismatch');weight=sample_weight
    end if
    call huber_solve(y,self%epsilon,self%alpha,self%tol,self%max_iter,point,weight,self%n_iter_, &
      self%loss_,self%converged_,self%optimization_message_,dense,sparse,dense_fortran)
    self%coef_=point(:p);self%scale_=point(size(point));self%intercept_=0
    if(self%fit_intercept)self%intercept_=point(size(point)-1)
    self%n_features_in_=p;self%n_targets_=1;self%matrix_target_=.false.;self%rank_=0
    if(present(dense))then
      self%outliers_=abs(y-matmul(dense,self%coef_)-self%intercept_)>self%epsilon*self%scale_
    else
      pred=sparse%matmul(reshape(self%coef_,[p,1]))
      self%outliers_=abs(y-pred(:,1)-self%intercept_)>self%epsilon*self%scale_
    end if
  end subroutine
  function huber_predict_sparse(self,x)result(y)
    class(huber_regressor),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp)::y(x%n_rows)
    real(dp),allocatable::pred(:,:)
    call require(allocated(self%coef_),'HuberRegressor is not fitted')
    call x%validate();call require(x%n_rows>0.and.x%n_cols==size(self%coef_),'Huber query shape differs')
    call require(all(ieee_is_finite(x%data)),'Huber query data must be finite')
    pred=x%matmul(reshape(self%coef_,[size(self%coef_),1]));y=pred(:,1)+self%intercept_
  end function
  real(dp)function huber_score_dw(self,x,y,sample_weight)result(value)
    class(huber_regressor),intent(in)::self
    real(dp),intent(in)::x(:,:),y(:),sample_weight(:)
    value=r2_score(y,self%predict(x),sample_weight=sample_weight)
  end function
  real(dp)function huber_score_sv(self,x,y,sample_weight)result(value)
    class(huber_regressor),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    value=r2_score(y,self%predict(x),sample_weight=sample_weight)
  end function
  real(dp)function huber_score_dm(self,x,y,sample_weight)result(value)
    class(huber_regressor),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    call require(size(y,2)==1,'Huber score requires one target column')
    value=r2_score(y(:,1),self%predict(x),sample_weight=sample_weight)
  end function
  real(dp)function huber_score_sm(self,x,y,sample_weight)result(value)
    class(huber_regressor),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    call require(size(y,2)==1,'Huber score requires one target column')
    value=r2_score(y(:,1),self%predict(x),sample_weight=sample_weight)
  end function
  real(dp) function sample_median(values) result(median)
    real(dp), intent(in) :: values(:)
    integer :: order(size(values)),n
    n=size(values); order=argsort(values)
    median=0.5_dp*(values(order((n+1)/2))+values(order(n/2+1)))
  end function
  subroutine ransac_set_min_integer(self,value)
    class(ransac_regressor),intent(inout)::self
    integer,intent(in)::value
    self%min_samples=value;self%explicit_min_samples_=.true.;self%fractional_min_samples_=.false.
  end subroutine
  subroutine ransac_set_min_fraction(self,value)
    class(ransac_regressor),intent(inout)::self
    real(dp),intent(in)::value
    self%min_samples=0;self%explicit_min_samples_=.true.;self%fractional_min_samples_=.true.
    self%min_samples_fraction_=value
  end subroutine
  subroutine ransac_clear_min(self)
    class(ransac_regressor),intent(inout)::self
    self%min_samples=0;self%explicit_min_samples_=.false.;self%fractional_min_samples_=.false.
  end subroutine
  subroutine ransac_fit(self,x,y)
    class(ransac_regressor),intent(inout)::self
    real(dp),intent(in)::x(:,:),y(:)
    call ransac_fit_common(self,reshape(y,[size(y),1]),.false.,dense=x)
  end subroutine
  subroutine ransac_fit_dw(self,x,y,sample_weight)
    class(ransac_regressor),intent(inout)::self
    real(dp),intent(in)::x(:,:),y(:),sample_weight(:)
    call ransac_fit_common(self,reshape(y,[size(y),1]),.false.,sample_weight,dense=x)
  end subroutine
  subroutine ransac_fit_dws(self,x,y,sample_weight)
    class(ransac_regressor),intent(inout)::self
    real(dp),intent(in)::x(:,:),y(:),sample_weight
    call ransac_fit_common(self,reshape(y,[size(y),1]),.false.,spread(sample_weight,1,size(y)),dense=x)
  end subroutine
  subroutine ransac_fit_sv(self,x,y,sample_weight)
    class(ransac_regressor),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    call ransac_fit_common(self,reshape(y,[size(y),1]),.false.,sample_weight,sparse=x)
  end subroutine
  subroutine ransac_fit_svs(self,x,y,sample_weight)
    class(ransac_regressor),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:),sample_weight
    call ransac_fit_common(self,reshape(y,[size(y),1]),.false.,spread(sample_weight,1,size(y)),sparse=x)
  end subroutine
  subroutine ransac_fit_dm(self,x,y,sample_weight)
    class(ransac_regressor),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    call ransac_fit_common(self,y,.true.,sample_weight,dense=x)
  end subroutine
  subroutine ransac_fit_dms(self,x,y,sample_weight)
    class(ransac_regressor),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:,:),sample_weight
    call ransac_fit_common(self,y,.true.,spread(sample_weight,1,size(y,1)),dense=x)
  end subroutine
  subroutine ransac_fit_sm(self,x,y,sample_weight)
    class(ransac_regressor),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    call ransac_fit_common(self,y,.true.,sample_weight,sparse=x)
  end subroutine
  subroutine ransac_fit_sms(self,x,y,sample_weight)
    class(ransac_regressor),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:,:),sample_weight
    call ransac_fit_common(self,y,.true.,spread(sample_weight,1,size(y,1)),sparse=x)
  end subroutine
  function ransac_csr_rows(x,rows)result(subset)
    type(csr_matrix),intent(in)::x
    integer,intent(in)::rows(:)
    type(csr_matrix)::subset
    integer::i,k,first,last,total
    subset%n_rows=size(rows);subset%n_cols=x%n_cols
    allocate(subset%indptr(size(rows)+1));subset%indptr(1)=1;total=0
    do i=1,size(rows)
      total=total+x%indptr(rows(i)+1)-x%indptr(rows(i));subset%indptr(i+1)=total+1
    end do
    allocate(subset%indices(total),subset%data(total));k=1
    do i=1,size(rows)
      first=x%indptr(rows(i));last=x%indptr(rows(i)+1)-1
      subset%indices(k:k+last-first)=x%indices(first:last)
      subset%data(k:k+last-first)=x%data(first:last);k=k+last-first+1
    end do
  end function
  subroutine ransac_fit_common(self,y,matrix_target,sample_weight,dense,sparse)
    class(ransac_regressor),intent(inout)::self
    real(dp),intent(in)::y(:,:)
    logical,intent(in)::matrix_target
    real(dp),optional,intent(in)::sample_weight(:),dense(:,:)
    type(csr_matrix),optional,intent(in)::sparse
    type(linear_regression)::candidate
    type(legacy_random_state)::rng
    type(csr_matrix)::subset
    integer::n,p,m,i,best_count,n_inliers,cap
    integer,allocatable::ids(:),inliers(:)
    real(dp),allocatable::callback_x(:,:),prediction(:,:),residual(:),weights(:)
    logical,allocatable::mask(:)
    real(dp)::threshold,score,best_score,nom,den,bound
    logical::found
    call check_positive_regression(self)
    call require(present(dense).neqv.present(sparse),'RANSAC requires one feature representation')
    n=size(y,1);call check_x(y)
    if(present(dense))then
      call check_xy(dense,n);p=size(dense,2)
    else
      call sparse%validate();p=sparse%n_cols
      call require(sparse%n_rows==n.and.n>0.and.p>0,'invalid RANSAC CSR shape')
      call require(all(ieee_is_finite(sparse%data)),'RANSAC CSR data must be finite')
    end if
    if(present(sample_weight))then
      call require(size(sample_weight)==n,'RANSAC sample_weight length mismatch')
      call require(all(ieee_is_finite(sample_weight)),'RANSAC sample_weight must be finite')
    end if
    m=self%min_samples
    if(m==0)then
      if(self%fractional_min_samples_)then
        call require(ieee_is_finite(self%min_samples_fraction_),'RANSAC min_samples must be finite')
        call require(self%min_samples_fraction_>0.and.self%min_samples_fraction_<=1, &
          'RANSAC min_samples fraction must be in (0,1]')
        m=1
        if(self%min_samples_fraction_<1)m=ceiling(self%min_samples_fraction_*n)
      else
        call require(.not.self%explicit_min_samples_,'RANSAC integer min_samples must be positive')
        m=p+1
      end if
    end if
    call require(m>=1.and.m<=n.and.self%max_trials>=0.and.self%max_skips>=0,'invalid RANSAC sample/trial controls')
    call require(ieee_is_finite(self%stop_probability),'RANSAC probability must be finite')
    call require(self%stop_probability>=0.and.self%stop_probability<=1.and.self%stop_n_inliers>=0, &
      'invalid RANSAC stopping controls')
    call require(.not.ieee_is_nan(self%stop_score),'RANSAC stop_score must not be NaN')
    call require(ieee_is_finite(self%residual_threshold),'RANSAC threshold must be finite')
    call require(self%residual_threshold>=0.or.self%residual_threshold==-1,'invalid RANSAC residual threshold')
    call require(associated(self%loss_callback).or.self%loss=='absolute_error'.or.self%loss=='squared_error', &
      'invalid RANSAC loss')
    threshold=self%residual_threshold
    if(threshold==-1)threshold=sample_median(abs(reshape(y,[size(y)])-sample_median(reshape(y,[size(y)]))))
    self%min_samples_=m;self%residual_threshold_=threshold
    self%n_trials_=0;self%n_skips_no_inliers_=0;self%n_skips_invalid_data_=0;self%n_skips_invalid_model_=0
    self%inlier_mask_=spread(.false.,1,n);self%skips_exceeded_=.false.
    allocate(mask(n))
    cap=self%max_trials;best_count=1;best_score=ieee_value(0._dp,ieee_negative_inf);found=.false.
    candidate%fit_intercept=self%fit_intercept;call rng%seed(self%random_state)
    do while(self%n_trials_<cap)
      self%n_trials_=self%n_trials_+1
      if(self%n_skips_no_inliers_+self%n_skips_invalid_data_+self%n_skips_invalid_model_>self%max_skips)exit
      ids=sample_indices(rng,n,m,.false.)
      if(present(sparse))subset=ransac_csr_rows(sparse,ids)
      if(associated(self%is_data_valid).or.associated(self%is_model_valid))then
        if(present(dense))then
          callback_x=dense(ids,:)
        else
          callback_x=subset%to_dense()
        end if
      end if
      if(associated(self%is_data_valid))then
        if(.not.self%is_data_valid(callback_x,y(ids,:)))then
          self%n_skips_invalid_data_=self%n_skips_invalid_data_+1;cycle
        end if
      end if
      call fit_candidate(ids)
      if(associated(self%is_model_valid))then
        if(.not.self%is_model_valid(candidate,callback_x,y(ids,:)))then
          self%n_skips_invalid_model_=self%n_skips_invalid_model_+1;cycle
        end if
      end if
      if(present(dense))then
        prediction=candidate%predict_matrix(dense)
      else
        prediction=candidate%predict_matrix(sparse)
      end if
      if(associated(self%loss_callback))then
        residual=self%loss_callback(y,prediction)
        call require(size(residual)==n,'RANSAC loss callback shape mismatch')
      else
        allocate(residual(n))
        do i=1,n
          if(self%loss=='absolute_error')then
            residual(i)=numpy_sum(abs(y(i,:)-prediction(i,:)))
          else
            residual(i)=numpy_sum((y(i,:)-prediction(i,:))**2)
          end if
        end do
      end if
      mask=.false.
      do i=1,n
        if(.not.ieee_is_nan(residual(i)))mask(i)=residual(i)<=threshold
      end do
      deallocate(residual);n_inliers=count(mask)
      if(n_inliers<best_count)then
        self%n_skips_no_inliers_=self%n_skips_no_inliers_+1;cycle
      end if
      inliers=pack([(i,i=1,n)],mask)
      if(present(dense))then
        score=candidate%score(dense(inliers,:),y(inliers,:))
      else
        subset=ransac_csr_rows(sparse,inliers);score=candidate%score(subset,y(inliers,:))
      end if
      if(n_inliers==best_count)then
        if(.not.ieee_is_nan(score).and..not.ieee_is_nan(best_score))then
          if(score<best_score)cycle
        end if
      end if
      best_count=n_inliers;best_score=score;self%inlier_mask_=mask;found=.true.
      nom=max(epsilon(1._dp),1-self%stop_probability)
      den=max(epsilon(1._dp),1-(real(n_inliers,dp)/n)**m)
      if(nom==1)then
        cap=0
      else if(den/=1)then
        bound=abs(log(nom)/log(den))
        if(bound<cap)cap=ceiling(bound)
      end if
      if(n_inliers>=self%stop_n_inliers)exit
      if(.not.ieee_is_nan(score))then
        if(score>=self%stop_score)exit
      end if
    end do
    self%skips_exceeded_=self%n_skips_no_inliers_+self%n_skips_invalid_data_+self%n_skips_invalid_model_>self%max_skips
    call require(found,'RANSAC could not find a valid consensus set')
    inliers=pack([(i,i=1,n)],self%inlier_mask_);call fit_candidate(inliers)
    self%estimator_=candidate;self%linear_regression=candidate
  contains
    subroutine fit_candidate(rows)
      integer,intent(in)::rows(:)
      if(allocated(weights))deallocate(weights)
      if(present(sample_weight))weights=sample_weight(rows)
      if(present(dense))then
        if(matrix_target)then
          call candidate%fit(dense(rows,:),y(rows,:),weights)
        else
          if(allocated(weights))then
            call candidate%fit(dense(rows,:),y(rows,1),weights)
          else
            call candidate%fit(dense(rows,:),y(rows,1))
          end if
        end if
      else
        subset=ransac_csr_rows(sparse,rows)
        if(matrix_target)then
          call candidate%fit(subset,y(rows,:),weights)
        else
          call candidate%fit(subset,y(rows,1),weights)
        end if
      end if
    end subroutine
  end subroutine
  function ransac_predict_sv(self,x)result(y)
    class(ransac_regressor),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp)::y(x%n_rows)
    y=self%estimator_%predict(x)
  end function
  function ransac_predict_dm(self,x)result(y)
    class(ransac_regressor),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),allocatable::y(:,:)
    y=self%estimator_%predict_matrix(x)
  end function
  function ransac_predict_sm(self,x)result(y)
    class(ransac_regressor),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::y(:,:)
    y=self%estimator_%predict_matrix(x)
  end function
  real(dp)function ransac_score_dw(self,x,y,sample_weight)result(value)
    class(ransac_regressor),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:)
    real(dp),intent(in)::sample_weight(:)
    value=self%estimator_%score(x,y,sample_weight)
  end function
  real(dp)function ransac_score_sv(self,x,y,sample_weight)result(value)
    class(ransac_regressor),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    value=self%estimator_%score(x,y,sample_weight)
  end function
  real(dp)function ransac_score_dm(self,x,y,sample_weight)result(value)
    class(ransac_regressor),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    value=self%estimator_%score(x,y,sample_weight)
  end function
  real(dp)function ransac_score_sm(self,x,y,sample_weight)result(value)
    class(ransac_regressor),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    value=self%estimator_%score(x,y,sample_weight)
  end function
  subroutine poisson_fit(self,x,y)
    class(poisson_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    self%power=1; self%link='log'; call tweedie_fit(self,x,y)
  end subroutine
  subroutine gamma_fit(self,x,y)
    class(gamma_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    self%power=2; self%link='log'; call tweedie_fit(self,x,y)
  end subroutine
  subroutine glm_objective(design,y,w,p,power,link,alpha,loss,g,h,valid,weight)
    real(dp), intent(in) :: design(:,:),y(:),w(:),power,alpha,weight(:)
    character(*), intent(in) :: link
    integer, intent(in) :: p
    real(dp), intent(out) :: loss,g(:),h(:,:)
    logical, intent(out) :: valid
    real(dp) :: eta(size(y)),mu(size(y)),gradient(size(y)),curvature(size(y)),a(size(y))
    integer :: j
    eta=matmul(design,w); valid=.true.; g=0; h=0; loss=huge(1._dp)
    if(link=='log') then
      valid=maxval(abs(eta))<0.25_dp*log(huge(1._dp))
      if(.not.valid) return
      mu=exp(eta)
    else
      mu=eta
    end if
    if(link=='identity'.and.power/=0) then
      valid=all(mu>0)
      if(.not.valid)return
    end if
    if(power/=0)then
      valid=maxval(abs(log(mu)))<0.25_dp*log(huge(1._dp))/max(2._dp,abs(2-power),abs(-power-1))
      if(.not.valid)return
    end if
    if(power==0) then
      a=0.5_dp*(mu-y)**2; gradient=mu-y; curvature=1
      if(link=='log')then
        gradient=gradient*mu;curvature=(2*mu-y)*mu
      end if
    else if(power==1) then
      a=mu-y*log(mu);gradient=(mu-y)/mu;curvature=y/mu**2
      if(link=='log')then
        gradient=mu-y;curvature=mu
      end if
    else if(power==2) then
      a=y/mu+log(mu);gradient=(1-y/mu)/mu;curvature=(2*y/mu-1)/mu**2
      if(link=='log')then
        gradient=1-y/mu;curvature=y/mu
      end if
    else
      ! Eta-independent constants do not change the optimum. expm1 keeps
      ! the objective well conditioned at powers close to one and two.
      a=exponential_minus_one((2-power)*log(mu))/(2-power) &
        -y*exponential_minus_one((1-power)*log(mu))/(1-power)
      gradient=(mu-y)*mu**(-power)
      curvature=mu**(-power)*(1-power*(1-y/mu))
      if(link=='log')then
        gradient=mu**(2-power)-y*mu**(1-power)
        curvature=(2-power)*mu**(2-power)+(power-1)*y*mu**(1-power)
      end if
    end if
    loss=numpy_sum(weight*a)/numpy_sum(weight)
    gradient=gradient*weight/numpy_sum(weight);curvature=curvature*weight/numpy_sum(weight)
    loss=loss+0.5_dp*alpha*sum(w(:p)**2)
    g=matmul(transpose(design),gradient); g(:p)=g(:p)+alpha*w(:p)
    h=matmul(transpose(design),design*spread(curvature,2,size(w)))
    do j=1,p
      h(j,j)=h(j,j)+alpha
    end do
  end subroutine
  elemental real(dp) function exponential_minus_one(z) result(value)
    real(dp), intent(in) :: z
    if(abs(z)<1.e-4_dp) then
      value=z*(1+z*(0.5_dp+z*(1._dp/6+z*(1._dp/24+z*(1._dp/120+z/720)))))
    else
      value=exp(z)-1
    end if
  end function
  subroutine tweedie_fit(self,x,y)
    class(tweedie_regressor),intent(inout)::self
    real(dp),intent(in)::x(:,:),y(:)
    call glm_fit_common(self,y,dense=x)
  end subroutine
  subroutine glm_fit_dw(self,x,y,sample_weight)
    class(tweedie_regressor),intent(inout)::self
    real(dp),intent(in)::x(:,:),y(:),sample_weight(:)
    call glm_fit_common(self,y,sample_weight,dense=x)
  end subroutine
  subroutine glm_fit_dws(self,x,y,sample_weight)
    class(tweedie_regressor),intent(inout)::self
    real(dp),intent(in)::x(:,:),y(:),sample_weight
    call glm_fit_common(self,y,spread(sample_weight,1,size(y)),dense=x)
  end subroutine
  subroutine glm_fit_dm(self,x,y,sample_weight)
    class(tweedie_regressor),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    call require(size(y,2)==1,'GLM requires one target column')
    call glm_fit_common(self,y(:,1),sample_weight,dense=x)
  end subroutine
  subroutine glm_fit_dms(self,x,y,sample_weight)
    class(tweedie_regressor),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:,:),sample_weight
    call require(size(y,2)==1,'GLM requires one target column')
    call glm_fit_common(self,y(:,1),spread(sample_weight,1,size(y,1)),dense=x)
  end subroutine
  subroutine glm_fit_sv(self,x,y,sample_weight)
    class(tweedie_regressor),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    call glm_fit_common(self,y,sample_weight,sparse=x)
  end subroutine
  subroutine glm_fit_svs(self,x,y,sample_weight)
    class(tweedie_regressor),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:),sample_weight
    call glm_fit_common(self,y,spread(sample_weight,1,size(y)),sparse=x)
  end subroutine
  subroutine glm_fit_sm(self,x,y,sample_weight)
    class(tweedie_regressor),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    call require(size(y,2)==1,'GLM requires one target column')
    call glm_fit_common(self,y(:,1),sample_weight,sparse=x)
  end subroutine
  subroutine glm_fit_sms(self,x,y,sample_weight)
    class(tweedie_regressor),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:,:),sample_weight
    call require(size(y,2)==1,'GLM requires one target column')
    call glm_fit_common(self,y(:,1),spread(sample_weight,1,size(y,1)),sparse=x)
  end subroutine
  subroutine glm_fit_common(self,y,sample_weight,dense,sparse)
    class(tweedie_regressor), intent(inout) :: self
    real(dp), intent(in) :: y(:)
    real(dp),optional,intent(in)::dense(:,:)
    type(csr_matrix),optional,intent(in)::sparse
    real(dp),optional,intent(in)::sample_weight(:)
    real(dp)::weight(size(y)),mean_y
    real(dp), allocatable :: design(:,:),w(:),trial(:),g(:),h(:,:),gt(:),ht(:,:),step(:),s(:),v(:,:)
    real(dp) :: loss,next,rate,cutoff
    integer :: n,p,q,it,ls
    logical :: valid
    call check_positive_regression(self)
    call require(present(dense).neqv.present(sparse),'GLM requires one feature representation')
    if(present(dense))then
      call check_xy(dense,size(y));p=size(dense,2)
    else
      call sparse%validate()
      call require(sparse%n_rows==size(y).and.sparse%n_rows>0.and.sparse%n_cols>0,'invalid GLM CSR shape')
      call require(all(ieee_is_finite(sparse%data)),'GLM CSR data must be finite');p=sparse%n_cols
    end if
    call require(self%solver=='lbfgs'.or.self%solver=='native-newton','unsupported GLM solver')
    if(self%solver=='native-newton')call require(present(dense),'native-newton currently requires dense GLM input')
    select type(self)
    class is(poisson_regressor)
      self%power=1;self%link='log'
    class is(gamma_regressor)
      self%power=2;self%link='log'
    end select
    call require(all(ieee_is_finite([self%power,self%alpha,self%tol])),'GLM controls must be finite')
    call require(self%alpha>=0.and.self%tol>0.and.self%max_iter>0,'invalid GLM controls')
    self%link_=self%link
    if(self%link=='auto') self%link_=merge('identity','log     ',self%power<=0)
    call require(self%link_=='identity'.or.self%link_=='log','GLM link must be auto, identity or log')
    call glm_target_check(y,self%power)
    weight=1
    if(present(sample_weight))then
      call require(size(sample_weight)==size(y),'GLM sample_weight length mismatch');weight=sample_weight
    end if
    call require(all(ieee_is_finite(weight)),'GLM sample weights must be finite')
    call require(numpy_sum(weight)/=0,'GLM sample weights must have nonzero sum')
    mean_y=numpy_sum(weight*y)/numpy_sum(weight)
    if(self%link_=='log') call require(mean_y>0,'log-link GLM requires a positive weighted target mean')
    n=size(y); q=p
    if(self%fit_intercept) q=q+1
    allocate(w(q))
    w=0
    if(self%warm_start.and.allocated(self%coef_))then
      call require(size(self%coef_)==p,'GLM warm-start feature count mismatch')
      w(:p)=self%coef_
      if(self%fit_intercept)w(q)=self%intercept_
    else if(self%fit_intercept)then
      w(q)=mean_y
      if(self%link_=='log')w(q)=log(mean_y)
    end if
    if(self%solver=='lbfgs')then
      call glm_lbfgs_solve(w,y,weight,self%power,self%link_,self%alpha,self%tol,self%max_iter, &
        self%n_iter_,self%converged_,self%optimization_message_,dense,sparse)
    else
      allocate(design(n,q),trial(q),g(q),h(q,q),gt(q),ht(q,q),step(q),s(q),v(q,q))
      design(:,:p)=dense;if(self%fit_intercept)design(:,q)=1
      call glm_objective(design,y,w,p,self%power,self%link_,self%alpha,loss,g,h,valid,weight)
      call require(valid,'GLM initial prediction exceeds numerical range')
      self%converged_=.false.; self%n_iter_=0
      do it=1,self%max_iter
        if(maxval(abs(g))<=self%tol) then
          self%converged_=.true.; exit
        end if
        call symmetric_eigen(h,s,v); cutoff=epsilon(1._dp)*q*maxval(s)
        step=0
        where(s>cutoff) step=matmul(transpose(v),g)/max(s,tiny(1._dp))
        step=matmul(v,step)
        if(dot_product(g,step)<=0)step=g
        rate=1
        do ls=1,80
          trial=w-rate*step
          call glm_objective(design,y,trial,p,self%power,self%link_,self%alpha,next,gt,ht,valid,weight)
          if(valid.and.next<=loss-1.e-4_dp*rate*dot_product(g,step)) exit
          if(valid.and.abs(next-loss)<=8*epsilon(1._dp)*max(1._dp,abs(loss)) &
            .and.maxval(abs(gt))<maxval(abs(g))) exit
          rate=rate*0.5_dp
        end do
        if(ls>80) exit
        w=trial; g=gt; h=ht; loss=next; self%n_iter_=it
      end do
      if(maxval(abs(g))<=self%tol) self%converged_=.true.
      self%optimization_message_='native damped eigen/Newton'
    end if
    self%power_=self%power;self%n_features_in_=p;self%n_targets_=1;self%matrix_target_=.false.
    self%coef_=w(:p); self%intercept_=0
    if(self%fit_intercept) self%intercept_=w(q)
  end subroutine
  function glm_predict(self,x) result(y)
    class(tweedie_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: y(size(x,1))
    call require(allocated(self%coef_),'GLM is not fitted')
    call check_xy(transpose(x),size(self%coef_)); y=matmul(x,self%coef_)+self%intercept_
    if(self%link_=='log') then
      call require(maxval(abs(y))<log(huge(1._dp)),'GLM prediction exceeds numerical range')
      y=exp(y)
    end if
  end function
  function glm_predict_sparse(self,x)result(y)
    class(tweedie_regressor),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp)::y(x%n_rows)
    real(dp),allocatable::pred(:,:)
    call require(allocated(self%coef_),'GLM is not fitted')
    call x%validate();call require(x%n_rows>0.and.x%n_cols==size(self%coef_),'GLM query shape differs')
    call require(all(ieee_is_finite(x%data)),'GLM query data must be finite')
    pred=x%matmul(reshape(self%coef_,[size(self%coef_),1]));y=pred(:,1)+self%intercept_
    if(self%link_=='log')then
      call require(maxval(abs(y))<log(huge(1._dp)),'GLM prediction exceeds numerical range');y=exp(y)
    end if
  end function
  real(dp) function tweedie_deviance(y,mu,power) result(d)
    real(dp), intent(in) :: y(:),mu(:),power
    real(dp) :: term(size(y)),a,b,ratio,log_ratio
    integer :: i
    if(power==0) then
      d=sum((y-mu)**2)/size(y)
    else if(power==1) then
      term=mu-y
      do i=1,size(y)
        if(y(i)>0) term(i)=term(i)+y(i)*log(y(i)/mu(i))
      end do
      d=2*sum(term)/size(y)
    else if(power==2) then
      d=2*sum(log(mu/y)+y/mu-1)/size(y)
    else if(power<0.or.power>2.or.(power>0.and.power<1))then
      a=2-power;b=1-power
      term=max(y,0._dp)**a/(a*b)-y*mu**b/b+mu**a/a
      d=2*sum(term)/size(y)
    else
      a=2-power; b=1-power
      do i=1,size(y)
        if(y(i)==0) then
          term(i)=mu(i)**a/a
        else
          ratio=y(i)/mu(i); log_ratio=log(ratio)
          if(power<=1.5_dp) then
            term(i)=mu(i)**a*(ratio*exponential_minus_one(b*log_ratio)/b-(ratio-1))/a
          else
            term(i)=mu(i)**a*(exponential_minus_one(a*log_ratio)/a-(ratio-1))/b
          end if
        end if
      end do
      d=2*sum(term)/size(y)
    end if
  end function
  subroutine glm_target_check(y,power)
    real(dp),intent(in)::y(:),power
    call require(size(y)>0,'GLM requires at least one target')
    call require(all(ieee_is_finite(y)),'GLM targets must be finite')
    if(power>0.and.power<2)call require(all(y>=0),'GLM targets must be nonnegative')
    if(power>=2)call require(all(y>0),'GLM targets must be positive')
  end subroutine
  real(dp) function glm_score(self,x,y) result(s)
    class(tweedie_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    s=glm_score_common(self,x,y)
  end function
  real(dp) function glm_score_dw(self,x,y,sample_weight) result(s)
    class(tweedie_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:),sample_weight(:)
    s=glm_score_common(self,x,y,sample_weight)
  end function
  real(dp) function glm_score_dm(self,x,y,sample_weight) result(s)
    class(tweedie_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    call require(size(y,2)==1,'GLM score requires one target column')
    s=glm_score_common(self,x,y(:,1),sample_weight)
  end function
  real(dp) function glm_score_common(self,x,y,sample_weight)result(s)
    class(tweedie_regressor),intent(in)::self
    real(dp),intent(in)::x(:,:),y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    call check_xy(x,size(y));s=glm_score_values(self,self%predict(x),y,sample_weight)
  end function
  real(dp) function glm_score_sv(self,x,y,sample_weight)result(s)
    class(tweedie_regressor),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    s=glm_score_values(self,self%predict(x),y,sample_weight)
  end function
  real(dp) function glm_score_sm(self,x,y,sample_weight)result(s)
    class(tweedie_regressor),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    call require(size(y,2)==1,'GLM score requires one target column')
    s=glm_score_values(self,self%predict(x),y(:,1),sample_weight)
  end function
  real(dp) function glm_score_values(self,prediction,y,sample_weight) result(s)
    class(tweedie_regressor), intent(in) :: self
    real(dp), intent(in) :: prediction(:),y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    real(dp)::weight(size(y)),mean_y,numerator,denominator,one(1),mu(1)
    integer::i
    call require(size(prediction)==size(y),'GLM score target length differs');call glm_target_check(y,self%power_)
    weight=1
    if(present(sample_weight))then
      call require(size(sample_weight)==size(y),'GLM score sample_weight length mismatch');weight=sample_weight
    end if
    call require(all(ieee_is_finite(weight)),'GLM score weights must be finite')
    call require(numpy_sum(weight)/=0,'GLM score weights must have nonzero sum')
    mean_y=numpy_sum(weight*y)/numpy_sum(weight)
    if(self%power_/=0)then
      call require(all(prediction>0).and.mean_y>0,'GLM deviance requires positive predictions and target mean')
    end if
    numerator=0;denominator=0
    do i=1,size(y)
      one=y(i);mu=prediction(i)
      numerator=numerator+weight(i)*tweedie_deviance(one,mu,self%power_)
      mu=mean_y;denominator=denominator+weight(i)*tweedie_deviance(one,mu,self%power_)
    end do
    s=1-ieee_quotient(numerator,denominator)
  end function
  ! Bayesian precision updates and OMP selection follow sklearn 1.6.1
  ! linear_model/_bayes.py and _omp.py. See THIRD_PARTY_NOTICES.md.
  subroutine bayesian_fit(self,x,y)
    class(bayesian_ridge), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp) :: xc(size(x,1),size(x,2)),yc(size(y)),xm(size(x,2)),ym
    real(dp) :: u(size(x,1),size(x,2)),s(size(x,2)),v(size(x,2),size(x,2))
    real(dp) :: rhs(size(x,2)),old(size(x,2)),weights(size(x,2)),gamma,rss
    integer :: it,j,p,k
    call check_positive_regression(self)
    call require(self%max_iter>0.and.self%tol>=0,'invalid BayesianRidge iteration parameters')
    call require(min(self%alpha_1,self%alpha_2,self%lambda_1,self%lambda_2)>0, &
      'BayesianRidge hyperpriors must be positive')
    call require((self%alpha_init==-1.or.self%alpha_init>0).and.self%lambda_init>0, &
      'invalid BayesianRidge initial precision')
    call centered(x,y,self%fit_intercept,xc,yc,xm,ym)
    p=size(x,2); k=min(size(x,1),p)
    call dense_svd(xc,s,u,v); rhs=s*matmul(transpose(u),yc)
    self%rank_=count(s>epsilon(1._dp)*maxval(s)*max(size(x,1),p))
    self%alpha_=self%alpha_init
    if(self%alpha_init==-1) self%alpha_=1/(sum((yc-sum(yc)/size(yc))**2)/size(yc)+epsilon(1._dp))
    self%lambda_=self%lambda_init; old=0; self%converged_=.false.
    do it=1,self%max_iter
      self%coef_=matmul(v,rhs/(s*s+self%lambda_/self%alpha_))
      rss=sum((yc-matmul(xc,self%coef_))**2)
      gamma=sum(self%alpha_*s*s/(self%lambda_+self%alpha_*s*s))
      self%lambda_=(gamma+2*self%lambda_1)/(sum(self%coef_**2)+2*self%lambda_2)
      self%alpha_=(size(y)-gamma+2*self%alpha_1)/(rss+2*self%alpha_2)
      self%n_iter_=it
      if(it>1.and.sum(abs(old-self%coef_))<self%tol) then
        self%converged_=.true.; exit
      end if
      old=self%coef_
    end do
    self%coef_=matmul(v,rhs/(s*s+self%lambda_/self%alpha_))
    self%intercept_=ym-dot_product(xm,self%coef_)
    weights=0; weights(:k)=1/(self%alpha_*s(:k)**2+self%lambda_)
    self%sigma_=matmul(v*spread(weights,1,p),transpose(v))
  end subroutine
  function bayesian_std(self,x) result(std)
    class(bayesian_ridge), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: std(size(x,1))
    call require(allocated(self%sigma_),'BayesianRidge is not fitted')
    call check_xy(transpose(x),size(self%coef_))
    std=sqrt(max(0._dp,sum(matmul(x,self%sigma_)*x,dim=2))+1/self%alpha_)
  end function
  function positive_inverse(a) result(inv)
    real(dp), intent(in) :: a(:,:)
    real(dp) :: inv(size(a,1),size(a,2)),l(size(a,1),size(a,2)),e(size(a,1))
    integer :: j
    l=cholesky_factor(a)
    do j=1,size(a,1)
      e=0; e(j)=1; inv(:,j)=solve_upper(transpose(l),solve_lower(l,e))
    end do
  end function
  subroutine ard_posterior(x,y,alpha,lambda,keep,sigma,coef)
    real(dp), intent(in) :: x(:,:),y(:),alpha,lambda(:)
    integer, intent(in) :: keep(:)
    real(dp), allocatable, intent(out) :: sigma(:,:)
    real(dp), intent(out) :: coef(:)
    real(dp) :: a(size(keep),size(keep))
    integer :: j
    coef=0
    a=alpha*matmul(transpose(x(:,keep)),x(:,keep))
    do j=1,size(keep)
      a(j,j)=a(j,j)+lambda(keep(j))
    end do
    sigma=positive_inverse(a)
    coef(keep)=alpha*matmul(sigma,matmul(transpose(x(:,keep)),y))
  end subroutine
  subroutine ard_fit(self,x,y)
    class(ard_regression), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp) :: xc(size(x,1),size(x,2)),yc(size(y)),xm(size(x,2)),ym
    real(dp) :: old(size(x,2)),gamma(size(x,2)),rss
    integer, allocatable :: keep(:)
    integer :: it,j,p
    call check_positive_regression(self)
    call require(self%max_iter>0.and.self%tol>=0.and.self%threshold_lambda>0,'invalid ARD parameters')
    call require(min(self%alpha_1,self%alpha_2,self%lambda_1,self%lambda_2)>0,'ARD hyperpriors must be positive')
    call centered(x,y,self%fit_intercept,xc,yc,xm,ym); p=size(x,2)
    self%lambda_=spread(1._dp,1,p); self%coef_=spread(0._dp,1,p)
    self%alpha_=1/(sum((yc-sum(yc)/size(yc))**2)/size(yc)+epsilon(1._dp))
    keep=[(j,j=1,p)]; old=0; self%converged_=.false.
    do it=1,self%max_iter
      call ard_posterior(xc,yc,self%alpha_,self%lambda_,keep,self%sigma_,self%coef_)
      rss=sum((yc-matmul(xc,self%coef_))**2)
      do j=1,size(keep)
        gamma(j)=1-self%lambda_(keep(j))*self%sigma_(j,j)
      end do
      self%lambda_(keep)=(gamma(:size(keep))+2*self%lambda_1)/(self%coef_(keep)**2+2*self%lambda_2)
      self%alpha_=(size(y)-sum(gamma(:size(keep)))+2*self%alpha_1)/(rss+2*self%alpha_2)
      keep=pack([(j,j=1,p)],self%lambda_<self%threshold_lambda)
      where(self%lambda_>=self%threshold_lambda) self%coef_=0
      self%n_iter_=it
      if((it>1.and.sum(abs(old-self%coef_))<self%tol).or.size(keep)==0) then
        self%converged_=.true.; exit
      end if
      old=self%coef_
    end do
    call ard_posterior(xc,yc,self%alpha_,self%lambda_,keep,self%sigma_,self%coef_)
    self%intercept_=ym-dot_product(xm,self%coef_); self%rank_=size(keep)
  end subroutine
  function ard_std(self,x) result(std)
    class(ard_regression), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: std(size(x,1))
    integer, allocatable :: keep(:)
    integer :: j
    call require(allocated(self%sigma_),'ARDRegression is not fitted')
    call check_xy(transpose(x),size(self%coef_))
    keep=pack([(j,j=1,size(self%coef_))],self%lambda_<self%threshold_lambda)
    std=sqrt(max(0._dp,sum(matmul(x(:,keep),self%sigma_)*x(:,keep),dim=2))+1/self%alpha_)
  end function
  subroutine omp_fit(self,x,y)
    class(orthogonal_matching_pursuit), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp) :: xc(size(x,1),size(x,2)),yc(size(y)),xm(size(x,2)),ym
    real(dp) :: residual(size(y)),corr(size(x,2))
    real(dp), allocatable :: s(:),u(:,:),v(:,:),c(:)
    integer :: p,k,j,limit,idx(size(x,2))
    call check_positive_regression(self)
    call centered(x,y,self%fit_intercept,xc,yc,xm,ym); p=size(x,2)
    call require(self%n_nonzero_coefs>=0.and.self%n_nonzero_coefs<=p,'invalid OMP sparsity')
    call require(self%tol>=0.or.self%tol==-1,'OMP tol must be nonnegative or -1')
    limit=self%n_nonzero_coefs
    if(limit==0) limit=max(1,p/10)
    if(self%tol>=0) limit=p
    residual=yc; self%coef_=spread(0._dp,1,p); self%n_iter_=0
    do k=1,limit
      corr=matmul(transpose(xc),residual); j=maxloc(abs(corr),dim=1)
      if(any(idx(:k-1)==j).or.corr(j)**2<epsilon(1._dp)) exit
      idx(k)=j
      if(allocated(s)) deallocate(s,u,v,c)
      allocate(s(k),u(size(y),k),v(k,k),c(k))
      call dense_svd(xc(:,idx(:k)),s,u,v)
      if(minval(s)<=sqrt(epsilon(1._dp))*maxval(s)) exit
      c=matmul(v,matmul(transpose(u),yc)/s)
      self%coef_=0; self%coef_(idx(:k))=c
      residual=yc-matmul(xc,self%coef_); self%n_iter_=k
      if(self%tol>=0.and.sum(residual**2)<=self%tol) exit
    end do
    self%rank_=self%n_iter_; self%intercept_=ym-dot_product(xm,self%coef_)
  end subroutine
  subroutine centered(x,y,fit_intercept,xc,yc,xmean,ymean)
    real(dp), intent(in) :: x(:,:),y(:)
    logical, intent(in) :: fit_intercept
    real(dp), intent(out) :: xc(:,:),yc(:),xmean(:),ymean
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    xmean=0; ymean=0
    if(fit_intercept) then
      xmean=sum(x,dim=1)/size(x,1); ymean=sum(y)/size(y)
    end if
    xc=x-spread(xmean,1,size(x,1)); yc=y-ymean
  end subroutine
  subroutine spectral_fit(self,x,y,alpha)
    class(linear_regression), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:),alpha
    call check_positive_regression(self)
    if(self%positive) then
      call positive_fit_core(self,x,y,alpha)
    else
      call spectral_unconstrained(self,x,y,alpha)
    end if
  end subroutine
  subroutine positive_fit_core(self,x,y,alpha)
    class(linear_regression), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:),alpha
    real(dp) :: xc(size(x,1),size(x,2)),yc(size(y)),xm(size(x,2)),ym
    call require(ieee_is_finite(alpha),'positive linear alpha must be finite')
    call require(alpha>=0,'alpha must be nonnegative')
    call positive_data_check(x,y)
    call centered(x,y,self%fit_intercept,xc,yc,xm,ym)
    call positive_least_squares(xc,yc,alpha,self%coef_)
    self%rank_=0;self%intercept_=ieee_difference(ym,positive_dot(xm,self%coef_))
    call require(ieee_is_finite(self%intercept_),'positive linear intercept exceeds working range')
  end subroutine
  subroutine spectral_unconstrained(self,x,y,alpha)
    class(linear_regression), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:),alpha
    real(dp) :: xc(size(x,1),size(x,2)),yc(size(y)),xm(size(x,2)),ym
    real(dp) :: u(size(x,1),size(x,2)),vals(size(x,2)),vecs(size(x,2),size(x,2))
    real(dp) :: rhs(size(x,2)),projection(size(x,2)),cutoff
    integer :: j,p
    call require(alpha>=0,'alpha must be nonnegative'); p=size(x,2)
    call centered(x,y,self%fit_intercept,xc,yc,xm,ym)
    call dense_svd(xc,vals,u,vecs)
    cutoff=epsilon(1._dp)*max(size(x,1),p)*maxval(vals)
    self%rank_=count(vals>cutoff); projection=matmul(transpose(u),yc)
    do j=1,p
      if(vals(j)>cutoff.or.alpha>0) then
        projection(j)=projection(j)*vals(j)/(vals(j)**2+alpha)
      else
        projection(j)=0
      end if
    end do
    self%coef_=matmul(vecs,projection); self%intercept_=ym-dot_product(xm,self%coef_)
  end subroutine
  subroutine ols_fit(self,x,y)
    class(linear_regression), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    call ols_fit_common(self,reshape(y,[size(y),1]),.false.,dense=x)
  end subroutine
  subroutine ols_fit_dw(self,x,y,sample_weight)
    class(linear_regression),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:)
    real(dp),intent(in)::sample_weight(:)
    call ols_direct_only(self)
    call ols_fit_common(self,reshape(y,[size(y),1]),.false.,sample_weight,dense=x)
  end subroutine
  subroutine ols_fit_dws(self,x,y,sample_weight)
    class(linear_regression),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:),sample_weight
    real(dp)::weights(size(x,1))
    weights=sample_weight
    call ols_fit_dw(self,x,y,weights)
  end subroutine
  subroutine ols_fit_sv(self,x,y,sample_weight)
    class(linear_regression),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    call ols_direct_only(self)
    call ols_fit_common(self,reshape(y,[size(y),1]),.false.,sample_weight,sparse=x)
  end subroutine
  subroutine ols_fit_svs(self,x,y,sample_weight)
    class(linear_regression),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:),sample_weight
    real(dp)::weights(x%n_rows)
    weights=sample_weight
    call ols_fit_sv(self,x,y,weights)
  end subroutine
  subroutine ols_fit_dm(self,x,y,sample_weight)
    class(linear_regression),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    call ols_direct_only(self)
    call ols_fit_common(self,y,.true.,sample_weight,dense=x)
  end subroutine
  subroutine ols_fit_dms(self,x,y,sample_weight)
    class(linear_regression),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:,:),sample_weight
    real(dp)::weights(size(x,1))
    weights=sample_weight
    call ols_fit_dm(self,x,y,weights)
  end subroutine
  subroutine ols_fit_sm(self,x,y,sample_weight)
    class(linear_regression),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    call ols_direct_only(self)
    call ols_fit_common(self,y,.true.,sample_weight,sparse=x)
  end subroutine
  subroutine ols_fit_sms(self,x,y,sample_weight)
    class(linear_regression),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:,:),sample_weight
    real(dp)::weights(x%n_rows)
    weights=sample_weight
    call ols_fit_sm(self,x,y,weights)
  end subroutine
  subroutine ols_direct_only(self)
    class(linear_regression),intent(in)::self
    select type(self)
    type is(linear_regression)
    class default
      call require(.false.,'these LinearRegression overloads require a direct LinearRegression estimator')
    end select
  end subroutine
  subroutine ols_fit_common(self,y,matrix_target,sample_weight,dense,sparse)
    class(linear_regression),intent(inout)::self
    real(dp),intent(in)::y(:,:)
    logical,intent(in)::matrix_target
    real(dp),optional,intent(in)::sample_weight(:),dense(:,:)
    type(csr_matrix),optional,intent(in)::sparse
    type(ols_solution)::fit
    if(present(dense))then
      fit=fit_ols_estimator(dense,y,self%fit_intercept,self%positive,sample_weight)
    else
      fit=fit_ols_estimator(sparse,y,self%fit_intercept,self%positive,sample_weight)
    end if
    self%coef_matrix_=fit%coefficients;self%intercepts_=fit%intercept
    self%n_targets_=size(y,2);self%n_features_in_=size(fit%coefficients,2)
    self%matrix_target_=matrix_target;self%rank_=fit%rank;self%least_squares_solver_=fit%solver
    if(allocated(self%singular_))deallocate(self%singular_)
    if(allocated(fit%singular))self%singular_=fit%singular
    if(allocated(self%lsqr_iterations_))deallocate(self%lsqr_iterations_,self%lsqr_stop_code_)
    if(allocated(fit%iterations))then
      self%lsqr_iterations_=fit%iterations;self%lsqr_stop_code_=fit%stop_code
    end if
    self%intercept_=0
    if(allocated(self%coef_))deallocate(self%coef_)
    if(.not.matrix_target)then
      self%coef_=fit%coefficients(1,:);self%intercept_=fit%intercept(1)
    end if
  end subroutine
  function ols_predict_sparse(self,x)result(y)
    class(linear_regression),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp)::y(x%n_rows)
    real(dp),allocatable::matrix(:,:)
    call ols_direct_only(self)
    call require(.not.self%matrix_target_,'matrix-target LinearRegression requires predict_matrix')
    call require(allocated(self%coef_),'LinearRegression is not fitted')
    call x%validate();call require(x%n_rows>0,'LinearRegression query must be nonempty')
    call require(x%n_cols==size(self%coef_),'LinearRegression query feature count differs')
    call require(all(ieee_is_finite(x%data)),'LinearRegression query data must be finite')
    matrix=x%matmul(reshape(self%coef_,[size(self%coef_),1]));y=matrix(:,1)+self%intercept_
  end function
  function ols_predict_dm(self,x)result(y)
    class(linear_regression),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),allocatable::y(:,:)
    call ols_direct_only(self)
    call require(allocated(self%coef_matrix_),'matrix LinearRegression is not fitted')
    call check_x(x);call require(size(x,2)==size(self%coef_matrix_,2),'LinearRegression query feature count differs')
    y=matmul(x,transpose(self%coef_matrix_))+spread(self%intercepts_,1,size(x,1))
  end function
  function ols_predict_sm(self,x)result(y)
    class(linear_regression),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::y(:,:)
    call ols_direct_only(self)
    call require(allocated(self%coef_matrix_),'matrix LinearRegression is not fitted')
    call x%validate();call require(x%n_rows>0,'LinearRegression query must be nonempty')
    call require(x%n_cols==size(self%coef_matrix_,2),'LinearRegression query feature count differs')
    call require(all(ieee_is_finite(x%data)),'LinearRegression query data must be finite')
    y=x%matmul(transpose(self%coef_matrix_))+spread(self%intercepts_,1,x%n_rows)
  end function
  real(dp)function ols_score_dense(self,x,y)result(value)
    class(linear_regression),intent(in)::self
    real(dp),intent(in)::x(:,:),y(:)
    value=r2_score(y,self%predict(x))
  end function
  real(dp)function ols_score_dw(self,x,y,sample_weight)result(value)
    class(linear_regression),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:)
    real(dp),intent(in)::sample_weight(:)
    call ols_direct_only(self)
    value=r2_score(y,self%predict(x),sample_weight=sample_weight)
  end function
  real(dp)function ols_score_sv(self,x,y,sample_weight)result(value)
    class(linear_regression),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    call ols_direct_only(self)
    value=r2_score(y,self%predict(x),sample_weight=sample_weight)
  end function
  real(dp)function ols_score_dm(self,x,y,sample_weight)result(value)
    class(linear_regression),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    call ols_direct_only(self)
    value=r2_score(y,self%predict_matrix(x),sample_weight=sample_weight)
  end function
  real(dp)function ols_score_sm(self,x,y,sample_weight)result(value)
    class(linear_regression),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    call ols_direct_only(self)
    value=r2_score(y,self%predict_matrix(x),sample_weight=sample_weight)
  end function
  subroutine ridge_fit(self,x,y)
    class(ridge),intent(inout)::self
    real(dp),intent(in)::x(:,:),y(:)
    call ridge_fit_common(self,reshape(y,[size(y),1]),.false.,dense=x)
  end subroutine

  subroutine ridge_fit_dw(self,x,y,sample_weight)
    class(ridge),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:)
    real(dp),intent(in)::sample_weight(:)
    call ridge_direct_only(self)
    call ridge_fit_common(self,reshape(y,[size(y),1]),.false.,sample_weight,dense=x)
  end subroutine
  subroutine ridge_fit_dws(self,x,y,sample_weight)
    class(ridge),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:),sample_weight
    real(dp)::weights(size(x,1))
    weights=sample_weight
    call ridge_fit_dw(self,x,y,weights)
  end subroutine
  subroutine ridge_fit_sv(self,x,y,sample_weight)
    class(ridge),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    call ridge_direct_only(self)
    call ridge_fit_common(self,reshape(y,[size(y),1]),.false.,sample_weight,sparse=x)
  end subroutine
  subroutine ridge_fit_svs(self,x,y,sample_weight)
    class(ridge),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:),sample_weight
    real(dp)::weights(x%n_rows)
    weights=sample_weight
    call ridge_fit_sv(self,x,y,weights)
  end subroutine
  subroutine ridge_fit_dm(self,x,y,sample_weight)
    class(ridge),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    call ridge_direct_only(self)
    call ridge_fit_common(self,y,.true.,sample_weight,dense=x)
  end subroutine
  subroutine ridge_fit_dms(self,x,y,sample_weight)
    class(ridge),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:,:),sample_weight
    real(dp)::weights(size(x,1))
    weights=sample_weight
    call ridge_fit_dm(self,x,y,weights)
  end subroutine
  subroutine ridge_fit_sm(self,x,y,sample_weight)
    class(ridge),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    call ridge_direct_only(self)
    call ridge_fit_common(self,y,.true.,sample_weight,sparse=x)
  end subroutine
  subroutine ridge_fit_sms(self,x,y,sample_weight)
    class(ridge),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:,:),sample_weight
    real(dp)::weights(x%n_rows)
    weights=sample_weight
    call ridge_fit_sm(self,x,y,weights)
  end subroutine
  subroutine ridge_direct_only(self)
    class(ridge),intent(in)::self
    select type(self)
    type is(ridge)
    class default
      call require(.false.,'these Ridge overloads require a direct Ridge estimator, not RidgeCV')
    end select
  end subroutine
  subroutine ridge_fit_common(self,y,matrix_target,sample_weight,dense,sparse)
    class(ridge),intent(inout)::self
    real(dp),intent(in)::y(:,:)
    logical,intent(in)::matrix_target
    real(dp),optional,intent(in)::sample_weight(:),dense(:,:)
    type(csr_matrix),optional,intent(in)::sparse
    type(ridge_solver_options)::o
    type(ridge_solution)::fit
    real(dp),allocatable::penalty(:)
    o%solver=self%solver;o%tol=self%tol;o%max_iter=self%max_iter
    o%random_state=self%random_state;o%positive=self%positive
    penalty=[self%alpha];if(allocated(self%alpha_per_target))penalty=self%alpha_per_target
    if(present(dense))then
      fit=fit_ridge_estimator(dense,y,penalty,o,self%fit_intercept,sample_weight)
    else
      fit=fit_ridge_estimator(sparse,y,penalty,o,self%fit_intercept,sample_weight)
    end if
    self%coef_matrix_=fit%coefficients;self%intercepts_=fit%intercept
    self%n_iter_=fit%n_iter;self%native_iterations_=fit%native_iterations;self%stop_code_=fit%stop_code
    if(allocated(self%singular_))deallocate(self%singular_)
    if(allocated(self%lsqr_iterations_))deallocate(self%lsqr_iterations_,self%lsqr_stop_code_)
    self%least_squares_solver_=''
    self%converged_per_target_=fit%converged;self%converged_=all(fit%converged);self%solver_=fit%solver
    self%n_targets_=size(y,2);self%n_features_in_=size(fit%coefficients,2);self%matrix_target_=matrix_target
    self%rank_=0;self%intercept_=0
    if(allocated(self%coef_))deallocate(self%coef_)
    if(.not.matrix_target)then
      self%coef_=fit%coefficients(1,:);self%intercept_=fit%intercept(1)
    end if
  end subroutine
  function ridge_predict_dense(self,x)result(y)
    class(ridge),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp)::y(size(x,1))
    call require(.not.self%matrix_target_,'matrix-target Ridge requires predict_matrix')
    y=linear_predict(self,x)
  end function
  function ridge_predict_sparse(self,x)result(y)
    class(ridge),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp)::y(x%n_rows)
    real(dp),allocatable::matrix(:,:)
    call require(.not.self%matrix_target_,'matrix-target Ridge requires predict_matrix')
    call require(allocated(self%coef_),'Ridge is not fitted')
    call x%validate();call require(x%n_rows>0,'Ridge query must be nonempty')
    call require(x%n_cols==size(self%coef_),'Ridge query feature count differs')
    call require(all(ieee_is_finite(x%data)),'Ridge query data must be finite')
    matrix=x%matmul(reshape(self%coef_,[size(self%coef_),1]));y=matrix(:,1)+self%intercept_
  end function
  function ridge_predict_dm(self,x)result(y)
    class(ridge),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),allocatable::y(:,:)
    call require(allocated(self%coef_matrix_),'matrix Ridge is not fitted')
    call check_x(x);call require(size(x,2)==size(self%coef_matrix_,2),'Ridge query feature count differs')
    y=matmul(x,transpose(self%coef_matrix_))+spread(self%intercepts_,1,size(x,1))
  end function
  function ridge_predict_sm(self,x)result(y)
    class(ridge),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::y(:,:)
    call require(allocated(self%coef_matrix_),'matrix Ridge is not fitted')
    call x%validate();call require(x%n_rows>0,'Ridge query must be nonempty')
    call require(x%n_cols==size(self%coef_matrix_,2),'Ridge query feature count differs')
    call require(all(ieee_is_finite(x%data)),'Ridge query data must be finite')
    y=x%matmul(transpose(self%coef_matrix_))+spread(self%intercepts_,1,x%n_rows)
  end function
  real(dp)function ridge_score_dense(self,x,y)result(value)
    class(ridge),intent(in)::self
    real(dp),intent(in)::x(:,:),y(:)
    value=r2_score(y,self%predict(x))
  end function
  real(dp)function ridge_score_dw(self,x,y,sample_weight)result(value)
    class(ridge),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:)
    real(dp),intent(in)::sample_weight(:)
    value=r2_score(y,self%predict(x),sample_weight=sample_weight)
  end function
  real(dp)function ridge_score_sv(self,x,y,sample_weight)result(value)
    class(ridge),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:)
    real(dp),optional,intent(in)::sample_weight(:)
    value=r2_score(y,self%predict(x),sample_weight=sample_weight)
  end function
  real(dp)function ridge_score_dm(self,x,y,sample_weight)result(value)
    class(ridge),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    value=r2_score(y,self%predict_matrix(x),sample_weight=sample_weight)
  end function
  real(dp)function ridge_score_sm(self,x,y,sample_weight)result(value)
    class(ridge),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    value=r2_score(y,self%predict_matrix(x),sample_weight=sample_weight)
  end function
  function linear_predict(self,x) result(y)
    class(linear_regression), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: y(size(x,1))
    integer :: i
    call require(.not.self%matrix_target_,'matrix-target linear model requires predict_matrix')
    call require(allocated(self%coef_),'linear model is not fitted')
    call check_xy(transpose(x),size(self%coef_))
    if(self%positive) then
      do i=1,size(x,1)
        y(i)=ieee_difference(positive_dot(x(i,:),self%coef_),-self%intercept_)
      end do
      call require(all(ieee_is_finite(y)),'positive linear prediction exceeds working range')
    else
      y=matmul(x,self%coef_)+self%intercept_
    end if
  end function
  real(dp) function linear_score(self,x,y) result(s)
    class(linear_regression), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    s=r2_score(y,self%predict(x))
  end function
  subroutine lasso_fit(self,x,y)
    class(lasso), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    self%l1_ratio=1
    call elastic_fit(self,x,y)
  end subroutine
  subroutine elastic_fit(self,x,y)
    class(elastic_net), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp) :: xc(size(x,1),size(x,2)),yc(size(y)),xm(size(x,2)),ym
    real(dp) :: residual(size(y)),old,update,rho,den,change
    integer :: it,j,n
    call check_positive_regression(self)
    if(self%positive) then
      call require(all(ieee_is_finite([self%alpha,self%l1_ratio,self%tol])),'positive elastic controls must be finite')
      call positive_data_check(x,y)
    end if
    call require(self%alpha>=0.and.self%l1_ratio>=0.and.self%l1_ratio<=1,'invalid elastic net penalties')
    call require(self%max_iter>0.and.self%tol>0,'invalid elastic net iteration controls')
    call centered(x,y,self%fit_intercept,xc,yc,xm,ym); n=size(y)
    self%coef_=spread(0._dp,1,size(x,2)); residual=yc; self%converged_=.false.
    do it=1,self%max_iter
      change=0
      do j=1,size(x,2)
        old=self%coef_(j); residual=residual+xc(:,j)*old
        rho=dot_product(xc(:,j),residual)/n
        den=sum(xc(:,j)**2)/n+self%alpha*(1-self%l1_ratio)
        update=0
        if(den>0) then
          if(self%positive) then
            update=ieee_quotient(max(rho-self%alpha*self%l1_ratio,0._dp),den)
            call require(ieee_is_finite(update),'positive elastic coordinate exceeds working range')
          else
            update=sign(max(abs(rho)-self%alpha*self%l1_ratio,0._dp),rho)/den
          end if
        end if
        self%coef_(j)=update; residual=residual-xc(:,j)*update
        change=max(change,abs(update-old))
      end do
      if(change<self%tol*max(1._dp,maxval(abs(self%coef_)))) then
        self%converged_=.true.; exit
      end if
    end do
    self%n_iter_=min(it,self%max_iter)
    if(self%positive) then
      self%intercept_=ieee_difference(ym,positive_dot(xm,self%coef_))
      call require(ieee_is_finite(self%intercept_),'positive linear intercept exceeds working range')
    else
      self%intercept_=ym-dot_product(xm,self%coef_)
    end if
  end subroutine
  subroutine lasso_cv_fit(self,x,y)
    class(lasso_cv), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    self%l1_ratios=[1._dp]; self%l1_ratio=1
    call elastic_cv_fit(self,x,y)
  end subroutine

  subroutine elastic_cv_fit(self,x,y)
    class(elastic_net_cv), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    type(elastic_net) :: model
    real(dp), allocatable :: ratios(:),grid(:),xc(:,:),yc(:),xm(:),prediction(:)
    integer, allocatable :: indices(:),train(:)
    real(dp) :: ym,alpha_max,mse
    integer :: n,p,na,nr,r,a,f,first,last,fold_size,i,best_a,best_r
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1])); n=size(y); p=size(x,2)
    if(self%positive) call positive_data_check(x,y)
    call require(self%cv>=2.and.self%cv<=n,'invalid coordinate CV fold count')
    call require(ieee_is_finite(self%tol),'nonfinite coordinate CV tolerance')
    call require(self%tol>0.and.self%max_iter>0,'invalid coordinate CV iteration controls')
    ratios=[self%l1_ratio]
    if(allocated(self%l1_ratios)) ratios=self%l1_ratios
    call require(size(ratios)>0,'empty coordinate CV l1 ratios')
    call require(all(ieee_is_finite(ratios)),'nonfinite coordinate CV l1 ratios')
    call require(all(ratios>0).and.all(ratios<=1),'coordinate CV l1 ratios must be in (0,1]')
    nr=size(ratios)
    if(allocated(self%alphas)) then
      call require(size(self%alphas)>0,'empty coordinate CV alpha grid')
      call require(all(ieee_is_finite(self%alphas)),'nonfinite coordinate CV alphas')
      call require(all(self%alphas>0),'coordinate CV alphas must be positive')
      grid=self%alphas(argsort(-self%alphas)); na=size(grid)
    else
      call require(self%n_alphas>0,'coordinate CV n_alphas must be positive')
      call require(ieee_is_finite(self%eps),'nonfinite coordinate CV eps')
      call require(self%eps>0.and.self%eps<=1,'coordinate CV eps must be in (0,1]')
      na=self%n_alphas
    end if
    if(allocated(self%alphas_)) deallocate(self%alphas_,self%mse_path_)
    allocate(self%alphas_(na,nr),self%mse_path_(na,self%cv,nr))
    allocate(xc(n,p),yc(n),xm(p)); call centered(x,y,self%fit_intercept,xc,yc,xm,ym)
    do r=1,nr
      if(allocated(self%alphas)) then
        self%alphas_(:,r)=grid
      else
        alpha_max=maxval(abs(matmul(transpose(xc),yc)))/(n*ratios(r))
        if(alpha_max<=1.e-15_dp) then
          self%alphas_(:,r)=1.e-15_dp
        else
          self%alphas_(:,r)=[(alpha_max*exp(log(self%eps)*real(a-1,dp)/max(1,na-1)),a=1,na)]
        end if
      end if
    end do
    indices=[(i,i=1,n)]; self%all_cv_converged_=.true.
    model%fit_intercept=self%fit_intercept; model%positive=self%positive
    model%max_iter=self%max_iter; model%tol=self%tol
    do r=1,nr
      model%l1_ratio=ratios(r); first=1
      do f=1,self%cv
        fold_size=n/self%cv
        if(f<=mod(n,self%cv)) fold_size=fold_size+1
        last=first+fold_size-1; train=pack(indices,indices<first.or.indices>last)
        do a=1,na
          model%alpha=self%alphas_(a,r); call model%fit(x(train,:),y(train))
          prediction=model%predict(x(first:last,:))
          self%mse_path_(a,f,r)=sum((prediction-y(first:last))**2)/fold_size
          self%all_cv_converged_=self%all_cv_converged_.and.model%converged_
        end do
        first=last+1
      end do
    end do
    self%best_mse_=huge(1._dp); best_a=1; best_r=1
    do r=1,nr
      do a=1,na
        mse=sum(self%mse_path_(a,:,r))/self%cv
        if(mse<self%best_mse_) then
          self%best_mse_=mse; best_a=a; best_r=r
        end if
      end do
    end do
    self%alpha_=self%alphas_(best_a,best_r); self%l1_ratio_=ratios(best_r)
    self%alpha=self%alpha_; self%l1_ratio=self%l1_ratio_
    call elastic_fit(self,x,y); self%n_features_in_=p
  end subroutine

  subroutine logistic_fit(self,x,y,sample_weight)
    class(logistic_regression), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    call logistic_fit_core(self,x,y,sample_weight)
  end subroutine
  subroutine logistic_fit_core(self,x,y,sample_weight,initial_coef,initial_intercept,classes)
    class(logistic_regression), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), intent(in), optional :: initial_coef(:,:),initial_intercept(:)
    integer, intent(in), optional :: classes(:)
    real(dp) :: observation_weight(size(y)),mass,gradient_tol
    real(dp), allocatable :: design(:,:),w(:,:),prob(:,:),gradient(:,:),hess(:,:),step(:),trial(:,:),scaling(:)
    real(dp) :: objective,new_objective,rate,penalty,v,objective_slack
    integer :: n,p,k,q,it,i,j,a,b,ia,ib,target
    call check_xy(x,size(y)); self%classes_=unique_labels(y)
    if(present(classes)) self%classes_=classes
    k=size(self%classes_)
    call require(k>=2,'logistic regression requires at least two classes')
    if(present(initial_coef)) then
      call require(self%c>0.and.self%tol>0.and.self%max_iter>=0,'invalid logistic controls')
    else
      call require(self%c>0.and.self%tol>0.and.self%max_iter>0,'invalid logistic controls')
    end if
    n=size(x,1); p=size(x,2); q=p+merge(1,0,self%fit_intercept)
    ! Binary has one logit; multiclass uses one logit per class and softmax.
    if(k==2) k=1
    allocate(design(n,q),w(q,k),prob(n,k),gradient(q,k),hess(q*k,q*k),step(q*k),trial(q,k),scaling(q*k))
    design(:,:p)=x; if(q>p) design(:,q)=1
    observation_weight=1; gradient_tol=self%tol
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'logistic sample weight length mismatch')
      call require(all(ieee_is_finite(sample_weight)),'logistic sample weights must be finite')
      call require(all(sample_weight>=0),'logistic sample weights must be nonnegative')
      call require(maxval(sample_weight)<=sqrt(huge(1._dp))/(16*real(n,dp)), &
        'logistic sample weight mass exceeds working range')
      mass=sum(sample_weight)
      call require(mass>0,'logistic sample weights require positive total')
      if(ieee_is_finite(self%c)) then
        call require(log(self%c)+log(mass)>=-log(huge(1._dp)/16), &
          'weighted logistic penalty exceeds working range')
        penalty=0
        if(mass<=1) then
          penalty=1/(self%c*mass)
        else if(self%c<=huge(1._dp)/mass) then
          penalty=1/(self%c*mass)
        end if
      else
        penalty=0
      end if
      observation_weight=sample_weight/mass
      if(mass>=1) gradient_tol=self%tol/mass
    else
      call require(self%c>=1/huge(1._dp),'logistic C is below native working range')
      penalty=1/self%c
    end if
    w=0
    if(present(initial_coef)) then
      call require(all(shape(initial_coef)==[k,p]),'logistic initial coefficient shape mismatch')
      call require(all(ieee_is_finite(initial_coef)),'logistic initial coefficients must be finite')
      w(:p,:)=transpose(initial_coef)
    end if
    if(present(initial_intercept).and.q>p) then
      call require(size(initial_intercept)==k,'logistic initial intercept shape mismatch')
      call require(all(ieee_is_finite(initial_intercept)),'logistic initial intercepts must be finite')
      w(q,:)=initial_intercept
    end if
    self%converged_=.false.
    do it=1,self%max_iter
      call evaluate(w,objective,prob)
      gradient=0; hess=0
      do i=1,n
        target=label_index(self%classes_,y(i))
        do a=1,k
          v=prob(i,a)
          if((k==1.and.target==2).or.(k>1.and.target==a)) v=v-1
          gradient(:,a)=gradient(:,a)+design(i,:)*(v*observation_weight(i))
          do b=1,k
            v=prob(i,a)*(merge(1._dp,0._dp,a==b)-prob(i,b))*observation_weight(i)
            do j=1,q
              ia=(a-1)*q+j
              hess(ia,(b-1)*q+1:b*q)=hess(ia,(b-1)*q+1:b*q)+v*design(i,j)*design(i,:)
            end do
          end do
        end do
      end do
      gradient(:p,:)=gradient(:p,:)+penalty*w(:p,:)
      do a=1,k
        do j=1,q
          ia=(a-1)*q+j
          if(j<=p) hess(ia,ia)=hess(ia,ia)+penalty
          hess(ia,ia)=hess(ia,ia)+1.e-10_dp
        end do
      end do
      if(maxval(abs(gradient))<gradient_tol) then
        self%converged_=.true.; exit
      end if
      if(present(sample_weight)) then
        ! Equilibrate the weighted Newton system, especially for tiny total
        ! weight with an unregularized intercept and a large scaled penalty.
        do j=1,q*k
          scaling(j)=sqrt(hess(j,j))
        end do
        hess=hess/spread(scaling,2,q*k)/spread(scaling,1,q*k)
        step=solve_system(hess,reshape(gradient,[q*k])/scaling)/scaling
      else
        step=solve_system(hess,reshape(gradient,[q*k]))
      end if
      rate=1
      ! Near a minimizer the predicted decrease can be below evaluation
      ! roundoff. Permit that roundoff so backtracking cannot falsely satisfy
      ! the small-step convergence test while the gradient is still too large.
      objective_slack=10*epsilon(1._dp)*max(1._dp,abs(objective))
      do j=1,50
        trial=w-rate*reshape(step,[q,k]); call evaluate(trial,new_objective,prob)
        if(new_objective<=objective-1.e-4_dp*rate*sum(reshape(gradient,[q*k])*step)+objective_slack) exit
        rate=rate/2
      end do
      w=trial
      if(maxval(abs(rate*step))<self%tol) then
        self%converged_=.true.; exit
      end if
    end do
    self%n_iter_=min(it,self%max_iter)
    self%coef_=transpose(w(:p,:)); self%intercept_=spread(0._dp,1,k)
    if(q>p) self%intercept_=w(q,:)
    ! Remove the softmax intercept null mode: fold-dependent numerical drift
    ! changes raw scores used by downstream calibration despite identical proba.
    if(k>1) self%intercept_=self%intercept_-sum(self%intercept_)/k
  contains
    subroutine evaluate(weights,loss,probabilities)
      real(dp), intent(in) :: weights(:,:)
      real(dp), intent(out) :: loss,probabilities(:,:)
      real(dp) :: logits(k),normalizer,z
      integer :: row,cl
      loss=0.5_dp*penalty*sum(weights(:p,:)**2)
      do row=1,n
        logits=matmul(design(row,:),weights)
        cl=label_index(self%classes_,y(row))
        if(k==1) then
          z=logits(1); probabilities(row,1)=sigmoid(z)
          loss=loss+observation_weight(row)*max(z,0._dp)+observation_weight(row)*log(1+exp(-abs(z)))
          if(cl==2) loss=loss-observation_weight(row)*z
        else
          normalizer=logsumexp(logits); probabilities(row,:)=exp(logits-normalizer)
          loss=loss+observation_weight(row)*normalizer-observation_weight(row)*logits(cl)
        end if
      end do
    end subroutine
  end subroutine
  function logistic_decision(self,x) result(z)
    class(logistic_regression), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call require(allocated(self%coef_),'logistic_regression is not fitted')
    call check_xy(transpose(x),size(self%coef_,2))
    z=matmul(x,transpose(self%coef_))+spread(self%intercept_,1,size(x,1))
  end function
  function logistic_proba(self,x) result(prob)
    class(logistic_regression), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: prob(:,:),z(:,:)
    integer :: i,k
    z=self%decision_function(x); k=size(self%classes_); allocate(prob(size(x,1),k))
    if(k==2) then
      prob(:,2)=sigmoid(z(:,1)); prob(:,1)=1-prob(:,2)
    else
      do i=1,size(x,1)
        prob(i,:)=exp(z(i,:)-logsumexp(z(i,:)))
      end do
    end if
  end function
  function logistic_predict(self,x) result(y)
    class(logistic_regression), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: y(size(x,1)),i
    real(dp), allocatable :: decision(:,:)
    decision=self%decision_function(x)
    if(size(self%classes_)==2) then
      y=self%classes_(merge(2,1,decision(:,1)>0))
    else
      do i=1,size(y)
        y(i)=self%classes_(maxloc(decision(i,:),dim=1))
      end do
    end if
  end function
  real(dp) function logistic_score(self,x,y,sample_weight) result(s)
    class(logistic_regression), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    if(.not.present(sample_weight)) then
      s=accuracy_score(y,self%predict(x)); return
    end if
    call require(size(y)==size(x,1),'logistic score target length mismatch')
    call score_weights(sample_weight,size(y),'logistic')
    s=ieee_mean(merge(1._dp,0._dp,y==self%predict(x)),sample_weight)
  end function
  logical function logistic_cv_scoring_supported(scoring) result(supported)
    character(*), intent(in) :: scoring
    supported=any(scoring==[character(20)::'accuracy','balanced_accuracy','neg_log_loss','roc_auc'])
  end function

  real(dp) function logistic_cv_measure(model,x,y,scoring,sample_weight) result(value)
    class(logistic_regression), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    character(*), intent(in) :: scoring
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: weights(:),prob(:,:),decision(:,:),terms(:)
    integer, allocatable :: predicted(:),classes(:)
    real(dp) :: mass,total
    integer :: i,j,nused
    call check_xy(x,size(y)); call score_weights(sample_weight,size(y),'LogisticRegressionCV')
    allocate(weights(size(y))); weights=1
    if(present(sample_weight)) weights=sample_weight
    call require(all(weights>=0),'LogisticRegressionCV score weights must be nonnegative')
    select case(scoring)
    case('accuracy')
      value=ieee_mean(merge(1._dp,0._dp,y==model%predict(x)),sample_weight)
    case('balanced_accuracy')
      predicted=model%predict(x); classes=unique_labels(y); total=0; nused=0
      ! Normalize weights before class sums to protect the scorer from overflow.
      weights=weights/maxval(weights)
      do j=1,size(classes)
        mass=sum(pack(weights,y==classes(j))); if(mass==0) cycle
        total=total+sum(pack(weights,y==classes(j).and.predicted==y))/mass; nused=nused+1
      end do
      value=total/nused
    case('neg_log_loss')
      prob=model%predict_proba(x); classes=unique_labels(y)
      call require(size(classes)>=2,'LogisticRegressionCV log loss needs at least two test classes')
      call require(size(classes)==size(prob,2),'LogisticRegressionCV log loss test classes do not match probabilities')
      allocate(terms(size(y)))
      do i=1,size(y)
        j=label_index(classes,y(i))
        terms(i)=log(max(epsilon(1._dp),min(1-epsilon(1._dp),prob(i,j))))
      end do
      value=ieee_mean(terms,sample_weight)
    case('roc_auc')
      call require(size(model%classes_)==2,'LogisticRegressionCV ROC AUC requires a binary model')
      decision=model%decision_function(x); value=roc_auc_score(y,decision(:,1),sample_weight=sample_weight)
    case default
      call require(.false.,'unknown LogisticRegressionCV scoring mode')
    end select
  end function

  real(dp) function logistic_cv_score(self,x,y,sample_weight) result(value)
    class(logistic_regression_cv), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    call require(allocated(self%coef_),'LogisticRegressionCV is not fitted')
    value=logistic_cv_measure(self,x,y,self%scoring,sample_weight)
  end function

  subroutine logistic_cv_folds(self,y)
    class(logistic_regression_cv), intent(inout) :: self
    integer, intent(in) :: y(:)
    integer, allocatable :: labels(:),counts(:),assignment(:),members(:)
    integer :: i,j,c,n,offset,position,width
    n=size(y)
    if(allocated(self%splits)) then
      self%cv_splits_=self%splits
      call require(size(self%cv_splits_)>0,'LogisticRegressionCV needs at least one split')
    else
      call require(self%cv>=2.and.self%cv<=n,'invalid LogisticRegressionCV cv count')
      labels=[integer::]
      do i=1,n
        if(.not.any(labels==y(i))) labels=[labels,y(i)]
      end do
      counts=[(count(y==labels(i)),i=1,size(labels))]
      call require(.not.all(counts<self%cv),'LogisticRegressionCV cv exceeds every class count')
      if(allocated(self%cv_splits_)) deallocate(self%cv_splits_)
      allocate(self%cv_splits_(self%cv),assignment(n)); offset=0
      do c=1,size(labels)
        members=pack([(i,i=1,n)],y==labels(c)); position=1
        do j=1,self%cv
          width=count(modulo(offset+[(i,i=0,counts(c)-1)],self%cv)==j-1)
          assignment(members(position:position+width-1))=j; position=position+width
        end do
        offset=offset+counts(c)
      end do
      do j=1,self%cv
        self%cv_splits_(j)%train=pack([(i,i=1,n)],assignment/=j)
        self%cv_splits_(j)%test=pack([(i,i=1,n)],assignment==j)
      end do
    end if
    do j=1,size(self%cv_splits_)
      call require(allocated(self%cv_splits_(j)%train).and.allocated(self%cv_splits_(j)%test), &
        'LogisticRegressionCV split indices are unallocated')
      call require(size(self%cv_splits_(j)%train)>0.and.size(self%cv_splits_(j)%test)>0, &
        'LogisticRegressionCV split indices must be nonempty')
      call require(all(self%cv_splits_(j)%train>=1).and.all(self%cv_splits_(j)%train<=n).and. &
        all(self%cv_splits_(j)%test>=1).and.all(self%cv_splits_(j)%test<=n), &
        'LogisticRegressionCV split index is out of bounds')
    end do
  end subroutine

  integer function logistic_cv_best(scores) result(index)
    real(dp), intent(in) :: scores(:)
    integer :: j
    index=1
    do j=1,size(scores)
      ! NumPy argmax returns the first NaN; otherwise the first maximum wins.
      if(ieee_is_nan(scores(j))) then
        index=j; return
      end if
      if(scores(j)>scores(index)) index=j
    end do
  end function

  subroutine logistic_cv_fit(self,x,y,sample_weight)
    class(logistic_regression_cv), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    type(logistic_regression) :: worker
    real(dp), allocatable :: raw_weight(:),fit_weight(:),initial_coef(:,:),initial_bias(:),averaged(:,:),totals(:)
    integer, allocatable :: train(:),test(:),present_classes(:)
    real(dp) :: bound,mass,chosen_c
    integer :: n,p,k,m,q,nfolds,f,j,c,i,best
    call check_xy(x,size(y)); n=size(y); p=size(x,2)
    self%classes_=unique_labels(y); k=size(self%classes_)
    call require(k>=2,'LogisticRegressionCV requires at least two classes')
    call require(ieee_is_finite(self%tol),'nonfinite LogisticRegressionCV tolerance')
    call require(self%tol>0.and.self%max_iter>=0,'invalid LogisticRegressionCV solver controls')
    call require(logistic_cv_scoring_supported(self%scoring),'unknown LogisticRegressionCV scoring mode')
    if(self%scoring=='roc_auc') call require(k==2,'LogisticRegressionCV ROC AUC requires a binary model')
    call require(maxval(abs(x))<=sqrt(huge(1._dp))/(16*real(n,dp)*real(p,dp)*real(k,dp)), &
      'LogisticRegressionCV data exceeds safe Hessian range')
    if(allocated(self%cs)) then
      call require(size(self%cs)>0,'LogisticRegressionCV C grid must be nonempty')
      call require(all(ieee_is_finite(self%cs)),'LogisticRegressionCV Cs must be finite')
      call require(all(self%cs>0),'LogisticRegressionCV Cs must be positive')
      self%cs_=self%cs
    else
      call require(self%n_cs>0,'LogisticRegressionCV n_cs must be positive')
      if(allocated(self%cs_)) deallocate(self%cs_)
      allocate(self%cs_(self%n_cs)); self%cs_=1.e-4_dp
      if(self%n_cs>1) self%cs_=[(10._dp**(-4+8*real(j-1,dp)/real(self%n_cs-1,dp)),j=1,self%n_cs)]
    end if
    allocate(raw_weight(n),fit_weight(n)); raw_weight=1
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'LogisticRegressionCV sample weight length mismatch')
      call require(all(ieee_is_finite(sample_weight)),'LogisticRegressionCV sample weights must be finite')
      raw_weight=sample_weight
    end if
    call require(all(raw_weight>=0),'LogisticRegressionCV sample weights must be nonnegative')
    bound=sqrt(huge(1._dp))/(16*real(n,dp))
    call require(maxval(raw_weight)<=bound,'LogisticRegressionCV weights exceed working range')
    call require(sum(raw_weight)>0,'LogisticRegressionCV sample weights require positive total')
    call require(.not.(self%class_weight_balanced.and.allocated(self%class_weight)), &
      'LogisticRegressionCV class weight modes are mutually exclusive')
    if(allocated(self%class_weight_)) deallocate(self%class_weight_)
    allocate(self%class_weight_(k)); self%class_weight_=1
    if(allocated(self%class_weight)) then
      call require(size(self%class_weight)==k,'LogisticRegressionCV class weight length mismatch')
      call require(all(ieee_is_finite(self%class_weight)),'LogisticRegressionCV class weights must be finite')
      call require(all(self%class_weight>=0),'LogisticRegressionCV class weights must be nonnegative')
      self%class_weight_=self%class_weight
    else if(self%class_weight_balanced) then
      do c=1,k
        self%class_weight_(c)=real(n,dp)/(real(k,dp)*count(y==self%classes_(c)))
      end do
    end if
    do i=1,n
      c=label_index(self%classes_,y(i)); fit_weight(i)=0
      if(raw_weight(i)==0.or.self%class_weight_(c)==0) cycle
      call require(log(raw_weight(i))+log(self%class_weight_(c))<=log(bound), &
        'LogisticRegressionCV combined weights exceed working range')
      fit_weight(i)=raw_weight(i)*self%class_weight_(c)
    end do
    call require(sum(fit_weight)>0,'LogisticRegressionCV effective weights require positive total')
    call logistic_cv_folds(self,y); nfolds=size(self%cv_splits_)
    m=k; if(k==2) m=1
    q=p+merge(1,0,self%fit_intercept); self%n_features_in_=p
    self%path_classes_=self%classes_; if(k==2) self%path_classes_=self%classes_(2:)
    if(allocated(self%coefs_paths_)) deallocate(self%coefs_paths_,self%n_iter_path_,self%converged_path_)
    if(allocated(self%scores_)) deallocate(self%scores_)
    allocate(self%coefs_paths_(m,nfolds,size(self%cs_),q),self%scores_(nfolds,size(self%cs_)), &
      self%n_iter_path_(nfolds,size(self%cs_)),self%converged_path_(nfolds,size(self%cs_)))
    allocate(initial_coef(m,p),initial_bias(m)); worker%fit_intercept=self%fit_intercept
    worker%tol=self%tol; worker%max_iter=self%max_iter
    do f=1,nfolds
      train=self%cv_splits_(f)%train; test=self%cv_splits_(f)%test
      call require(sum(fit_weight(train))>0,'LogisticRegressionCV training weights require positive total')
      call require(sum(raw_weight(test))>0,'LogisticRegressionCV test weights require positive total')
      present_classes=unique_labels(y(train))
      if(k>2) call require(size(present_classes)==k,'LogisticRegressionCV multiclass training fold lacks a class')
      if(self%fit_intercept.and.self%max_iter>0) then
        do c=1,k
          mass=sum(pack(fit_weight(train),y(train)==self%classes_(c)))
          call require(mass>0,'LogisticRegressionCV intercept fit needs positive class mass in every training fold')
        end do
      end if
      initial_coef=0; initial_bias=0
      do j=1,size(self%cs_)
        worker%c=self%cs_(j)
        call logistic_fit_core(worker,x(train,:),y(train),fit_weight(train),initial_coef,initial_bias,self%classes_)
        self%coefs_paths_(:,f,j,:p)=worker%coef_
        if(q>p) self%coefs_paths_(:,f,j,q)=worker%intercept_
        self%n_iter_path_(f,j)=worker%n_iter_; self%converged_path_(f,j)=worker%converged_
        self%scores_(f,j)=logistic_cv_measure(worker,x(test,:),y(test),self%scoring,raw_weight(test))
        initial_coef=worker%coef_; initial_bias=worker%intercept_
      end do
    end do
    self%all_cv_converged_=all(self%converged_path_)
    allocate(averaged(m,q)); averaged=0
    if(allocated(self%best_c_indices_)) deallocate(self%best_c_indices_)
    allocate(self%best_c_indices_(nfolds))
    if(self%refit) then
      totals=sum(self%scores_,dim=1); best=logistic_cv_best(totals)
      self%best_c_indices_=best; chosen_c=self%cs_(best)
      do f=1,nfolds
        averaged=averaged+self%coefs_paths_(:,f,best,:)/nfolds
      end do
    else
      chosen_c=0
      do f=1,nfolds
        best=logistic_cv_best(self%scores_(f,:)); self%best_c_indices_(f)=best
        chosen_c=chosen_c+self%cs_(best)/nfolds
        averaged=averaged+self%coefs_paths_(:,f,best,:)/nfolds
      end do
    end if
    self%c_=spread(chosen_c,1,m); self%c=chosen_c; self%refit_init_=averaged
    if(self%refit) then
      initial_coef=averaged(:,:p); initial_bias=0; if(q>p) initial_bias=averaged(:,q)
      worker%c=chosen_c
      call logistic_fit_core(worker,x,y,fit_weight,initial_coef,initial_bias,self%classes_)
      self%coef_=worker%coef_; self%intercept_=worker%intercept_
      self%n_iter_=worker%n_iter_; self%converged_=worker%converged_
    else
      self%coef_=averaged(:,:p); self%intercept_=spread(0._dp,1,m)
      if(q>p) self%intercept_=averaged(:,q)
      self%n_iter_=0; self%converged_=.true.
      do f=1,nfolds
        self%converged_=self%converged_.and.self%converged_path_(f,self%best_c_indices_(f))
      end do
    end if
  end subroutine

  subroutine perceptron_fit(self,x,y)
    class(perceptron), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp) :: target,margin
    integer :: it,i,c,k,errors,heads,positive
    call check_xy(x,size(y)); self%classes_=unique_labels(y); k=size(self%classes_)
    call require(k>=2.and.self%max_iter>0.and.self%eta0>0,'invalid perceptron controls')
    if(allocated(self%coef_)) deallocate(self%coef_,self%intercept_)
    heads=k; if(k==2) heads=1
    allocate(self%coef_(heads,size(x,2)),self%intercept_(heads)); self%coef_=0; self%intercept_=0
    do it=1,self%max_iter
      errors=0
      do i=1,size(y)
        do c=1,heads
          positive=c; if(k==2) positive=2
          target=merge(1._dp,-1._dp,y(i)==self%classes_(positive))
          margin=dot_product(self%coef_(c,:),x(i,:))+self%intercept_(c)
          if(target*margin<=0) then
            self%coef_(c,:)=self%coef_(c,:)+self%eta0*target*x(i,:)
            if(self%fit_intercept) self%intercept_(c)=self%intercept_(c)+self%eta0*target
            errors=errors+1
          end if
        end do
      end do
      if(errors==0) exit
    end do
    self%n_iter_=min(it,self%max_iter)
  end subroutine
  function perceptron_decision(self,x) result(z)
    class(perceptron), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call require(allocated(self%coef_),'perceptron is not fitted')
    call check_xy(transpose(x),size(self%coef_,2))
    z=matmul(x,transpose(self%coef_))+spread(self%intercept_,1,size(x,1))
  end function
  function perceptron_predict(self,x) result(y)
    class(perceptron), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: y(size(x,1)),i
    real(dp), allocatable :: z(:,:)
    z=self%decision_function(x)
    do i=1,size(y)
      if(size(self%classes_)==2) then
        y(i)=self%classes_(merge(2,1,z(i,1)>0))
      else
        y(i)=self%classes_(maxloc(z(i,:),dim=1))
      end if
    end do
  end function
  real(dp) function perceptron_score(self,x,y) result(s)
    class(perceptron), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    s=accuracy_score(y,self%predict(x))
  end function
end module
