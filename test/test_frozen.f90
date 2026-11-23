program test_frozen
  use sklearn_common, only: dp,require
  use sklearn_frozen, only: frozen_estimator,frozen_parameters
  use sklearn_pipeline, only: pipeline
  use sklearn_preprocessing, only: standard_scaler,normalizer,polynomial_features,function_transformer
  use sklearn_linear_model, only: linear_regression,ridge
  use sklearn_naive_bayes, only: gaussian_nb
  use sklearn_neural_network, only: mlp_regressor,mlp_classifier
  use sklearn_calibrated_classifier, only: calibrated_classifier_cv
  use sklearn_decomposition, only: pca
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_is_finite
  implicit none
  type(standard_scaler), target :: scale
  type(linear_regression), target :: reg
  type(gaussian_nb), target :: nb
  type(normalizer), target :: norm
  type(polynomial_features), target :: poly
  type(function_transformer), target :: identity
  type(mlp_regressor), target :: joint
  type(mlp_classifier), target :: multilabel
  type(pca), target :: projection
  type(pipeline), target :: pipe,solo,classpipe
  type(frozen_estimator), target :: frozen,scale_view,nb_view,outer
  class(frozen_estimator), pointer :: alias
  type(frozen_parameters) :: params
  type(ridge) :: manual
  type(calibrated_classifier_cv) :: calibrated
  real(dp) :: x(9,3),y(9),q(4,3),nan,old_mean(3),responses(9,2)
  real(dp), allocatable :: a(:,:),b(:,:),v(:),logp(:,:)
  integer :: i,labels(9),indicators(9,2)
  do i=1,9
    x(i,:)=[real(i,dp)/9,sin(real(i,dp)),cos(real(i,dp))]
    y(i)=1+.3_dp*x(i,1)-.8_dp*x(i,2)+1.4_dp*x(i,3);labels(i)=3+4*mod(i,3)
  end do
  q=x(:4,:);nan=ieee_value(0._dp,ieee_quiet_nan)
  call scale%fit(x);old_mean=scale%mean_;scale_view%estimator=>scale
  call require(scale_view%is_fitted().and.scale_view%can_transform(),'frozen scaler fitted capability')
  call scale_view%fit(42,'ignored',sample_weight=nan)
  call scale_view%fit(x(:0,:),[integer::])
  call require(all(scale%mean_==old_mean),'FrozenEstimator fit ignores all data and weights')
  a=scale_view%transform(q);b=scale%transform(q)
  call require(all(a==b),'frozen transform delegates without a copy of the model')
  b=scale_view%inverse_transform(a);call require(maxval(abs(b-q))<1.e-14_dp,'frozen inverse delegation')
  call require(size(scale_view%get_feature_names_out())==3,'frozen feature names')
  params=scale_view%get_params(.true.)
  call require(associated(params%estimator,scale),'frozen params preserve estimator identity')
  alias=>scale_view%clone();call require(associated(alias,scale_view),'frozen clone aliases the wrapper')
  scale%mean_=old_mean+1;b=scale_view%transform(q);a=scale%transform(q)
  call require(all(a==b),'external model mutation is visible through a frozen wrapper')
  scale%mean_=old_mean
  call reg%fit(x,y);call frozen%set_params(reg)
  call frozen%fit(q,spread(nan,1,1));v=frozen%predict(q)
  call require(maxval(abs(v-reg%predict(q)))==0,'frozen regression delegation')
  call require(abs(frozen%score(x,y)-1)<1.e-12_dp,'frozen regression score')
  call nb%fit(x,labels);nb_view%estimator=>nb
  call nb_view%fit(x+10,labels+100)
  call require(all(nb_view%predict_class(q)==nb%predict(q)),'frozen classifier keeps fitted labels')
  a=nb_view%predict_proba(q);b=nb%predict_proba(q);call require(all(a==b),'frozen probabilities')
  logp=nb_view%predict_log_proba(q*10000);b=nb%predict_log_proba(q*10000)
  call require(all(ieee_is_finite(logp)).and.all(logp==b),'frozen Gaussian log probabilities remain stable')
  call require(all(nb_view%get_classes()==nb%classes_),'frozen live class metadata')
  call require(.not.nb_view%can_transform().and..not.nb_view%has_method('fit_transform'), &
    'frozen predictor is not exposed as a transformer')
  call require(.not.nb_view%has_method('fit_predict'),'frozen fit_predict is unavailable')
  call alias%set_params(reg)
  call require(scale_view%bridge_kind()==2,'identity-preserving clone observes wrapper rebinding')
  scale_view%estimator=>scale
  frozen%estimator=>norm;call require(frozen%is_fitted(),'stateless normalizer needs no fit')
  call frozen%fit();a=frozen%transform(q);b=norm%transform(q)
  call require(all(a==b).and.norm%n_features_in_==0,'frozen stateless fit does not initialize metadata')
  frozen%estimator=>poly;call require(.not.frozen%is_fitted(),'unfitted polynomial is not fitted')
  call poly%fit(x);call frozen%fit();a=frozen%transform(q);b=poly%transform(q)
  call require(all(a==b),'frozen polynomial bridge delegation')
  frozen%estimator=>identity;call frozen%fit();a=frozen%transform(q)
  call require(all(a==q).and..not.identity%inverse_checked_,'frozen stateless function avoids fit side effects')
  call require(.not.frozen%has_method('get_feature_names_out'),'conditional function-transformer names')
  allocate(pipe%steps(2));pipe%steps(1)%name='fixed';pipe%steps(2)%name='ridge'
  allocate(pipe%steps(1)%model,source=scale_view);allocate(pipe%steps(2)%model,source=manual)
  call pipe%fit(x+5,y);call manual%fit(scale%transform(x+5),y)
  call require(all(scale%mean_==old_mean),'pipeline keeps a frozen transformer fitted on original data')
  call require(maxval(abs(pipe%predict(q)-manual%predict(scale%transform(q))))<1.e-12_dp,'frozen pipeline transform flow')
  frozen%estimator=>pipe;v=frozen%predict(q)
  call require(all(v==pipe%predict(q)).and.frozen%bridge_kind()==2,'frozen fitted predictor pipeline')
  allocate(solo%steps(1));solo%steps(1)%name='fixed';frozen%estimator=>reg
  allocate(solo%steps(1)%model,source=frozen)
  call solo%fit(x(:2,:1),[real(dp)::]);call require(all(solo%predict(q)==reg%predict(q)), &
    'frozen-only regressor pipeline ignores fit data and empty target')
  allocate(classpipe%steps(1));classpipe%steps(1)%name='fixed'
  allocate(classpipe%steps(1)%model,source=nb_view)
  call classpipe%fit(x(:2,:1))
  call require(all(classpipe%predict_class(q)==nb%predict(q)),'frozen-only classifier pipeline needs no fit target')
  frozen%estimator=>classpipe;a=frozen%predict_log_proba(q);b=nb%predict_log_proba(q)
  call require(all(a==b).and..not.frozen%can_transform(),'frozen classifier pipeline log delegation and capabilities')
  outer%estimator=>nb_view;call outer%fit()
  call require(all(outer%predict_class(q)==nb%predict(q)),'nested frozen wrappers')
  responses(:,1)=y;responses(:,2)=y*2;joint%hidden_layer_sizes=[integer::];joint%max_iter=2
  call joint%fit(x,responses);frozen%estimator=>joint;a=frozen%predict_matrix(q);b=joint%predict_multioutput(q)
  call require(all(a==b),'frozen joint-response MLP')
  indicators(:,1)=mod([(i,i=1,9)],2);indicators(:,2)=1-indicators(:,1)
  multilabel%hidden_layer_sizes=[integer::];multilabel%max_iter=2;call multilabel%fit(x,indicators)
  frozen%estimator=>multilabel
  call require(all(frozen%predict_indicator(q)==multilabel%predict_indicator(q)),'frozen multilabel MLP')
  calibrated%prefit=.true.;calibrated%method='isotonic';a=nb%theta_
  call calibrated%fit(x+2,labels,nb_view)
  call require(all(a==nb%theta_),'prefit frozen calibration leaves the original base unchanged')
  b=calibrated%predict_proba(q)
  call require(all(ieee_is_finite(b)).and.maxval(abs(sum(b,dim=2)-1))<1.e-14_dp,'frozen calibration delegation')
  projection%n_components=2;call projection%fit(x);frozen%estimator=>projection
  call require(frozen%has_method('score').and.frozen%has_method('score_samples'),'frozen PCA score capabilities')
  call require(frozen%score(q)==projection%score(q),'frozen unsupervised score delegation')
  v=frozen%score_samples(q)
  call require(all(v==projection%score_samples(q)),'frozen per-sample likelihood delegation')
  print *, 'Frozen estimator identity, no-op fits, delegation and pipelines: PASS'
end program
