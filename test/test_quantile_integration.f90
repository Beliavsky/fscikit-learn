program test_quantile_integration
  use sklearn_common,only:dp
  use sklearn_quantile,only:quantile_regressor
  use sklearn_search,only:grid_search_cv,param
  use sklearn_pipeline,only:pipeline
  use sklearn_preprocessing,only:standard_scaler
  use sklearn_model_selection,only:cross_val_score
  use sklearn_common,only:require
  use sklearn_sgd_clone,only:clone_quantile_parameters
  use sklearn_base_dispatch,only:base_fitted,base_supported,base_weighted
  implicit none
  type(quantile_regressor)::base,copy
  type(grid_search_cv)::search
  type(pipeline)::pipe
  type(standard_scaler)::scale
  real(dp)::x(24,2),y(24),w(24)
  real(dp),allocatable::scores(:),before(:)
  integer::i
  do i=1,24
    x(i,:)=[real(i-12,dp)/24,sin(real(i,dp))]
  end do
  y=.7_dp+.4_dp*x(:,1)-.2_dp*x(:,2);w=1;w(:2)=0
  base%alpha=0;call base%fit(x,y);before=base%coef_;copy=clone_quantile_parameters(base)
  call require(.not.allocated(copy%coef_).and..not.allocated(copy%dual_coef_),'quantile clone learned state')
  call require(copy%n_features_in_==0.and.copy%n_iter_==0.and..not.copy%converged_,'quantile clone diagnostics')
  call require(base_fitted(base).and..not.base_fitted(copy),'quantile fitted dispatch')
  call require(base_supported(base,.false.).and.base_weighted(base),'quantile regressor dispatch')
  scores=cross_val_score(base,x,y,cv=4,sample_weight=w)
  call require(minval(scores)>1-1.e-10_dp,'quantile weighted CV')
  allocate(search%estimator,source=base);allocate(search%param_grid(1));allocate(search%param_grid(1)%axes(1))
  search%param_grid(1)%axes(1)%name='alpha';search%param_grid(1)%axes(1)%values=[param(0._dp),param(.1_dp)]
  search%cv=4;call search%fit(x,y,w)
  call require(search%best_index_==1.and.maxval(abs(search%predict(x)-y))<1.e-10_dp,'quantile GridSearchCV refit')
  call require(all(base%coef_==before),'quantile caller retained')
  allocate(pipe%steps(2));pipe%steps(1)%name='scale';pipe%steps(2)%name='model'
  allocate(pipe%steps(1)%model,source=scale);allocate(pipe%steps(2)%model,source=base)
  call pipe%fit(x,y,w)
  call require(maxval(abs(pipe%predict(x)-y))<1.e-10_dp,'quantile weighted Pipeline')
  print *,'Quantile clones, cross-validation, search and weighted Pipeline: PASS'
end program
