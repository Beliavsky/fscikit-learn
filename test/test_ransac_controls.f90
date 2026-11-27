program test_ransac_controls
  use sklearn_common,only:dp,require,seed_rng
  use sklearn_linear_model,only:ransac_regressor,linear_regression
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  use sklearn_base_dispatch,only:base_weighted,train_regressor
  use sklearn_search_adapters,only:set_search_params
  use sklearn_parameter_space,only:parameter_set,param
  use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan
  implicit none
  type(ransac_regressor)::model,repeat,multi
  type(csr_matrix)::sparse
  type(parameter_set)::parameters
  real(dp)::x(40,2),y(40),target(40,2),weight(40),a,b,c,d
  real(dp),allocatable::prediction(:,:),vector(:)
  logical::mask(40)
  integer::i,calls=0
  do i=1,40
    x(i,:)=[real(i-1,dp),sin(real(i,dp))]
    y(i)=.03_dp*x(i,1)+1.2_dp*x(i,2)+.7_dp+.01_dp*cos(real(3*i,dp))
    if(mod(i,7)==0)y(i)=y(i)+4
    weight(i)=.5_dp+.2_dp*mod(i,3)
  end do
  target(:,1)=y;target(:,2)=2*y+.2_dp;sparse=csr_from_dense(x)
  allocate(parameters%items(3))
  parameters%items(1)%name='min_samples';parameters%items(1)%value=param(.2_dp)
  parameters%items(2)%name='estimator__fit_intercept';parameters%items(2)%value=param(.true.)
  parameters%items(3)%name='residual_threshold';parameters%items(3)%value=param(.1_dp)
  call set_search_params(model,parameters);model%random_state=42
  call seed_rng(817);call random_number(a)
  call model%fit(x,y,weight);call random_number(b)
  call seed_rng(817);call random_number(c);call random_number(d)
  call require(a==c.and.b==d,'RANSAC local NumPy RNG preserves global Fortran RNG state')
  call require(model%min_samples_==8.and.model%n_features_in_==2,'RANSAC fractional sample and learned shape')
  call require(all(.not.model%inlier_mask_(7::7)),'RANSAC rejects separated outliers')
  call require(maxval(abs(model%coef_-[.03_dp,1.2_dp]))<.02_dp,'RANSAC weighted final OLS recovery')
  call require(allocated(model%estimator_%coef_),'RANSAC fitted estimator state')
  repeat=model;call repeat%fit(x,y,weight)
  call require(all(model%inlier_mask_.eqv.repeat%inlier_mask_).and.model%n_trials_==repeat%n_trials_, &
    'RANSAC seeded repeated fits')
  call require(base_weighted(model),'RANSAC common weighted capability')
  call train_regressor(repeat,x,y,weight)
  call require(maxval(abs(repeat%coef_-model%coef_))<1.e-12_dp,'RANSAC weighted common dispatch')
  multi=model;call multi%fit(sparse,target,weight)
  prediction=multi%predict_matrix(sparse)
  call require(all(shape(prediction)==[40,2]).and.multi%n_targets_==2,'RANSAC joint sparse output shape')
  call require(maxval(abs(multi%coef_matrix_(2,:)-2*multi%coef_matrix_(1,:)))<.001_dp, &
    'RANSAC joint OLS target relation')
  call require(abs(multi%score(sparse,target,weight)-multi%estimator_%score(sparse,target,weight))<1.e-12_dp, &
    'RANSAC matrix weighted score reaches fitted base')
  repeat=model;repeat%is_data_valid=>check_data;repeat%is_model_valid=>check_model
  repeat%loss_callback=>loss;call repeat%fit(sparse,y,weight)
  call require(calls>0.and..not.repeat%inlier_mask_(1),'RANSAC dense subset callbacks and NaN residual exclusion')
  repeat=ransac_regressor();call repeat%set_min_samples(1._dp)
  repeat%residual_threshold=.0001_dp;repeat%stop_probability=0;repeat%random_state=0
  call repeat%fit(x,y)
  call require(repeat%min_samples_==1.and.repeat%n_trials_==1,'RANSAC real one is a count; zero probability stops')
  call require(count(repeat%inlier_mask_)==1,'RANSAC NaN one-inlier candidate score follows Source comparison')
  call repeat%set_min_samples(.25_dp);call repeat%clear_min_samples();call repeat%fit(x,y)
  call require(repeat%min_samples_==3,'RANSAC clearing typed sample control restores default')
  print *, 'RANSAC NumPy RNG, weights, joint CSR, callbacks, search and typed minimum samples: PASS'
contains
  logical function check_data(a,b)result(ok)
    real(dp),intent(in)::a(:,:),b(:,:)
    calls=calls+1;ok=.true.
  end function
  logical function check_model(candidate,a,b)result(ok)
    type(linear_regression),intent(in)::candidate
    real(dp),intent(in)::a(:,:),b(:,:)
    ok=allocated(candidate%coef_matrix_)
  end function
  function loss(actual,predicted)result(residual)
    real(dp),intent(in)::actual(:,:),predicted(:,:)
    real(dp)::residual(size(actual,1))
    residual=sum(abs(actual-predicted),dim=2);residual(1)=ieee_value(0._dp,ieee_quiet_nan)
  end function
end program
