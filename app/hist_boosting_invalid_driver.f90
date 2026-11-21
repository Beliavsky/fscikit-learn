program hist_boosting_invalid_driver
  use sklearn_common, only: dp,require
  use sklearn_hist_gradient_boosting
  use sklearn_legacy_rng, only: legacy_random_state,seed_legacy_uint32
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(hist_gradient_boosting_regressor) :: reg
  type(hist_gradient_boosting_classifier) :: clf
  type(legacy_random_state) :: rng
  real(dp), allocatable :: x(:,:),q(:,:),y(:),w(:),prediction(:),proba(:,:),staged(:,:)
  integer, allocatable :: labels(:)
  real(dp) :: nan,inf
  integer :: j
  logical :: classification
  character(48) :: name
  call get_command_argument(1,name)
  nan=ieee_value(0._dp,ieee_quiet_nan);inf=ieee_value(0._dp,ieee_positive_inf)
  allocate(x(6,2),q(2,2),y(6),w(6),labels(6));x=reshape([(real(j,dp),j=1,12)],[6,2]);q=2
  y=[-1._dp,0._dp,1._dp,2._dp,3._dp,4._dp];w=1;labels=[-3,-3,7,7,11,11]
  reg%max_iter=3;reg%min_samples_leaf=1;clf%max_iter=3;clf%min_samples_leaf=1;classification=.false.
  select case(trim(name))
  case('unfitted_reg');prediction=reg%predict(q);stop
  case('unfitted_staged');staged=reg%staged_predict(q);stop
  case('unfitted_clf');proba=clf%predict_proba(q);stop
  case('uint32_negative');call seed_legacy_uint32(rng,-1_int64);stop
  case('uint32_overflow');call seed_legacy_uint32(rng,4294967296_int64);stop
  case('max_iter_zero');reg%max_iter=0
  case('max_iter_negative');reg%max_iter=-1
  case('predictor_capacity');reg%max_iter=huge(1)
  case('max_leaves_one');reg%max_leaf_nodes=1
  case('max_leaves_negative');reg%max_leaf_nodes=-1
  case('depth_negative');reg%max_depth=-1
  case('leaf_zero');reg%min_samples_leaf=0
  case('leaf_negative');reg%min_samples_leaf=-1
  case('bins_small');reg%max_bins=1
  case('bins_large');reg%max_bins=256
  case('rate_zero');reg%learning_rate=0
  case('rate_negative');reg%learning_rate=-1
  case('rate_nan');reg%learning_rate=nan
  case('rate_inf');reg%learning_rate=inf
  case('l2_negative');reg%l2_regularization=-1
  case('l2_nan');reg%l2_regularization=nan
  case('l2_inf');reg%l2_regularization=inf
  case('tol_negative');reg%tol=-1
  case('tol_nan');reg%tol=nan
  case('tol_inf');reg%tol=inf
  case('patience_zero');reg%n_iter_no_change=0
  case('seed_negative');reg%random_state=-1
  case('monotonic_width');reg%monotonic_cst=[1]
  case('monotonic_value');reg%monotonic_cst=[0,2]
  case('monotonic_minint');reg%monotonic_cst=[0,-huge(1)-1]
  case('loss_reg');reg%loss='bad'
  case('quantile_low');reg%quantile=-.1_dp
  case('quantile_high');reg%quantile=1.1_dp
  case('quantile_nan');reg%quantile=nan
  case('quantile_inf');reg%quantile=inf
  case('poisson_negative');reg%loss='poisson'
  case('poisson_zero');reg%loss='poisson';y=0
  case('gamma_zero');reg%loss='gamma';y=1;y(1)=0
  case('gamma_negative');reg%loss='gamma'
  case('y_nan');y(1)=nan
  case('y_inf');y(1)=inf
  case('y_count');deallocate(y);allocate(y(5));y=1
  case('weights_count');deallocate(w);allocate(w(5));w=1
  case('weights_zero');w=0
  case('weights_negative');w(1)=-1
  case('weights_nan');w(1)=nan
  case('weights_inf');w(1)=inf
  case('weights_overflow');w=1.e308_dp
  case('baseline_overflow');y=1.e308_dp
  case('gradient_range');y=[(real(j,dp)*1.e100_dp,j=1,6)]
  case('empty_fit');deallocate(x,y,w);allocate(x(0,2),y(0),w(0))
  case('zero_width');deallocate(x);allocate(x(6,0))
  case('query_width');deallocate(q);allocate(q(2,3));q=1
  case('empty_query');deallocate(q);allocate(q(0,2))
  case('loss_clf');classification=.true.;clf%loss='bad'
  case('class_mode');classification=.true.;clf%class_weight='bad'
  case('class_weights_width');classification=.true.;clf%class_weights=[1._dp]
  case('class_weights_negative');classification=.true.;clf%class_weights=[1._dp,-1._dp,1._dp]
  case('class_weights_nan');classification=.true.;clf%class_weights=[1._dp,nan,1._dp]
  case('class_weights_inf');classification=.true.;clf%class_weights=[1._dp,inf,1._dp]
  case('class_weights_zero');classification=.true.;clf%class_weights=[0._dp,0._dp,0._dp]
  case('class_weights_overflow');classification=.true.;w=1.e100_dp;clf%class_weights=[1.e300_dp,1.e300_dp,1.e300_dp]
  case('class_count');classification=.true.;deallocate(labels);allocate(labels(5));labels=1
  case('multiclass_monotonic');classification=.true.;clf%monotonic_cst=[0,0]
  case('warm_decrease');call reg%fit(x,y);reg%warm_start=.true.;reg%max_iter=2
  case('warm_width');call reg%fit(x,y);reg%warm_start=.true.;deallocate(x);allocate(x(6,3));x=1
  case('warm_loss');call reg%fit(x,y);reg%warm_start=.true.;reg%loss='absolute_error'
  case('warm_class_width');classification=.true.;call clf%fit(x,labels);clf%warm_start=.true.;labels=7
  case('warm_class_labels');classification=.true.;call clf%fit(x,labels);clf%warm_start=.true.;labels=labels+1
  case default;call require(.false.,'unknown histogram invalid-input case')
  end select
  if(classification) then
    call clf%fit(x,labels,w);proba=clf%predict_proba(q)
  else
    call reg%fit(x,y,w);prediction=reg%predict(q)
  end if
end program
