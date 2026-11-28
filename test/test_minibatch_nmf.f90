program test_minibatch_nmf
  use sklearn_minibatch_nmf, only: mini_batch_nmf
  use sklearn_common, only: dp,require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan,ieee_is_finite
  use,intrinsic::ieee_exceptions,only:ieee_get_flag,ieee_invalid
  implicit none
  type(mini_batch_nmf) :: model,stream,copy,zero_model,large_model
  real(dp) :: x(2,2),w0(2,1),h0(1,2),before(2,2)
  real(dp), allocatable :: w(:,:),reconstructed(:,:)
  real(dp)::large_x(6,2),large_w(6,1),large_h(1,2)
  logical::invalid_before,invalid_after
  x=1; w0=1; h0=1; before=x
  model%n_components=1; model%init='custom'; model%max_iter=1; model%tol=0
  w=model%fit_transform(x,w0,h0)
  call require(all(w==1).and.all(model%components_==1),'MiniBatchNMF exact rank-one fixed point')
  call require(maxval(abs(model%components_numerator_-2.7_dp))<1.e-14_dp,'MiniBatchNMF analytic numerator')
  call require(maxval(abs(model%components_denominator_-2.7_dp))<1.e-14_dp,'MiniBatchNMF analytic denominator')
  call require(model%n_steps_==1.and.model%n_iter_==1,'MiniBatchNMF fit counters')
  call require(model%reconstruction_err_==0.and.all(x==before).and.all(w0==1).and.all(h0==1), &
    'MiniBatchNMF reconstruction and input preservation')
  w=model%transform(x); reconstructed=model%inverse_transform(w)
  call require(all(w==1).and.all(reconstructed==1),'MiniBatchNMF transform/inverse fixed point')
  stream%n_components=1; stream%init='custom'
  call stream%partial_fit(x,w0,h0); copy=stream
  call stream%partial_fit(2*x); call copy%partial_fit(2*x)
  call require(maxval(abs(stream%components_numerator_-9.89_dp))<1.e-12_dp,'MiniBatchNMF streaming numerator recurrence')
  call require(maxval(abs(stream%components_denominator_-9.89_dp))<1.e-12_dp,'MiniBatchNMF streaming denominator recurrence')
  call require(all(stream%components_==copy%components_),'MiniBatchNMF copy continuation')
  call require(ieee_is_nan(stream%reconstruction_err_).and.stream%n_iter_==0.and.stream%n_steps_==2, &
    'MiniBatchNMF partial-only metadata')
  model%max_iter=1000000000; model%tol=1.e9_dp; w=model%fit_transform(x,w0,h0)
  call require(model%n_steps_==2.and.model%n_iter_==2,'MiniBatchNMF huge budget motion stopping')
  model%max_iter=1; model%tol=0; model%beta_loss=1
  w=model%fit_transform(x,w0,h0)
  call require(all(w==1).and.all(model%components_==1).and.model%reconstruction_err_==0, &
    'MiniBatchNMF KL exact fixed point')
  model%beta_loss=0; w=model%fit_transform(x,w0,h0)
  call require(maxval(abs(w-1))<1.e-14_dp.and.maxval(abs(model%components_-1))<1.e-14_dp, &
    'MiniBatchNMF Itakura-Saito exact fixed point')
  zero_model%n_components=2; zero_model%init='random'; zero_model%max_iter=20
  w=zero_model%fit_transform(0*x)
  call require(all(w==0).and.all(zero_model%components_==0).and.zero_model%reconstruction_err_==0, &
    'MiniBatchNMF zero model without floating-point traps')
  call require(zero_model%n_steps_==12,'MiniBatchNMF zero-norm motion does not falsely stop')
  zero_model%init='nndsvda'; w=zero_model%fit_transform(0*x)
  call require(all(w==0).and.all(zero_model%components_==0),'MiniBatchNMF null NNDSVD directions handled explicitly')
  ! Finite factors can produce a NaN monitoring cost through 0*overflow.
  ! Source keeps that NaN distinct from its initial None state.
  large_x=1;large_w=1;large_h=1.e155_dp
  large_model%n_components=1;large_model%init='custom';large_model%max_iter=12;large_model%batch_size=3
  large_model%tol=0;large_model%max_no_improvement=2;large_model%fresh_restarts=.true.
  large_model%fresh_restarts_max_iter=1;large_model%transform_max_iter=1
  call ieee_get_flag(ieee_invalid,invalid_before)
  w=large_model%fit_transform(large_x,large_w,large_h)
  call ieee_get_flag(ieee_invalid,invalid_after)
  call require(large_model%n_steps_==4.and.large_model%no_improvement_==2,'NaN cost retains Source patience state')
  call require(ieee_is_nan(large_model%ewa_cost_).and.ieee_is_nan(large_model%ewa_cost_min_), &
    'NaN cost is preserved rather than reinitialized')
  call require(all(ieee_is_finite(large_model%components_)).and.all(w==0),'NaN monitoring cost permits finite model')
  call require(invalid_before.eqv.invalid_after,'dense MiniBatch restores IEEE invalid flag')
  large_model%batch_size=2
  w=large_model%fit_transform(x,w0,h0)
  call require(large_model%n_steps_==4.and.large_model%ewa_cost_==0.and.large_model%ewa_cost_min_==0, &
    'MiniBatch refit resets initialized monitoring state')
  print *, 'MiniBatchNMF tests passed'
end program
