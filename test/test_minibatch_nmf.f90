program test_minibatch_nmf
  use sklearn_minibatch_nmf, only: mini_batch_nmf
  use sklearn_common, only: dp,require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
  implicit none
  type(mini_batch_nmf) :: model,stream,copy,zero_model
  real(dp) :: x(2,2),w0(2,1),h0(1,2),before(2,2)
  real(dp), allocatable :: w(:,:),reconstructed(:,:)
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
  print *, 'MiniBatchNMF tests passed'
end program
