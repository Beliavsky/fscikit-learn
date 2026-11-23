program test_nmf_mu
  use sklearn_decomposition, only: nmf
  use sklearn_common, only: dp,require
  implicit none
  type(nmf) :: model,copy
  real(dp) :: x(4,3),w0(4,1),h0(1,3)
  real(dp) :: small_x(1,1),small_w(1,1),small_h(1,1)
  real(dp), allocatable :: w(:,:),z(:,:),before(:,:)
  integer :: i,steps
  do i=1,4
    x(i,:)=real(i,dp)*[1._dp,2._dp,3._dp];w0(i,1)=i
  end do
  h0(1,:)=[1._dp,2._dp,3._dp]
  model%init='custom';model%n_components=1;model%max_iter=25
  w=model%fit_transform(x,w0,h0)
  call require(model%n_iter_==25.and..not.model%converged_,'zero initial MU error follows IEEE stopping')
  call require(model%mu_initial_error_==0.and.model%reconstruction_err_==0,'exact custom MU factorization')
  before=model%components_;steps=model%n_iter_;z=model%transform(x)
  call require(maxval(abs(model%inverse_transform(z)-x))<1.e-12_dp,'fixed-H MU transform')
  call require(all(before==model%components_).and.steps==model%n_iter_,'MU transform preserves fit state')
  model=nmf();model%init='random';model%random_state=42;model%max_iter=13;model%tol=0
  model%beta_loss=1;model%alpha_w=.01_dp;model%alpha_h=.02_dp;model%l1_ratio=.5_dp
  copy=model;copy%shuffle=.true.;w=model%fit_transform(x);z=copy%fit_transform(x)
  call require(all(w==z).and.all(model%components_==copy%components_),'MU ignores shuffle as sklearn does')
  call require(model%n_iter_==13.and.all(w>=0).and.all(model%components_>=0),'finite-iteration KL MU')
  call require(model%cd_violation_==0.and.model%cd_initial_violation_==0,'MU clears CD diagnostics')
  model=nmf();model%init='random';model%max_iter=11;x=0
  w=model%fit_transform(x)
  call require(model%n_iter_==11.and.model%reconstruction_err_==0,'zero matrix MU fit')
  model=nmf();model%n_components=1;model%init='custom';model%beta_loss=2.5_dp;model%max_iter=1;model%tol=0
  small_x=1.e-215_dp;small_w=2.e-216_dp;small_h=1
  w=model%fit_transform(small_x,small_w,small_h)
  call require(w(1,1)>0,'MU preserves a coordinate supported by a rounded subnormal denominator')
  call require(abs(w(1,1)/4.1601676461038081e-216_dp-1)<3.e-12_dp,'MU subnormal denominator matches sklearn')
  print *, 'NMF multiplicative updates: PASS'
end program
