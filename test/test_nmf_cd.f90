program test_nmf_cd
  use sklearn_decomposition, only: nmf
  use sklearn_common, only: dp,require
  implicit none
  type(nmf) :: model,copy
  real(dp) :: x(4,3),initial_w(4,1),initial_h(1,3)
  real(dp), allocatable :: w(:,:),z(:,:),before(:,:),custom_w(:,:),custom_h(:,:)
  integer :: i,iterations
  do i=1,4
    x(i,:)=real(i,dp)*[1._dp,2._dp,3._dp]
  end do
  model%solver='cd';model%n_components=1;model%init='custom';model%max_iter=100
  initial_w=1;initial_h(1,:)=[1._dp,2._dp,3._dp]
  w=model%fit_transform(x,initial_w,initial_h)
  call require(maxval(abs(matmul(w,model%components_)-x))<1.e-12_dp,'CD exact rank-one fit')
  call require(model%converged_.and.model%n_iter_<=3,'CD projected-gradient stopping')
  call require(model%n_components_==1.and.model%n_features_in_==3,'CD fitted metadata')
  call require(all(initial_w==1).and.all(initial_h(1,:)==[1._dp,2._dp,3._dp]),'custom initializers preserved')
  before=model%components_;iterations=model%n_iter_;z=model%transform(x)
  call require(maxval(abs(model%inverse_transform(z)-x))<1.e-12_dp,'CD fixed-component transform')
  call require(all(before==model%components_).and.iterations==model%n_iter_,'transform preserves fit state')
  model%alpha_w=.1_dp;model%alpha_h=.2_dp;model%l1_ratio=.5_dp
  w=model%fit_transform(x,initial_w,initial_h)
  call require(all(w>=0).and.all(model%components_>=0),'regularized CD nonnegativity')
  model=nmf();model%solver='cd';model%n_components=2;model%init='random';model%shuffle=.true.
  model%random_state=42;model%tol=0;model%max_iter=7
  w=model%fit_transform(x);copy=model;z=copy%fit_transform(x)
  call require(all(w==z).and.all(model%components_==copy%components_),'seeded shuffle reproducibility')
  call require(model%n_iter_==7.or.model%cd_initial_violation_==0,'zero-tolerance iteration count')
  model%init='custom';model%n_components=0
  allocate(custom_w(4,2),custom_h(2,3));custom_w=.3_dp;custom_h=.2_dp
  w=model%fit_transform(x,custom_w,custom_h)
  call require(model%n_components_==2,'custom auto component inference')
  model=nmf();model%solver='cd';model%init='random';model%n_components=1
  x=0;w=model%fit_transform(x)
  call require(model%n_iter_==1.and.model%cd_initial_violation_==0.and.model%reconstruction_err_==0, &
    'zero initial violation stops immediately')
  print *, 'NMF coordinate descent: PASS'
end program
