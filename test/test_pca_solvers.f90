program test_pca_solvers
  use sklearn_common, only: dp,require
  use sklearn_decomposition, only: pca
  use sklearn_randomized_svd_core, only: randomized_svd
  use sklearn_legacy_rng, only: legacy_random_state
  implicit none
  type(pca) :: full,cov,random,repeat
  type(legacy_random_state) :: rng1,rng2
  real(dp) :: x(80,24),large(520,60)
  real(dp), allocatable :: z(:,:),projected(:,:),s1(:),s2(:),u1(:,:),u2(:,:),v1(:,:),v2(:,:)
  integer :: i,j
  real(dp) :: initial_gap
  do j=1,24
    do i=1,80
      x(i,j)=sin(.21_dp*i)*cos(.31_dp*j)+cos(.13_dp*i)*sin(.43_dp*j)+.2_dp*sin(real(i*j,dp))
    end do
  end do
  full%n_components=3;call full%fit(x)
  cov=full;cov%svd_solver='covariance_eigh';call cov%fit(x)
  call require(maxval(abs(full%explained_variance_-cov%explained_variance_))<1.e-9_dp,'PCA covariance solver spectrum')
  call require(maxval(abs(full%get_covariance()-cov%get_covariance()))<1.e-9_dp,'PCA covariance solver model')
  random%n_components=3;random%svd_solver='randomized';random%iterated_power=0;random%n_oversamples=2
  random%random_state=42;z=random%fit_transform(x);projected=random%transform(x)
  call require(maxval(abs(z-projected))>1.e-5_dp,'PCA randomized fit_transform uses projected left singular coordinates')
  initial_gap=maxval(abs(random%explained_variance_-full%explained_variance_))
  repeat=random;projected=repeat%fit_transform(x)
  call require(all(z==projected),'PCA randomized fit_transform reproducibility')
  random%power_iteration_normalizer='QR';random%iterated_power=3;call random%fit(x)
  call require(maxval(abs(random%explained_variance_-full%explained_variance_))<initial_gap, &
    'PCA QR power steps improve approximation')
  cov%svd_solver='auto';call cov%fit(x)
  call require(cov%fit_svd_solver_=='full','PCA small automatic full solver')
  call cov%fit(x(:,:6));call require(cov%fit_svd_solver_=='covariance_eigh','PCA tall automatic covariance solver')
  do j=1,60
    do i=1,520
      large(i,j)=sin(.21_dp*i)*cos(.31_dp*j)+cos(.13_dp*i)*sin(.43_dp*j)+.2_dp*sin(real(i*j,dp))
    end do
  end do
  call cov%fit(large);call require(cov%fit_svd_solver_=='randomized','PCA automatic randomized branch')
  random%whiten=.true.;x=0;z=random%fit_transform(x)
  call require(maxval(abs(matmul(random%components_,transpose(random%components_))-diagonal3()))<1.e-12_dp, &
    'PCA randomized zero covariance retains an orthogonal right null basis')
  call require(maxval(abs(matmul(transpose(z),z)/79-diagonal3()))<1.e-12_dp, &
    'PCA randomized zero covariance fit_transform retains orthogonal left coordinates')
  call rng1%seed(42);call rng2%seed(42)
  call randomized_svd(large,3,rng1,s1,u1,v1)
  call randomized_svd(large,3,rng2,s2,u2,v2,n_iter=7,n_oversamples=10,power_iteration_normalizer='auto',flip_sign=.true.)
  call require(all(s1==s2).and.all(u1==u2).and.all(v1==v2),'randomized SVD previous auto-seven defaults preserved')
  print *, 'PCA full, randomized, covariance and automatic solvers: PASS'
contains
  function diagonal3() result(eye)
    real(dp) :: eye(3,3)
    integer :: j
    eye=0;do j=1,3
      eye(j,j)=1
    end do
  end function
end program
