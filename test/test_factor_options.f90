program test_factor_options
  use sklearn_common, only: dp,require
  use sklearn_factor_analysis, only: factor_analysis
  use sklearn_randomized_svd_core, only: randomized_svd
  use sklearn_legacy_rng, only: legacy_random_state
  implicit none
  type(factor_analysis) :: base,rotated,repeat
  type(legacy_random_state) :: rng1,rng2
  real(dp) :: x(70,16),eye(3,3),before,after
  real(dp), allocatable :: cov(:,:),precision(:,:),z(:,:),s1(:),s2(:),u1(:,:),u2(:,:),v1(:,:),v2(:,:)
  integer :: i,j,method
  do j=1,16
    do i=1,70
      x(i,j)=sin(.21_dp*i)*cos(.31_dp*j)+cos(.13_dp*i)*sin(.43_dp*j)+.2_dp*sin(real(i*j,dp))
    end do
  end do
  base%n_components=3;base%tol=1.e-7_dp;call base%fit(x);cov=base%get_covariance()
  eye=0;do j=1,3
    eye(j,j)=1
  end do
  do method=1,2
    rotated=base
    if(method==1) then
      rotated%rotation='varimax'
    else
      rotated%rotation='quartimax'
    end if
    call rotated%fit(x)
    call require(maxval(abs(rotated%get_covariance()-cov))<1.e-10_dp,'FA rotation preserves covariance')
    call require(all(rotated%noise_variance_==base%noise_variance_),'FA rotation preserves noise')
    call require(all(rotated%loglike_==base%loglike_),'FA rotation is after fitting')
    call require(maxval(abs(matmul(transpose(rotated%rotation_matrix_),rotated%rotation_matrix_)-eye))<1.e-10_dp, &
      'FA fitted rotation is orthogonal')
    before=sum(base%components_**4);after=sum(rotated%components_**4)
    if(method==1) then
      before=before-sum(sum(base%components_**2,dim=2)**2)/16
      after=after-sum(sum(rotated%components_**2,dim=2)**2)/16
    end if
    call require(after>=before-1.e-10_dp,'FA rotation improves its criterion')
  end do
  rotated%rotation='none';rotated%svd_method='randomized';rotated%iterated_power=0;rotated%random_state=42
  z=rotated%fit_transform(x);repeat=rotated;call repeat%fit(x)
  call require(all(repeat%components_==rotated%components_),'FA randomized fit seeded reproducibility')
  call require(rotated%n_rotation_iter_==0.and.rotated%rotation_converged_,'FA refit clears rotation diagnostics')
  rotated%n_components=0;rotated%rotation='varimax';z=rotated%fit_transform(x)
  call require(size(z,2)==0.and.size(rotated%rotation_matrix_)==0,'FA zero-factor rotation')
  rotated%n_components=3;rotated%svd_method='lapack';rotated%noise_variance_init=spread(100._dp,1,16)
  rotated%max_iter=1;call rotated%fit(x)
  call require(all(rotated%components_==0),'FA degenerate loading fixture')
  call require(maxval(abs(matmul(transpose(rotated%rotation_matrix_),rotated%rotation_matrix_)-eye))<1.e-12_dp, &
    'FA zero loading polar factor stays orthogonal')
  call rng1%seed(42);call rng2%seed(42)
  call randomized_svd(x,3,rng1,s1,u1,v1);call randomized_svd(x,3,rng2,s2,u2,v2,n_iter=4)
  call require(all(s1==s2).and.all(u1==u2).and.all(v1==v2),'randomized SVD legacy default preserved')
  print *, 'Factor analysis rotations and randomized SVD: PASS'
end program
