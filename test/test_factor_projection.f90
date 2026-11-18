program test_factor_projection
  use sklearn
  use sklearn_common, only: require,normal_random,seed_rng
  implicit none
  type(factor_analysis) :: fa
  type(gaussian_random_projection) :: gp,repeat
  type(sparse_random_projection) :: sp
  real(dp) :: x(200,12),factors(200,2),large(5,1000),eye(12,12),coin_mean,variance
  real(dp), allocatable :: z(:,:),cov(:,:),precision(:,:),inverse(:,:)
  integer :: j
  call seed_rng(122); factors=normal_random(200,2); x=normal_random(200,12)*0.15_dp
  do j=1,12
    x(:,j)=x(:,j)+factors(:,1)*cos(real(j,dp))+factors(:,2)*sin(real(j,dp))
  end do
  fa%n_components=2; fa%tol=1.e-6_dp; z=fa%fit_transform(x)
  call require(fa%converged_.and.all(shape(z)==[200,2]),'FactorAnalysis convergence and shape')
  call require(all(fa%loglike_(2:)>=fa%loglike_(:fa%n_iter_-1)-1.e-9_dp),'factor likelihood improves')
  cov=fa%get_covariance(); precision=fa%get_precision(); eye=0
  do j=1,12
    eye(j,j)=1
  end do
  call require(maxval(abs(matmul(cov,precision)-eye))<1.e-10_dp,'factor covariance inverse')
  call require(abs(fa%score(x)-sum(fa%score_samples(x))/200)<1.e-10_dp,'factor likelihood aggregation')
  fa%n_components=0; z=fa%fit_transform(x)
  call require(size(z,2)==0.and.size(fa%components_,1)==0,'zero latent dimension')
  call require(maxval(abs(fa%noise_variance_-sum((x-spread(fa%mean_,1,200))**2,dim=1)/200)) &
    <1.e-10_dp,'zero-factor diagonal covariance')
  gp%n_components=12; gp%random_state=44; z=gp%fit_transform(x)
  inverse=gp%inverse_transform(z)
  call require(maxval(abs(inverse-x))<1.e-9_dp,'square Gaussian projection recovers input')
  gp%n_components=6; gp%compute_inverse_components=.true.; z=gp%fit_transform(x)
  inverse=gp%inverse_transform(z)
  call require(maxval(abs(gp%transform(inverse)-z))<1.e-10_dp,'projection inverse reconstructs coordinates')
  repeat%n_components=6; repeat%random_state=44; call repeat%fit(x)
  call require(all(gp%components_==repeat%components_),'Gaussian seed reproducibility')
  gp%compute_inverse_components=.false.; call gp%fit(x)
  call require(.not.allocated(gp%inverse_components_),'projection refit clears inverse cache')
  sp%n_components=6; sp%random_state=22; z=sp%fit_transform(x)
  call require(abs(sp%density_-1/sqrt(12._dp))<1.e-15_dp,'automatic sparse density')
  call require(all(abs(abs(sp%components_)-sqrt(1/(sp%density_*6)))<1.e-15_dp.or.sp%components_==0), &
    'sparse projection three-point distribution')
  sp%density=1.e-100_dp; z=sp%fit_transform(x)
  call require(all(z==0).and.all(sp%inverse_transform(z)==0),'zero-rank sparse pseudoinverse')
  ! Larger draws test scale, density, and distance preservation independently
  ! of the algebra reference checks, which inject the native random matrix.
  large=0; gp%n_components=200; call gp%fit(large)
  coin_mean=sum(gp%components_)/size(gp%components_)
  variance=sum(gp%components_**2)/size(gp%components_)
  call require(abs(coin_mean)<0.001_dp.and.abs(variance-0.005_dp)<0.0001_dp,'Gaussian projection moments')
  large(1,:)=1; z=gp%transform(large)
  call require(abs(sum(z(1,:)**2)/1000-1)<0.25_dp,'Gaussian distance preservation')
  sp%n_components=200; sp%density=0.1_dp; call sp%fit(large)
  call require(abs(real(count(sp%components_/=0),dp)/size(sp%components_)-0.1_dp)<0.005_dp,'sparse density')
  call require(johnson_lindenstrauss_min_dim(100,0.5_dp)==221,'JL dimension bound')
  print *, 'factor analysis and random projections: PASS'
end program
