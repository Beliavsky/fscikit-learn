program test_pca_probability
  use sklearn_common, only: dp,require
  use sklearn_decomposition, only: pca
  implicit none
  type(pca) :: model
  real(dp) :: x(120,8),query(3,8),eye(8,8),tied(4,2)
  real(dp), allocatable :: z(:,:),back(:,:),cov(:,:),precision(:,:),scores(:),base(:,:)
  integer :: i,j
  do j=1,8
    do i=1,120
      x(i,j)=sin(.21_dp*i)*cos(.31_dp*j)+cos(.13_dp*i)*sin(.43_dp*j)+.2_dp*sin(real(i*j,dp))
    end do
  end do
  query=x(:3,:)+.01_dp;model%n_components=2;call model%fit(x)
  cov=model%get_covariance();precision=model%get_precision();eye=0
  do j=1,8
    eye(j,j)=1
  end do
  call require(maxval(abs(matmul(cov,precision)-eye))<1.e-9_dp,'PCA covariance precision inverse')
  scores=model%score_samples(query)
  call require(abs(model%score(query)-sum(scores)/3)<1.e-12_dp,'PCA Gaussian score aggregation')
  base=cov;model%whiten=.true.;cov=model%get_covariance()
  call require(maxval(abs(cov-base))>1.e-3_dp,'PCA sklearn whitened generative covariance scaling')
  model%whiten=.false.;model%dimension_mode='mle';call model%fit(x)
  call require(model%n_components_==6,'PCA Minka matches the pinned sklearn fixture rank')
  call require(allocated(model%mle_log_likelihood_),'PCA MLE likelihood diagnostics')
  model%dimension_mode='variance';model%variance_fraction=.8_dp;call model%fit(x)
  call require(sum(model%explained_variance_ratio_)>.8_dp,'PCA fraction uses strict greater-than selection')
  call require(.not.allocated(model%mle_log_likelihood_),'PCA refit clears MLE diagnostics')
  tied=reshape([-1._dp,1._dp,0._dp,0._dp,0._dp,0._dp,-1._dp,1._dp],[4,2])
  model%variance_fraction=.5_dp;call model%fit(tied)
  call require(model%n_components_==2,'PCA fraction exact-threshold right-sided selection')
  model%dimension_mode='zero';z=model%fit_transform(x);back=model%inverse_transform(z)
  call require(all(shape(z)==[120,0]),'PCA zero-dimensional transform')
  call require(maxval(abs(back-spread(model%mean_,1,120)))==0,'PCA zero-dimensional inverse is mean')
  cov=model%get_covariance();precision=model%get_precision()
  call require(maxval(abs(matmul(cov,precision)-eye))<1.e-12_dp,'PCA zero-dimensional isotropic probability model')
  model%dimension_mode='count';model%n_components=0;call model%fit(x)
  call require(model%n_components_==8,'PCA legacy zero count still selects all')
  model%dimension_mode='mle';call model%fit(x(:,:1))
  call require(model%n_components_==0,'PCA single-feature MLE zero-dimensional selection')
  print *, 'PCA dimensionality selection and probabilistic models: PASS'
end program
