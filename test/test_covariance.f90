program test_covariance
  use sklearn_common
  use sklearn_covariance
  implicit none
  type(empirical_covariance) :: ec
  type(shrunk_covariance) :: sc
  type(ledoit_wolf) :: lw
  type(oas) :: oa
  real(dp) :: x(20,3),identity(3,3)
  integer :: i
  call seed_rng(12); x=normal_random(20,3); x(:,2)=x(:,2)*2
  identity=0
  do i=1,3
    identity(i,i)=1
  end do
  call ec%fit(x)
  call require(maxval(abs(matmul(ec%covariance_,ec%precision_)-identity))<1.e-10_dp,'covariance inverse')
  call require(abs(sum(ec%mahalanobis(x))/20-3)<1.e-10_dp,'mean squared Mahalanobis')
  call sc%fit(x); call require(sc%error_norm(ec%covariance_)>0,'shrinkage changes covariance')
  call lw%fit(x); call require(lw%shrinkage_>=0.and.lw%shrinkage_<=1,'Ledoit-Wolf shrinkage bounds')
  call oa%fit(x); call require(oa%shrinkage_>=0.and.oa%shrinkage_<=1,'OAS shrinkage bounds')
  call require(abs(ec%error_norm(ec%covariance_))<1.e-12_dp,'covariance self error')
  print *, 'covariance estimators: PASS'
end program
