program covariance_functions
  use sklearn,only:dp,empirical_covariance,shrunk_covariance,ledoit_wolf,oas,covariance_estimate
  implicit none
  real(dp)::x(4,2)
  real(dp),allocatable::cov(:,:)
  type(covariance_estimate)::estimate
  x(:,1)=[1._dp,2._dp,3._dp,5._dp];x(:,2)=[2._dp,1._dp,4._dp,3._dp]
  cov=empirical_covariance(x)
  print *,'Empirical covariance:',cov
  print *,'Diagonal shrinkage:',shrunk_covariance(cov,.2_dp)
  estimate=ledoit_wolf(x)
  print *,'Ledoit-Wolf coefficient:',estimate%shrinkage
  estimate=oas(x)
  print *,'OAS covariance:',estimate%covariance
end program
