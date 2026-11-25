program distribution_errors
 use sklearn,only:dp,mean_squared_log_error,root_mean_squared_log_error, &
  mean_squared_log_error_raw,mean_poisson_deviance,mean_gamma_deviance
 implicit none
 real(dp)::y(3,2),p(3,2)
 y(:,1)=[0._dp,1._dp,3._dp];y(:,2)=y(:,1);p=y;p(1,1)=1
 print *, 'MSLE:',mean_squared_log_error(y,p)
 print *, 'RMSLE:',root_mean_squared_log_error(y,p)
 print *, 'Raw MSLE:',mean_squared_log_error_raw(y,p)
 print *, 'Poisson:',mean_poisson_deviance([2._dp,0._dp,1._dp,4._dp],[.5_dp,.5_dp,2._dp,2._dp])
 print *, 'Gamma:',mean_gamma_deviance([2._dp,.5_dp,1._dp,4._dp],[.5_dp,.5_dp,2._dp,2._dp])
end program
