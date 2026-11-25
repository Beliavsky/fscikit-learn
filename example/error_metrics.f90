program error_metrics
 use sklearn,only:dp,mean_squared_error,root_mean_squared_error,mean_absolute_error, &
  mean_absolute_percentage_error,root_mean_squared_error_raw
 implicit none
 real(dp)::y(3,2),p(3,2)
 y(:,1)=[1._dp,2._dp,3._dp];y(:,2)=y(:,1)
 p=y;p(2,1)=3
 print *, 'MSE:',mean_squared_error(y,p)
 print *, 'RMSE:',root_mean_squared_error(y,p)
 print *, 'MAE:',mean_absolute_error(y,p)
 print *, 'MAPE:',mean_absolute_percentage_error(y,p)
 print *, 'Per-output RMSE:',root_mean_squared_error_raw(y,p)
end program
