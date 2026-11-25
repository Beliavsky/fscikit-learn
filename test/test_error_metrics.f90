program test_error_metrics
 use sklearn,only:dp,mean_squared_error,root_mean_squared_error,mean_absolute_error, &
  mean_absolute_percentage_error,mean_squared_error_raw,root_mean_squared_error_raw, &
  mean_absolute_error_raw,mean_absolute_percentage_error_raw
 use sklearn_common,only:require
 use,intrinsic::ieee_arithmetic
 implicit none
 real(dp)::y(3),p(3),a(3,2),b(3,2),v,nan
 real(dp),allocatable::raw(:)
 integer::n
 integer,allocatable::before(:),after(:)
 call random_seed(size=n);allocate(before(n),after(n));call random_seed(get=before)
 y=[1._dp,2._dp,3._dp];p=[1._dp,3._dp,3._dp]
 call require(abs(mean_squared_error(y,p)-1._dp/3)<1.e-14_dp,'MSE legacy call')
 call require(mean_squared_error(y,p,sample_weight=[0._dp,2._dp,0._dp])==1,'weighted MSE')
 call require(abs(mean_absolute_error(y,p)-1._dp/3)<1.e-14_dp,'MAE legacy call')
 call require(abs(mean_absolute_percentage_error(y,p)-1._dp/6)<1.e-14_dp,'MAPE legacy call')
 call require(mean_absolute_percentage_error([0._dp],[1._dp])==1/epsilon(1._dp),'MAPE epsilon floor')
 a(:,1)=y;a(:,2)=y;b(:,1)=p;b(:,2)=y
 raw=mean_squared_error_raw(a,b)
 call require(size(raw)==2.and.abs(raw(1)-1._dp/3)<1.e-14_dp.and.raw(2)==0,'owned raw MSE')
 v=root_mean_squared_error(a,b,raw_values=raw)
 call require(abs(v-sqrt(1._dp/3)/2)<1.e-14_dp,'RMSE averages per-output roots')
 call require(abs(v-sqrt(mean_squared_error(a,b)))>0.1_dp,'RMSE differs from root of aggregate MSE')
 call require(root_mean_squared_error(y,p,multioutput=[1._dp])==sqrt(mean_squared_error(y,p)), &
  'RMSE accepts singleton output weights')
 nan=ieee_value(0._dp,ieee_quiet_nan)
 call require(ieee_is_nan(root_mean_squared_error(a,b,multioutput=[nan,1._dp])), &
  'RMSE permits nonfinite output weights')
 v=root_mean_squared_error([0._dp,1._dp],[2._dp,1._dp],sample_weight=[-1._dp,2._dp])
 call require(ieee_is_nan(v),'negative weighted MSE gives quiet RMSE NaN')
 call require(mean_squared_error([huge(1._dp)],[-huge(1._dp)])>huge(1._dp),'quiet overflow')
 call require(mean_absolute_error(y,reshape(p,[3,1]))==mean_absolute_error(reshape(y,[3,1]),p),'mixed ranks')
 raw=mean_absolute_error_raw(a,b);call require(raw(2)==0,'raw MAE')
 raw=mean_absolute_percentage_error_raw(a,b);call require(raw(2)==0,'raw MAPE')
 raw=root_mean_squared_error_raw(a,b);call require(raw(2)==0,'raw RMSE')
 v=mean_squared_error(a,b,multioutput=[1._dp,3._dp],raw_values=raw)
 call require(abs(v-1._dp/12)<1.e-14_dp.and.size(raw)==2,'custom output weights and raw result')
 call random_seed(get=after);call require(all(before==after),'error metrics preserve RNG')
 print *, 'Weighted multioutput MSE/RMSE/MAE/MAPE: PASS'
end program
