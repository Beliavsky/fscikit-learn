program test_distribution_errors
 use sklearn,only:dp,mean_squared_log_error,root_mean_squared_log_error, &
  mean_squared_log_error_raw,root_mean_squared_log_error_raw,mean_poisson_deviance,mean_gamma_deviance
 use sklearn_regression_metrics,only:legacy_rmsle=>root_mean_squared_log_error
 use sklearn_common,only:require
 use,intrinsic::ieee_arithmetic
 implicit none
 real(dp)::y(3),p(3),a(3,2),b(3,2),v
 real(dp),allocatable::raw(:),empty_weights(:)
 integer::n
 integer,allocatable::before(:),after(:)
 call random_seed(size=n);allocate(before(n),after(n));call random_seed(get=before)
 y=[0._dp,1._dp,3._dp];p=[1._dp,1._dp,3._dp]
 call require(abs(mean_squared_log_error(y,p)-log(2._dp)**2/3)<1.e-14_dp,'ordinary MSLE')
 call require(abs(mean_squared_log_error([-.5_dp],[0._dp])-log(2._dp)**2)<1.e-14_dp,'valid negative log target')
 call require(mean_squared_log_error([0._dp],[1.e-18_dp])>0,'log1p preserves tiny nonzero loss')
 call require(mean_squared_log_error(y,p,sample_weight=[0._dp,2._dp,1._dp])==0,'weighted log error')
 a(:,1)=y;a(:,2)=y;b(:,1)=p;b(:,2)=y
 raw=mean_squared_log_error_raw(a,b)
 call require(size(raw)==2.and.raw(2)==0,'owned raw MSLE')
 v=root_mean_squared_log_error(a,b,raw_values=raw)
 call require(abs(v-log(2._dp)/sqrt(3._dp)/2)<1.e-14_dp,'RMSLE averages roots per output')
 call require(abs(v-sqrt(mean_squared_log_error(a,b)))>.05_dp,'RMSLE differs from root of aggregate')
 v=legacy_rmsle(y=a,predicted=b,output_weights=[1._dp,3._dp],raw_values=raw)
 call require(abs(v-log(2._dp)/sqrt(3._dp)/4)<1.e-14_dp.and.raw(2)==0,'legacy RMSLE keyword adapter')
 call require(v==legacy_rmsle(a,b,output_weights=[1._dp,3._dp]),'legacy module export')
 v=legacy_rmsle(a,b,output_weights=empty_weights)
 call require(abs(v-log(2._dp)/sqrt(3._dp)/2)<1.e-14_dp,'unallocated named legacy weights mean absent')
 v=legacy_rmsle(a,b,output_weights=empty_weights,raw_values=raw)
 call require(size(raw)==2,'legacy optional output result allocated')
 v=legacy_rmsle(a,b,empty_weights,empty_weights,raw)
 call require(v==legacy_rmsle(a,b),'legacy named absent semantics')
 raw=root_mean_squared_log_error_raw(a,b);call require(raw(2)==0,'owned raw RMSLE')
 v=root_mean_squared_log_error([0._dp,1._dp],[1._dp,1._dp],sample_weight=[-1._dp,2._dp])
 call require(ieee_is_nan(v),'signed mean log square gives quiet negative-root NaN')
 call require(abs(mean_poisson_deviance([0._dp],[2._dp])-4)<1.e-14_dp,'Poisson xlogy zero')
 call require(mean_poisson_deviance([1._dp,2._dp],[1._dp,2._dp])==0,'perfect Poisson')
 call require(mean_gamma_deviance([1._dp,2._dp],[1._dp,2._dp])==0,'perfect Gamma')
 call require(abs(mean_gamma_deviance([2._dp],[1._dp])-2*(1-log(2._dp)))<1.e-14_dp,'Gamma formula')
 call require(mean_poisson_deviance(reshape([0._dp,1._dp],[2,1]),[2._dp,1._dp], &
  sample_weight=[1._dp,0._dp])==4,'weighted column Poisson')
 call require(mean_gamma_deviance([1._dp,2._dp],reshape([1._dp,2._dp],[2,1]))==0,'mixed rank Gamma')
 call random_seed(get=after);call require(all(before==after),'distribution metrics preserve RNG')
 print *, 'Weighted log errors and Poisson/Gamma deviances: PASS'
end program
