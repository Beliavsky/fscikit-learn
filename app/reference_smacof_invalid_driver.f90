program reference_smacof_invalid_driver
 use sklearn_common,only:dp
 use sklearn_smacof,only:smacof,smacof_result
 use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan
 implicit none
 type(smacof_result)::fit
 real(dp)::d(3,3),initial(3,2),eps
 integer::which,k,runs,limit,seed
 logical::metric
 character(16)::argument,normalized
 call get_command_argument(1,argument);read(argument,*)which
 d=reshape([0._dp,1._dp,3._dp,1._dp,0._dp,2._dp,3._dp,2._dp,0._dp],[3,3])
 initial=1;k=2;runs=1;limit=3;seed=0;eps=.001_dp;metric=.true.;normalized='auto'
 select case(which)
 case(1)
  k=0
 case(2)
  runs=0
 case(3)
  limit=0
 case(4)
  eps=-1
 case(5)
  eps=ieee_value(1._dp,ieee_quiet_nan)
 case(6)
  seed=-1
 case(7)
  normalized='unknown'
 case(8)
  normalized='true'
 case(9)
  d(1,2)=4
 case(10)
  fit=smacof(d(:,:2));stop 0
 case(11)
  fit=smacof(d(:0,:0));stop 0
 case(12)
  d(1,1)=ieee_value(1._dp,ieee_quiet_nan)
 case(13)
  fit=smacof(d,init=initial(:2,:));stop 0
 case(14)
  initial(1,1)=ieee_value(1._dp,ieee_quiet_nan);fit=smacof(d,init=initial);stop 0
 case(15)
  metric=.false.;d=0
 case(16)
  metric=.false.;initial=0
  fit=smacof(d,metric=metric,init=initial,max_iter=3);stop 0
 end select
 fit=smacof(d,n_components=k,n_init=runs,max_iter=limit,eps=eps,random_state=seed, &
  metric=metric,normalized_stress=trim(normalized))
end program
