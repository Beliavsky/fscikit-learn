program reference_qda_invalid_driver
 use sklearn_common,only:dp
 use sklearn_discriminant_analysis,only:quadratic_discriminant_analysis
 use,intrinsic::ieee_arithmetic
 implicit none
 type(quadratic_discriminant_analysis)::model
 real(dp)::x(12,2),nan
 real(dp),allocatable::out(:,:)
 integer::i,which,y(12)
 character(24)::arg
 call get_command_argument(1,arg);read(arg,*)which
 do i=1,12
  x(i,:)=[sin(real(i,dp)),cos(.7_dp*i)];y(i)=merge(-3,7,i<=6)
 end do
 nan=ieee_value(0._dp,ieee_quiet_nan);model%reg_param=.2_dp
 select case(which)
 case(1);call model%fit(x(:0,:),y(:0));stop
 case(2);call model%fit(x(:,:0),y);stop
 case(3);x(1,1)=nan
 case(4);call model%fit(x,y(:11));stop
 case(5);y=1
 case(6);y=7;y(1)=-3
 case(7);model%reg_param=-.1_dp
 case(8);model%reg_param=1.1_dp
 case(9);model%reg_param=nan
 case(10);model%reg_param=ieee_value(0._dp,ieee_positive_inf)
 case(11);model%tol=-1
 case(12);model%tol=nan
 case(13);model%tol=ieee_value(0._dp,ieee_positive_inf)
 case(14);out=model%predict_log_proba(x);stop
 case(18);model%priors=[.2_dp,.3_dp,.5_dp]
 case(19);model%priors=[real(dp)::]
 end select
 call model%fit(x,y)
 select case(which)
 case(15);out=model%decision_function(x(:,:1))
 case(16);x(1,1)=nan;out=model%predict_proba(x)
 case(17);out=model%predict_log_proba(x(:0,:))
 case(18:19);out=model%predict_log_proba(x)
 end select
end program
