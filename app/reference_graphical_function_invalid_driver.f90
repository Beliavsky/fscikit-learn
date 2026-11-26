program reference_graphical_function_invalid_driver
 use sklearn_common,only:dp
 use sklearn_covariance,only:graphical_lasso,graphical_lasso_result
 use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan
 implicit none
 type(graphical_lasso_result)::fit
 real(dp)::c(2,2),alpha,tol,enet,eps
 integer::which,limit
 character(16)::argument,mode
 call get_command_argument(1,argument);read(argument,*)which
 c=reshape([2._dp,.8_dp,.8_dp,1._dp],[2,2]);alpha=.3_dp;tol=.0001_dp;enet=tol;eps=epsilon(1._dp);limit=100;mode='cd'
 select case(which)
 case(1)
  alpha=-1
 case(2)
  alpha=ieee_value(1._dp,ieee_quiet_nan)
 case(3)
  mode='unknown'
 case(4)
  tol=0
 case(5)
  tol=ieee_value(1._dp,ieee_quiet_nan)
 case(6)
  enet=-1
 case(7)
  enet=ieee_value(1._dp,ieee_quiet_nan)
 case(8)
  eps=-1
 case(9)
  eps=ieee_value(1._dp,ieee_quiet_nan)
 case(10)
  limit=-1
 case(11)
  fit=graphical_lasso(c(:,:1),alpha);stop 0
 case(12)
  fit=graphical_lasso(c(:1,:1),alpha);stop 0
 case(13)
  fit=graphical_lasso(c(:0,:0),alpha);stop 0
 case(14)
  c(1,1)=ieee_value(1._dp,ieee_quiet_nan)
 case(15)
  c=0;alpha=0
 end select
 fit=graphical_lasso(c,alpha,mode=trim(mode),tol=tol,enet_tol=enet,max_iter=limit,eps=eps)
end program
