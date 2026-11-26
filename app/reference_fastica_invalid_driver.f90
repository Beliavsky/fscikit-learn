program reference_fastica_invalid_driver
 use sklearn_common,only:dp
 use sklearn_fast_ica,only:fastica,fastica_result
 use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan
 implicit none
 type(fastica_result)::fit
 real(dp)::x(5,2),w(2,2),alpha,tol
 integer::which,i,k,seed,limit
 character(24)::argument,algorithm,whiten,fun,solver
 call get_command_argument(1,argument);read(argument,*)which
 do i=1,5
  x(i,:)=[real(i,dp),sin(real(i,dp))]
 end do
 w=0;alpha=1;tol=.0001_dp;k=2;seed=0;limit=200
 algorithm='parallel';whiten='unit-variance';fun='logcosh';solver='svd'
 select case(which)
 case(1)
  k=-1
 case(2)
  limit=0
 case(3)
  tol=-1
 case(4)
  tol=ieee_value(1._dp,ieee_quiet_nan)
 case(5)
  alpha=.5_dp
 case(6)
  alpha=ieee_value(1._dp,ieee_quiet_nan)
 case(7)
  algorithm='unknown'
 case(8)
  whiten='unknown'
 case(9)
  fun='unknown'
 case(10)
  solver='unknown'
 case(11)
  seed=-1
 case(12)
  x(1,1)=ieee_value(1._dp,ieee_quiet_nan)
 case(13)
  fit=fastica(x(:1,:));stop 0
 case(14)
  fit=fastica(x,w_init=w(:1,:1));stop 0
 case(15)
  w(1,1)=ieee_value(1._dp,ieee_quiet_nan);fit=fastica(x,w_init=w);stop 0
 case(16)
  fit=fastica(x,w_init=w);stop 0
 end select
 fit=fastica(x,n_components=k,max_iter=limit,tol=tol,alpha=alpha,algorithm=trim(algorithm), &
  whiten=trim(whiten),fun=trim(fun),whiten_solver=trim(solver),random_state=seed)
end program
