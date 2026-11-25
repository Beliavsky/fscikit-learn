program reference_dictionary_functions_invalid_driver
  use sklearn_common,only:dp
  use sklearn_dictionary,only:dict_learning,dict_learning_online,dictionary_learning_result,online_dictionary_result
  use,intrinsic::ieee_arithmetic
  implicit none
  type(dictionary_learning_result)::a
  type(online_dictionary_result)::b
  real(dp)::x(3,2),d(2,2),u(3,2)
  character(16)::arg
  integer::mode
  call get_command_argument(1,arg);read(arg,*)mode
  x=1;d=1;u=1
  select case(mode)
  case(1)
    a=dict_learning(x(:0,:),2,1._dp)
  case(2)
    a=dict_learning(x(:,:0),2,1._dp)
  case(3)
    a=dict_learning(x,-1,1._dp)
  case(4)
    a=dict_learning(x,2,-1._dp)
  case(5)
    a=dict_learning(x,2,ieee_value(0._dp,ieee_quiet_nan))
  case(6)
    a=dict_learning(x,2,1._dp,tol=-1._dp)
  case(7)
    a=dict_learning(x,2,1._dp,max_iter=-1)
  case(8)
    a=dict_learning(x,2,1._dp,method_max_iter=-1)
  case(9)
    a=dict_learning(x,2,1._dp,method='lars_bad')
  case(10)
    a=dict_learning(x,2,1._dp,random_state=-1)
  case(11)
    a=dict_learning(x,2,1._dp,positive_code=.true.,method='lars')
  case(12)
    a=dict_learning(x,2,1._dp,dict_init=d,code_init=u(:2,:))
  case(13)
    d(1,1)=ieee_value(0._dp,ieee_positive_inf);a=dict_learning(x,2,1._dp,dict_init=d,code_init=u)
  case(14)
    b=dict_learning_online(x,batch_size=0)
  case(15)
    b=dict_learning_online(x,n_components=-1)
  case(16)
    b=dict_learning_online(x,max_no_improvement=-2)
  case(17)
    b=dict_learning_online(x,dict_init=d(:,:1))
  case(18)
    b=dict_learning_online(x,alpha=ieee_value(0._dp,ieee_quiet_nan))
  case(19)
    b=dict_learning_online(x,method='lars_bad')
  case(20)
    b=dict_learning_online(x,positive_code=.true.,method='lars')
  end select
  print *,'guard unexpectedly returned'
end program
