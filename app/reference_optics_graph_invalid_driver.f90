program reference_optics_graph_invalid_driver
  use sklearn_optics,only:compute_optics_graph,optics_graph_result
  use sklearn_common,only:dp
  use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan
  implicit none
  type(optics_graph_result)::graph
  real(dp)::x(3,2),eps,power,fraction
  integer::k,which
  character(20)::argument,metric
  call get_command_argument(1,argument);read(argument,*)which
  x=1;eps=1;power=2;k=2;fraction=.5_dp;metric='minkowski'
  select case(which)
  case(1)
    k=1
  case(2)
    k=4
  case(3)
    graph=compute_optics_graph(x,0._dp)
    stop 0
  case(4)
    graph=compute_optics_graph(x,1.1_dp)
    stop 0
  case(5)
    graph=compute_optics_graph(x,ieee_value(1._dp,ieee_quiet_nan))
    stop 0
  case(6)
    eps=-1
  case(7)
    eps=ieee_value(1._dp,ieee_quiet_nan)
  case(8)
    power=0
  case(9)
    power=ieee_value(1._dp,ieee_quiet_nan)
  case(10)
    metric='unknown'
  case(11)
    metric='precomputed'
  case(12)
    graph=compute_optics_graph(reshape([-1._dp,0._dp,0._dp,-1._dp],[2,2]),2,metric='precomputed')
    stop 0
  case(13)
    x=ieee_value(1._dp,ieee_quiet_nan)
  case(14)
    graph=compute_optics_graph(x(:0,:),2)
    stop 0
  case(15)
    graph=compute_optics_graph(x,2,distance=bad)
    stop 0
  case(16)
    graph=compute_optics_graph(x,2,distance=nonfinite)
    stop 0
  end select
  graph=compute_optics_graph(x,k,eps,metric,power)
contains
  function bad(a,b)result(value)
    real(dp),intent(in)::a(:),b(:)
    real(dp)::value
    value=-1
  end function
  function nonfinite(a,b)result(value)
    real(dp),intent(in)::a(:),b(:)
    real(dp)::value
    value=ieee_value(1._dp,ieee_quiet_nan)
  end function
end program
