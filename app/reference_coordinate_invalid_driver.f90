program reference_coordinate_invalid_driver
  use sklearn_coordinate_paths,only:enet_path,coordinate_path_options,coordinate_path_result
  use sklearn_common,only:dp,require
  use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(coordinate_path_options)::options
  type(coordinate_path_result)::fit
  real(dp)::x(2,2),y(2),gram(2,2),nan,inf
  character(64)::name
  x=reshape([1._dp,0._dp,0._dp,1._dp],[2,2]);y=[1._dp,2._dp];gram=x
  nan=ieee_value(0._dp,ieee_quiet_nan);inf=ieee_value(0._dp,ieee_positive_inf)
  call get_command_argument(1,name)
  select case(trim(name))
  case('empty_x');fit=enet_path(x(:0,:),y(:0))
  case('target_length');fit=enet_path(x,y(:1))
  case('empty_targets');fit=enet_path(x,reshape([real(dp)::],[2,0]))
  case('x_nan');x(1,1)=nan;fit=enet_path(x,y)
  case('y_inf');y(1)=inf;fit=enet_path(x,y)
  case('mix_negative');options%l1_ratio=-1;fit=enet_path(x,y,options)
  case('mix_many');options%l1_ratio=2;fit=enet_path(x,y,options)
  case('mix_nan');options%l1_ratio=nan;fit=enet_path(x,y,options)
  case('mix_inf');options%l1_ratio=inf;fit=enet_path(x,y,options)
  case('mix_zero_auto');options%l1_ratio=0;fit=enet_path(x,y,options)
  case('eps_zero');options%eps=0;fit=enet_path(x,y,options)
  case('eps_negative');options%eps=-1;fit=enet_path(x,y,options)
  case('eps_nan');options%eps=nan;fit=enet_path(x,y,options)
  case('eps_inf');options%eps=inf;fit=enet_path(x,y,options)
  case('grid_zero');options%n_alphas=0;fit=enet_path(x,y,options)
  case('grid_negative');options%n_alphas=-1;fit=enet_path(x,y,options)
  case('tol_negative');options%tol=-1;fit=enet_path(x,y,options)
  case('tol_nan');options%tol=nan;fit=enet_path(x,y,options)
  case('tol_inf');options%tol=inf;fit=enet_path(x,y,options)
  case('iteration_negative');options%max_iter=-1;fit=enet_path(x,y,options)
  case('seed_negative');options%random_state=-1;fit=enet_path(x,y,options)
  case('selection');options%selection='invalid';fit=enet_path(x,y,options)
  case('precompute');options%precompute='invalid';fit=enet_path(x,y,options)
  case('positive_column');options%positive=.true.;fit=enet_path(x,reshape(y,[2,1]),options)
  case('positive_multi');options%positive=.true.;fit=enet_path(x,spread(y,2,2),options)
  case('alpha_negative');fit=enet_path(x,y,alphas=[-1._dp])
  case('alpha_nan');fit=enet_path(x,y,alphas=[nan])
  case('alpha_inf');fit=enet_path(x,y,alphas=[inf])
  case('initial_shape');fit=enet_path(x,y,coef_init=[1._dp])
  case('initial_matrix');fit=enet_path(x,spread(y,2,2),coef_init=reshape(y,[2,1]))
  case('initial_nan');fit=enet_path(x,y,coef_init=[nan,0._dp])
  case('gram_shape');fit=enet_path(x,y,gram=x(:,:1))
  case('gram_nan');gram(1,1)=nan;fit=enet_path(x,y,gram=gram)
  case('gram_sampled');gram(2,2)=2;fit=enet_path(x,y,gram=gram)
  case('xy_empty');fit=enet_path(x,y,xy=[real(dp)::])
  case('xy_shape');options%precompute='true';fit=enet_path(x,y,options,xy=[1._dp])
  case('xy_matrix_shape');options%precompute='true';fit=enet_path(x,spread(y,2,2),options,xy=reshape(y,[2,1]))
  case('xy_nan');fit=enet_path(x,y,xy=[nan,1._dp])
  case('gram_denominator')
    gram(1,1)=-1;options%l1_ratio=1
    fit=enet_path(x,y,options,alphas=[.1_dp],gram=gram)
  case default;call require(.false.,'unknown coordinate path invalid fixture')
  end select
  print *,fit%alphas
  error stop 'coordinate invalid input unexpectedly accepted'
end program
