program reference_sparse_coordinate_invalid_driver
  use sklearn_coordinate_paths,only:enet_path,coordinate_path_options,coordinate_path_result
  use sklearn_sparse,only:csr_matrix,csr_from_arrays
  use sklearn_common,only:dp,require
  use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(csr_matrix)::x,z
  type(coordinate_path_options)::options
  type(coordinate_path_result)::fit
  real(dp)::y(3),nan,inf
  character(64)::name
  x=csr_from_arrays(3,3,[1,2,3,4],[1,2,3],[1._dp,1._dp,1._dp]);y=[1._dp,2._dp,3._dp]
  nan=ieee_value(0._dp,ieee_quiet_nan);inf=ieee_value(0._dp,ieee_positive_inf)
  call get_command_argument(1,name)
  select case(trim(name))
  case('unallocated');fit=enet_path(z,y)
  case('pointers');x%indptr(4)=3;fit=enet_path(x,y)
  case('indices');x%indices(1)=0;fit=enet_path(x,y)
  case('data_count');x%data=[1._dp];fit=enet_path(x,y)
  case('empty_rows');z=csr_from_arrays(0,3,[1],[integer::],[real(dp)::]);fit=enet_path(z,y(:0))
  case('empty_features');z=csr_from_arrays(3,0,[1,1,1,1],[integer::],[real(dp)::]);fit=enet_path(z,y)
  case('feature_capacity');x%n_cols=huge(1);fit=enet_path(x,y)
  case('target_length');fit=enet_path(x,y(:2))
  case('data_nan');x%data(1)=nan;fit=enet_path(x,y)
  case('target_inf');y(1)=inf;fit=enet_path(x,y)
  case('weight_length');fit=enet_path(x,y,sample_weight=[1._dp,1._dp])
  case('weight_nan');fit=enet_path(x,y,sample_weight=[nan,1._dp,1._dp])
  case('weight_inf');fit=enet_path(x,y,sample_weight=[inf,1._dp,1._dp])
  case('offset_needs_scale');fit=enet_path(x,y,x_offset=[0._dp,0._dp,0._dp])
  case('offset_length');fit=enet_path(x,y,x_offset=[0._dp],x_scale=[1._dp,1._dp,1._dp])
  case('scale_length');fit=enet_path(x,y,x_offset=[0._dp,0._dp,0._dp],x_scale=[1._dp])
  case('offset_nan');fit=enet_path(x,y,x_offset=[nan,0._dp,0._dp],x_scale=[1._dp,1._dp,1._dp])
  case('scale_inf');fit=enet_path(x,y,x_offset=[0._dp,0._dp,0._dp],x_scale=[inf,1._dp,1._dp])
  case('scale_zero');fit=enet_path(x,y,x_offset=[0._dp,0._dp,0._dp],x_scale=[0._dp,1._dp,1._dp])
  case('mean_overflow')
    fit=enet_path(x,y,x_offset=[huge(1._dp),0._dp,0._dp],x_scale=[tiny(1._dp),1._dp,1._dp])
  case('initial_length');fit=enet_path(x,y,coef_init=[1._dp])
  case('initial_nan');fit=enet_path(x,y,coef_init=[nan,0._dp,0._dp])
  case('xy_empty');fit=enet_path(x,y,xy=[real(dp)::])
  case('xy_nan');fit=enet_path(x,y,xy=[nan])
  case('alpha_negative');fit=enet_path(x,y,alphas=[-1._dp])
  case('alpha_nan');fit=enet_path(x,y,alphas=[nan])
  case('zero_mix_grid');options%l1_ratio=0;fit=enet_path(x,y,options)
  case('negative_tolerance');options%tol=-1;fit=enet_path(x,y,options)
  case('negative_iterations');options%max_iter=-1;fit=enet_path(x,y,options)
  case('negative_seed');options%random_state=-1;fit=enet_path(x,y,options)
  case('selection');options%selection='invalid';fit=enet_path(x,y,options)
  case('precompute');options%precompute='invalid';fit=enet_path(x,y,options)
  case('zero_denominator')
    z=csr_from_arrays(2,1,[1,2,2],[1],[1._dp])
    fit=enet_path(z,[1._dp,0._dp],options,alphas=[1._dp],sample_weight=[-1._dp,1._dp])
  case default;call require(.false.,'unknown sparse coordinate invalid fixture')
  end select
  print *,fit%alphas
  error stop 'sparse coordinate invalid fixture unexpectedly accepted'
end program
