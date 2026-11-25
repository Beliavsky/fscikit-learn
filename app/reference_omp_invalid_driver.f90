program reference_omp_invalid_driver
  use sklearn_omp_paths,only:orthogonal_mp,orthogonal_mp_gram,omp_result
  use sklearn_common,only:dp,require
  use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(omp_result)::fit
  real(dp)::x(2,2),y(2),bad,nan,inf
  character(64)::name
  x=reshape([1._dp,0._dp,0._dp,1._dp],[2,2]);y=[1._dp,2._dp]
  nan=ieee_value(0._dp,ieee_quiet_nan);inf=ieee_value(0._dp,ieee_positive_inf)
  call get_command_argument(1,name)
  select case(trim(name))
  case('empty_x');fit=orthogonal_mp(x(:0,:),y(:0))
  case('target_length');fit=orthogonal_mp(x,y(:1))
  case('empty_targets');fit=orthogonal_mp(x,reshape([real(dp)::],[2,0]))
  case('x_nan');x(1,1)=nan;fit=orthogonal_mp(x,y)
  case('y_inf');y(1)=inf;fit=orthogonal_mp(x,y)
  case('atoms_zero');fit=orthogonal_mp(x,y,n_nonzero_coefs=0)
  case('atoms_negative');fit=orthogonal_mp(x,y,n_nonzero_coefs=-1)
  case('atoms_many');fit=orthogonal_mp(x,y,n_nonzero_coefs=3)
  case('atoms_zero_tolerance');fit=orthogonal_mp(x,y,n_nonzero_coefs=0,tol=1._dp)
  case('tolerance_negative');fit=orthogonal_mp(x,y,tol=-1._dp)
  case('tolerance_nan');fit=orthogonal_mp(x,y,tol=nan)
  case('tolerance_inf');fit=orthogonal_mp(x,y,tol=inf)
  case('precompute_mode');fit=orthogonal_mp(x,y,precompute='invalid')
  case('precompute_matrix');fit=orthogonal_mp(x,reshape(y,[2,1]),precompute='invalid')
  case('gram_empty');fit=orthogonal_mp_gram(x(:0,:0),y(:0))
  case('gram_shape');fit=orthogonal_mp_gram(x(:,:1),y,n_nonzero_coefs=1)
  case('gram_features');fit=orthogonal_mp_gram(x,y(:1),n_nonzero_coefs=1)
  case('gram_default_zero');fit=orthogonal_mp_gram(x,y)
  case('gram_atoms_many');fit=orthogonal_mp_gram(x,y,n_nonzero_coefs=3)
  case('gram_missing_norm');fit=orthogonal_mp_gram(x,y,tol=1._dp)
  case('gram_norm_length')
    fit=orthogonal_mp_gram(x,spread(y,2,2),tol=1._dp,norms_squared=[5._dp])
  case('gram_norm_nan');fit=orthogonal_mp_gram(x,y,tol=1._dp,norms_squared=nan)
  case('gram_norm_inf');fit=orthogonal_mp_gram(x,y,tol=1._dp,norms_squared=inf)
  case('gram_xy_nan');y(1)=nan;fit=orthogonal_mp_gram(x,y,n_nonzero_coefs=1)
  case('gram_nan');x(1,1)=nan;fit=orthogonal_mp_gram(x,y,n_nonzero_coefs=1)
  case('gram_zero_pivot');x=0;fit=orthogonal_mp_gram(x,y,n_nonzero_coefs=1)
  case('gram_negative_pivot');x=-x;fit=orthogonal_mp_gram(x,y,n_nonzero_coefs=1)
  case default;call require(.false.,'unknown OMP invalid fixture')
  end select
  bad=sum(fit%coefficients)
  print *,bad
  error stop 'OMP invalid fixture unexpectedly accepted'
end program
