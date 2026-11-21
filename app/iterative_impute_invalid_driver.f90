program iterative_impute_invalid_driver
  use sklearn_iterative_impute
  use sklearn_common, only: dp,require
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(iterative_imputer) :: model
  real(dp), allocatable :: x(:,:),q(:,:),z(:,:)
  character(128), allocatable :: names(:)
  character(40) :: name
  real(dp) :: nan,inf
  integer :: j
  call get_command_argument(1,name)
  nan=ieee_value(0._dp,ieee_quiet_nan); inf=ieee_value(0._dp,ieee_positive_inf)
  allocate(x(4,3),q(2,3));x=reshape([(real(j,dp),j=1,12)],[4,3]);x(1,1)=nan;q=2
  select case(trim(name))
  case('unfitted'); z=model%transform(q); stop
  case('names_unfitted'); names=model%get_feature_names_out(); stop
  case('negative_iter'); model%max_iter=-1
  case('huge_iter'); model%max_iter=huge(1)
  case('negative_tol'); model%tol=-1
  case('nan_tol'); model%tol=nan
  case('inf_tol'); model%tol=inf
  case('negative_nearest'); model%n_nearest_features=-1
  case('negative_seed'); model%random_state=-1
  case('strategy'); model%initial_strategy='bad'
  case('order'); model%imputation_order='bad'
  case('nan_fill'); model%fill_value=nan
  case('inf_fill'); model%fill_value=inf
  case('infinite_fit'); x(2,2)=inf
  case('negative_infinite_fit'); x(2,2)=-inf
  case('finite_sentinel_nan'); model%missing_is_nan=.false.
  case('nan_sentinel'); model%missing_is_nan=.false.;model%missing_values=nan
  case('inf_sentinel'); model%missing_is_nan=.false.;model%missing_values=inf
  case('min_nan'); model%min_value=nan
  case('max_nan'); model%max_value=nan
  case('equal_bounds'); model%min_value=1;model%max_value=1
  case('reversed_bounds'); model%min_value=2;model%max_value=-1
  case('min_vector'); model%min_values=[0._dp]
  case('max_vector'); model%max_values=[1._dp,2._dp]
  case('nan_vector'); model%min_values=[0._dp,nan,0._dp]
  case('reversed_vector'); model%min_values=[0._dp,1._dp,2._dp];model%max_values=[1._dp,1._dp,3._dp]
  case('empty_fit'); deallocate(x);allocate(x(0,3))
  case('zero_width_fit'); deallocate(x);allocate(x(4,0))
  case('query_width'); deallocate(q);allocate(q(2,2));q=1
  case('empty_query'); deallocate(q);allocate(q(0,3))
  case('query_inf'); q(1,1)=inf
  case('name_width')
  case('estimator_iterations'); model%estimator%max_iter=0
  case('estimator_prior'); model%estimator%alpha_1=0
  case default; call require(.false.,'unknown invalid iterative-imputer case')
  end select
  call model%fit(x)
  if(trim(name)=='name_width') then
    names=model%get_feature_names_out([character(2)::'x'])
  else
    z=model%transform(q)
  end if
end program
