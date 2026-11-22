program target_wrappers_invalid_driver
  use sklearn_common, only: dp,require
  use sklearn_function_transformer
  use sklearn_feature_extraction, only: feature_string
  use sklearn_compose
  use sklearn_tree, only: decision_tree_regressor
  use, intrinsic :: ieee_arithmetic
  implicit none
  type(function_transformer) :: ft
  type(transformed_target_regressor) :: target
  type(decision_tree_regressor) :: tree
  type(feature_string) :: names(2),short(1)
  type(feature_string), allocatable :: output(:)
  real(dp) :: x(4,2),y(4),w(4),bad(4,2),nan,inf,score
  real(dp), allocatable :: z(:,:),v(:)
  character(64) :: arg
  call get_command_argument(1,arg)
  x=1;y=[1._dp,2._dp,3._dp,4._dp];w=1
  nan=ieee_value(0._dp,ieee_quiet_nan);inf=ieee_value(0._dp,ieee_positive_inf)
  names(1)%text='a';names(2)%text='b';short(1)%text='a'
  ft%validate=.true.
  select case(trim(arg))
  case('ft_control')
    ft%feature_names_out='invalid';call ft%fit(x)
  case('ft_fit_empty')
    call ft%fit(x(:0,:))
  case('ft_fit_nan')
    bad=x;bad(1,1)=nan;call ft%fit(bad)
  case('ft_fit_inf')
    bad=x;bad(1,1)=inf;call ft%fit(bad)
  case('ft_fit_names')
    call ft%fit(x,short)
  case('ft_unallocated_name')
    deallocate(names(1)%text);call ft%fit(x,names)
  case('ft_transform_width')
    call ft%fit(x);z=ft%transform(x(:,:1))
  case('ft_inverse_nan')
    bad=x;bad(1,1)=nan;z=ft%inverse_transform(bad)
  case('ft_inverse_empty')
    z=ft%inverse_transform(x(:0,:))
  case('ft_names_none')
    output=ft%get_feature_names_out()
  case('ft_names_unknown')
    ft%feature_names_out='one-to-one';output=ft%get_feature_names_out()
  case('ft_names_width')
    ft%feature_names_out='one-to-one';call ft%fit(x);output=ft%get_feature_names_out(short)
  case('ft_names_differ')
    ft%feature_names_out='one-to-one';call ft%fit(x,names)
    names(1)%text='z';output=ft%get_feature_names_out(names)
  case('ft_names_length')
    ft%feature_names_out='one-to-one';call ft%fit(x,names)
    names(1)%text='a ';output=ft%get_feature_names_out(names)
  case('target_unfitted')
    v=target%predict(x)
  case('target_missing_inverse')
    target%func=>identity;call target%fit(x,y)
  case('target_missing_forward')
    target%inverse_func=>identity;call target%fit(x,y)
  case('target_conflict')
    target%func=>identity;allocate(target%transformer,source=ft);call target%fit(x,y)
  case('target_transformer')
    allocate(target%transformer,source=1);call target%fit(x,y)
  case('target_regressor')
    allocate(target%regressor,source=1);call target%fit(x,y)
  case('target_y_nan')
    y(1)=nan;call target%fit(x,y)
  case('target_y_inf')
    y(1)=inf;call target%fit(x,y)
  case('target_sample_count')
    call target%fit(x,y(:3))
  case('target_weight_count')
    call target%fit(x,y,w(:3))
  case('target_weight_nan')
    w(1)=nan;call target%fit(x,y,w)
  case('target_weight_inf')
    w(1)=inf;call target%fit(x,y,w)
  case('target_weight_negative')
    w(1)=-1;call target%fit(x,y,w)
  case('target_latent_nan')
    target%func=>nonfinite;target%inverse_func=>identity;target%check_inverse=.false.;call target%fit(x,y)
  case('target_latent_rows')
    target%func=>drop_row;target%inverse_func=>identity;target%check_inverse=.false.;call target%fit(x,y)
  case('target_joint_tree')
    allocate(target%regressor,source=tree);call target%fit(x,x)
  case('target_query_width')
    call target%fit(x,y);v=target%predict(x(:,:1))
  case('target_vector_multi')
    call target%fit(x,x);v=target%predict(x)
  case('target_score_shape')
    call target%fit(x,x);score=target%score(x,x(:,:1))
  case default
    call require(.false.,'unknown invalid case')
  end select
  error stop 'invalid case unexpectedly accepted'
contains
  function identity(a) result(b)
    real(dp), intent(in) :: a(:,:)
    real(dp), allocatable :: b(:,:)
    b=a
  end function
  function nonfinite(a) result(b)
    real(dp), intent(in) :: a(:,:)
    real(dp), allocatable :: b(:,:)
    b=a*0+nan
  end function
  function drop_row(a) result(b)
    real(dp), intent(in) :: a(:,:)
    real(dp), allocatable :: b(:,:)
    b=a(:size(a,1)-1,:)
  end function
end program
