program feature_maps_invalid_driver
  use sklearn_tree_embedding
  use sklearn_kernel_centering
  use sklearn_common, only: dp
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(random_trees_embedding) :: embedding
  type(kernel_centerer) :: centerer
  real(dp) :: x(12,2),wide(12,3),w(12),kernel(3,3),nan,infinity
  real(dp), allocatable :: matrix(:,:)
  integer, allocatable :: indices(:,:)
  character(64), allocatable :: names(:)
  character(32), allocatable :: short_names(:)
  character(32) :: command
  integer :: i
  call get_command_argument(1,command)
  do i=1,12
    x(i,:)=[real(i,dp),real(mod(i,3),dp)]
  end do
  wide=1; w=1; kernel=reshape([(real(i,dp),i=1,9)],[3,3])
  embedding%n_estimators=3; embedding%max_depth=2
  nan=ieee_value(0._dp,ieee_quiet_nan); infinity=ieee_value(0._dp,ieee_positive_inf)
  select case(trim(command))
  case('rte_unfitted')
    matrix=embedding%transform(x)
  case('rte_apply_unfitted')
    indices=embedding%apply(x)
  case('rte_path_unfitted')
    indices=embedding%decision_path(x)
  case('rte_samples_unfitted')
    indices=embedding%estimators_samples()
  case('rte_names_unfitted')
    names=embedding%get_feature_names_out()
  case('rte_estimators_zero')
    embedding%n_estimators=0; call embedding%fit(x)
  case('rte_estimators_negative')
    embedding%n_estimators=-1; call embedding%fit(x)
  case('rte_estimators_overflow')
    embedding%n_estimators=huge(1); call embedding%fit(x)
  case('rte_seed')
    embedding%random_state=-1; call embedding%fit(x)
  case('rte_depth')
    embedding%max_depth=-2; call embedding%fit(x)
  case('rte_leaf')
    embedding%min_samples_leaf=0; call embedding%fit(x)
  case('rte_split')
    embedding%min_samples_split=1; call embedding%fit(x)
  case('rte_impurity_negative')
    embedding%min_impurity_decrease=-1; call embedding%fit(x)
  case('rte_impurity_nan')
    embedding%min_impurity_decrease=nan; call embedding%fit(x)
  case('rte_impurity_inf')
    embedding%min_impurity_decrease=infinity; call embedding%fit(x)
  case('rte_weight_length')
    call embedding%fit(x,w(:11))
  case('rte_weight_nan')
    w(1)=nan; call embedding%fit(x,w)
  case('rte_weight_inf')
    w(1)=infinity; call embedding%fit(x,w)
  case('rte_empty_rows')
    call embedding%fit(x(:0,:))
  case('rte_empty_features')
    call embedding%fit(x(:,:0))
  case('rte_fit_nan')
    x(1,1)=nan; call embedding%fit(x)
  case('rte_fit_inf')
    x(1,1)=infinity; call embedding%fit(x)
  case('rte_fit_range')
    x(1,1)=1.e39_dp; call embedding%fit(x)
  case('rte_query_width')
    call embedding%fit(x); matrix=embedding%transform(wide)
  case('rte_query_empty')
    call embedding%fit(x); matrix=embedding%transform(x(:0,:))
  case('rte_query_inf')
    call embedding%fit(x); x(1,1)=infinity; matrix=embedding%transform(x)
  case('rte_query_range')
    call embedding%fit(x); x(1,1)=1.e39_dp; matrix=embedding%transform(x)
  case('rte_names_width')
    call embedding%fit(x); names=embedding%get_feature_names_out([character(1)::'a'])
  case('rte_warm_shrink')
    call embedding%fit(x); embedding%warm_start=.true.; embedding%n_estimators=2
    call embedding%fit(x)
  case('rte_warm_width')
    call embedding%fit(x); embedding%warm_start=.true.; call embedding%fit(wide)
  case('rte_warm_unknown')
    embedding%n_estimators=1; call embedding%fit(x(:,:1)); embedding%warm_start=.true.
    call embedding%fit(x(:1,:1)); matrix=embedding%transform(x(12:12,:1))
  case('rte_encoder_reset')
    embedding%n_estimators=1; call embedding%fit(x(:,:1)); embedding%warm_start=.true.
    embedding%one_hot_encoder_%handle_unknown_ignore=.true.
    call embedding%fit(x(:1,:1)); matrix=embedding%transform(x(12:12,:1))
  case('kc_unfitted')
    matrix=centerer%transform(kernel)
  case('kc_names_unfitted')
    short_names=centerer%get_feature_names_out()
  case('kc_not_square')
    call centerer%fit(kernel(:2,:))
  case('kc_empty_fit')
    call centerer%fit(kernel(:0,:0))
  case('kc_empty_features')
    call centerer%fit(kernel(:,:0))
  case('kc_fit_nan')
    kernel(1,1)=nan; call centerer%fit(kernel)
  case('kc_fit_inf')
    kernel(1,1)=infinity; call centerer%fit(kernel)
  case('kc_query_width')
    call centerer%fit(kernel); matrix=centerer%transform(kernel(:,:2))
  case('kc_query_nan')
    call centerer%fit(kernel); kernel(1,1)=nan; matrix=centerer%transform(kernel)
  case('kc_query_inf')
    call centerer%fit(kernel); kernel(1,1)=infinity; matrix=centerer%transform(kernel)
  case('kc_query_empty')
    call centerer%fit(kernel); matrix=centerer%transform(kernel(:0,:))
  case('kc_names_width')
    call centerer%fit(kernel); short_names=centerer%get_feature_names_out([character(1)::'a'])
  case default
    error stop 'unknown feature map invalid-input case'
  end select
  print *, 'invalid input was accepted'
end program
