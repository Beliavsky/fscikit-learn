program nca_invalid_driver
  use sklearn_nca
  use sklearn_common, only: dp
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(neighborhood_components_analysis) :: model
  real(dp) :: x(6,3),a(2,3),g(2,3),f,nan,inf
  real(dp), allocatable :: z(:,:)
  integer :: y(6),i
  character(40) :: arg
  call get_command_argument(1,arg)
  do i=1,6
    x(i,:)=[real(i,dp),sin(real(i,dp)),cos(real(i,dp))]
  end do
  y=[0,1,0,1,0,1];a=.1_dp
  nan=ieee_value(0._dp,ieee_quiet_nan);inf=ieee_value(0._dp,ieee_positive_inf)
  model%init='identity';model%max_iter=2
  select case(trim(arg))
  case('unfitted');z=model%transform(x);stop 0
  case('empty_rows');call model%fit(x(:0,:),y(:0));stop 0
  case('empty_columns');call model%fit(x(:,:0),y);stop 0
  case('one_row');call model%fit(x(:1,:),y(:1));stop 0
  case('wrong_labels');call model%fit(x,y(:5));stop 0
  case('nan_x');x(1,1)=nan
  case('inf_x');x(1,1)=inf
  case('negative_components');model%n_components=-1
  case('large_components');model%n_components=4
  case('zero_iterations');model%max_iter=0
  case('negative_iterations');model%max_iter=-1
  case('negative_tol');model%tol=-1
  case('nan_tol');model%tol=nan
  case('inf_tol');model%tol=inf
  case('large_tol');model%tol=huge(1._dp)
  case('negative_seed');model%random_state=-1
  case('unknown_init');model%init='unknown'
  case('empty_init');allocate(model%init_matrix(0,3))
  case('init_width');allocate(model%init_matrix(2,4));model%init_matrix=0
  case('init_height');allocate(model%init_matrix(4,3));model%init_matrix=0
  case('init_components');model%init_matrix=a;model%n_components=1
  case('nan_init');model%init_matrix=a;model%init_matrix(1,1)=nan
  case('pca_samples');model%init='pca';model%n_components=3;call model%fit(x(:2,:),y(:2));stop 0
  case('lda_all_classes');model%init='lda';model%n_components=1;y=[1,2,3,4,5,6]
  case('lda_one_class');model%init='lda';model%n_components=1;y=1
  case('lda_within_rank');model%init='lda';model%n_components=1;x=0
  case('lda_between_rank');model%init='lda';model%n_components=1;x(4:6,:)=x(:3,:);y=[0,0,0,1,1,1]
  case('warm_width')
    call model%fit(x,y);model%warm_start=.true.;call model%fit(x(:,:2),y);stop 0
  case('query_width');call model%fit(x,y);z=model%transform(x(:,:2));stop 0
  case('query_nan');call model%fit(x,y);x(1,1)=nan;z=model%transform(x);stop 0
  case('query_inf');call model%fit(x,y);x(1,1)=inf;z=model%transform(x);stop 0
  case('query_overflow');call model%fit(x,y);x=huge(1._dp);z=model%transform(x);stop 0
  case('large_input');x=huge(1._dp)
  case('gradient_shape');call nca_objective_gradient(x,y,a(:,:2),f,g);stop 0
  case('gradient_output');call nca_objective_gradient(x,y,a,f,g(:1,:));stop 0
  case('gradient_one_row');call nca_objective_gradient(x(:1,:),y(:1),a,f,g);stop 0
  case('gradient_nan');a(1,1)=nan;call nca_objective_gradient(x,y,a,f,g);stop 0
  case('gradient_embedding');a=huge(1._dp);call nca_objective_gradient(x,y,a,f,g);stop 0
  case('gradient_distances');a=1.e200_dp;call nca_objective_gradient(x,y,a,f,g);stop 0
  case('gradient_products');x=1.e308_dp;a=1.e-308_dp;call nca_objective_gradient(x,y,a,f,g);stop 0
  case default;stop 2
  end select
  call model%fit(x,y)
end program
