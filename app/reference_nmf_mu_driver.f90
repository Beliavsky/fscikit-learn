program reference_nmf_mu_driver
  use sklearn_decomposition, only: nmf
  use sklearn_common, only: dp
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(nmf) :: model
  real(dp), allocatable :: x(:,:),q(:,:),w(:,:),w0(:,:),h0(:,:),z(:,:),before(:,:)
  integer :: n,p,m,k,initial_k,steps,seed,shuffle,method,query_enabled,old_iter
  character(8), parameter :: methods(6)=[character(8)::'auto','random','nndsvd','nndsvda','nndsvdar','custom']
  character(40) :: argument
  model%solver='mu'
  call get_command_argument(1,argument)
  if(len_trim(argument)>0) then
    call reject(trim(argument));error stop 'NMF invalid input accepted'
  end if
  read(*,*) n,p,m,k,initial_k,steps,seed,shuffle,method,query_enabled, &
    model%beta_loss,model%alpha_w,model%alpha_h,model%l1_ratio,model%tol
  model%n_components=k;model%max_iter=steps;model%random_state=seed
  model%shuffle=shuffle/=0;model%init=methods(method+1)
  allocate(x(n,p),q(m,p));read(*,*) x;read(*,*) q
  if(model%init=='custom') then
    allocate(w0(n,initial_k),h0(initial_k,p));read(*,*) w0;read(*,*) h0
    w=model%fit_transform(x,w0,h0)
  else
    w=model%fit_transform(x)
  end if
  call matrix(w);call matrix(model%components_)
  call vector([model%reconstruction_err_]);call vector(real([model%n_iter_,model%n_components_,model%n_features_in_],dp))
  call vector([model%mu_initial_error_,model%mu_error_,model%mu_previous_error_]);call vector([merge(1._dp,0._dp,model%converged_)])
  call matrix(model%inverse_transform(w));before=model%components_;old_iter=model%n_iter_
  if(query_enabled/=0) then
    z=model%transform(q);call matrix(z);call matrix(model%inverse_transform(z))
  end if
  if(any(before/=model%components_).or.old_iter/=model%n_iter_) error stop 'NMF transform changed fitted state'
contains
  subroutine vector(values)
    real(dp), intent(in) :: values(:)
    call matrix(reshape(values,[size(values),1]))
  end subroutine
  subroutine matrix(values)
    real(dp), intent(in) :: values(:,:)
    print *,shape(values)
    print '(*(es25.16e3,1x))',values
  end subroutine
  subroutine reject(name)
    character(*), intent(in) :: name
    type(nmf) :: test
    real(dp) :: data(4,3),a(4,2),b(2,3),bad
    real(dp), allocatable :: result(:,:)
    test%solver='mu';test%init='custom';test%max_iter=10;a=.3_dp;b=.4_dp
    data(1,:)=[1._dp,2._dp,3._dp];data(2,:)=[2._dp,1._dp,3._dp]
    data(3,:)=[3._dp,1._dp,2._dp];data(4,:)=[1._dp,3._dp,2._dp]
    bad=ieee_value(0._dp,ieee_quiet_nan)
    select case(name)
    case('solver');test%solver='invalid'
    case('initializer');test%init='invalid'
    case('components');test%n_components=-1
    case('iterations');test%max_iter=0
    case('seed');test%random_state=-1
    case('tol_negative');test%tol=-1
    case('tol_nan');test%tol=bad
    case('alpha_w_negative');test%alpha_w=-1
    case('alpha_h_negative');test%alpha_h=-2
    case('alpha_inf');test%alpha_w=ieee_value(0._dp,ieee_positive_inf)
    case('l1_range');test%l1_ratio=2
    case('l1_nan');test%l1_ratio=bad
    case('data_negative');data(1,1)=-1
    case('data_nan');data(1,1)=bad
    case('data_empty');result=test%fit_transform(data(:0,:),a(:0,:),b);return
    case('missing_w');result=test%fit_transform(data,h0=b);return
    case('missing_h');result=test%fit_transform(data,w0=a);return
    case('w_shape');result=test%fit_transform(data,a(:3,:),b);return
    case('h_shape');result=test%fit_transform(data,a,b(:,:2));return
    case('w_nan');a(1,1)=bad
    case('h_inf');b(1,1)=ieee_value(0._dp,ieee_positive_inf)
    case('w_negative');a(1,1)=-1
    case('h_negative');b(1,1)=-1
    case('w_zero');a=0
    case('h_zero');b=0
    case('nn_rank');test%init='nndsvda';test%n_components=4
    case('unfitted');result=test%transform(data);return
    case('query_width');call test%fit(data,a,b);result=test%transform(data(:,:2));return
    case('query_negative');call test%fit(data,a,b);data(1,1)=-1;result=test%transform(data);return
    case('zero_h_transform')
      test%init='random';data=0;call test%fit(data);result=test%transform(data);return
    case('data_range');data=huge(1._dp)
    case('factor_range');a=huge(1._dp)/8
    case('regularization_range');test%alpha_w=huge(1._dp)
    case('cd_beta');test%solver='cd';test%beta_loss=1
    case('beta_nan');test%beta_loss=bad
    case('beta_inf');test%beta_loss=ieee_value(0._dp,ieee_positive_inf)
    case('beta_zero_input');test%beta_loss=0;data(1,1)=0
    case('beta_negative_input');test%beta_loss=-1;data(1,1)=0
    case('beta_query_zero')
      test%beta_loss=0;call test%fit(data,a,b);data(1,1)=0;result=test%transform(data);return
    case('fitted_h_nan');call test%fit(data,a,b);test%components_(1,1)=bad;result=test%transform(data);return
    case('fitted_h_negative');call test%fit(data,a,b);test%components_(1,1)=-1;result=test%transform(data);return
    case('power_range');test%beta_loss=huge(1._dp)/16
    case('negative_power_range');test%beta_loss=-huge(1._dp)/16
    case('product_range');a=1.e100_dp;b=1.e100_dp
    case default;error stop 'unknown NMF rejection case'
    end select
    result=test%fit_transform(data,a,b)
  end subroutine
end program
