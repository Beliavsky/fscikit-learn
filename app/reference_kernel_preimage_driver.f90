program reference_kernel_preimage_driver
  use sklearn_common, only: dp
  use sklearn_kernel_pca, only: kernel_pca
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  implicit none
  type(kernel_pca) :: model
  real(dp), allocatable :: x(:,:),q(:,:),z(:,:),t(:,:)
  integer :: n,p,m,k,code,remove,i,operation
  real(dp) :: alpha,gamma,coef0,bad
  character(16), parameter :: kernels(5)=[character(16)::'linear','rbf','poly','cosine','sigmoid']
  character(80) :: arg
  if(command_argument_count()>0) then
    call get_command_argument(1,arg);call reject(trim(arg));stop 2
  end if
  read(*,*) n,p,m,k,code,remove,operation,alpha,gamma,coef0
  allocate(x(n,p),q(m,p));read(*,*) x;read(*,*) q
  model%n_components=k;model%kernel=kernels(code+1);model%remove_zero_eig=remove==1
  model%degree=2;model%alpha=alpha;model%gamma=gamma;model%coef0=coef0;model%fit_inverse_transform=.true.
  if(operation==0) then
    z=model%fit_transform(x)
  else
    call model%fit(x);z=model%x_transformed_fit_
  end if
  call vector(model%eigenvalues_);call matrix(z);call matrix(model%transform(x))
  t=model%transform(q);call matrix(t)
  call matrix(model%x_transformed_fit_);call matrix(model%dual_coef_)
  call matrix(model%inverse_transform(z));call matrix(model%inverse_transform(t))
  t=.73_dp*t+.11_dp;call matrix(model%inverse_transform(t))
  model%fit_inverse_transform=.false.;call model%fit(x)
  call vector([merge(1._dp,0._dp,.not.allocated(model%dual_coef_).and..not.allocated(model%x_transformed_fit_))])
contains
  subroutine vector(a)
    real(dp), intent(in) :: a(:)
    print *,size(a);print '(*(es26.17e3,1x))',a
  end subroutine
  subroutine matrix(a)
    real(dp), intent(in) :: a(:,:)
    print *,shape(a);print '(*(es26.17e3,1x))',reshape(a,[size(a)])
  end subroutine
  subroutine reject(name)
    character(*), intent(in) :: name
    real(dp) :: data(4,3)
    data=reshape([1._dp,2._dp,0._dp,1._dp,0._dp,1._dp,2._dp,0._dp,1._dp,0._dp,1._dp,2._dp],[4,3])
    model%n_components=2;model%fit_inverse_transform=.true.;bad=ieee_value(0._dp,ieee_quiet_nan)
    select case(name)
    case('negative_alpha');model%alpha=-1
    case('nan_alpha');model%alpha=bad
    case('nan_gamma');model%gamma=bad
    case('nan_coef0');model%coef0=bad
    case('precomputed');model%kernel='precomputed'
    case('unfitted');z=model%inverse_transform(data(:,:2));return
    case('disabled');model%fit_inverse_transform=.false.;call model%fit(data);z=model%inverse_transform(data(:,:2));return
    case('width');call model%fit(data);z=model%inverse_transform(data);return
    case('query_nan');call model%fit(data);data(1,1)=bad;z=model%inverse_transform(data(:,:2));return
    case('query_empty');call model%fit(data);z=model%inverse_transform(data(:0,:2));return
    case('singular');model%alpha=0;data=0
    case('zero_components');model%n_components=0;data=0
    case('indefinite')
      model%kernel='sigmoid';model%gamma=.05_dp;model%coef0=-.3_dp;model%alpha=.01_dp;model%n_components=1
      data(1:2,:)=-.4_dp;data(3:4,:)=.7_dp
    case('refit_disabled')
      call model%fit(data);model%fit_inverse_transform=.false.;call model%fit(data)
      model%fit_inverse_transform=.true.;z=model%inverse_transform(data(:,:2));return
    case default;stop 3
    end select
    call model%fit(data)
  end subroutine
end program
