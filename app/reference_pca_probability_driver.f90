program reference_pca_probability_driver
  use sklearn_common, only: dp
  use sklearn_decomposition, only: pca
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_value,ieee_quiet_nan
  implicit none
  type(pca) :: model
  real(dp), allocatable :: x(:,:),q(:,:),z(:,:),train(:,:),values(:),kinds(:),empty(:,:)
  integer :: n,p,m,mode,k,white,i,probability
  real(dp) :: fraction,bad,value
  character(80) :: argument
  if(command_argument_count()>0) then
    call get_command_argument(1,argument);call reject(trim(argument));stop 2
  end if
  read(*,*) n,p,m,mode,k,white,fraction
  allocate(x(n,p),q(m,p));read(*,*) x;read(*,*) q
  model%n_components=k;model%whiten=white/=0;model%variance_fraction=fraction
  select case(mode)
  case(1);model%dimension_mode='variance'
  case(2);model%dimension_mode='mle'
  case(3);model%dimension_mode='zero'
  end select
  call model%fit(x);z=model%transform(q);train=model%transform(x)
  call vector(model%mean_);call matrix(model%components_);call vector(model%explained_variance_)
  call vector(model%explained_variance_ratio_);call vector(model%singular_values_)
  call vector([model%noise_variance_]);call vector(real([model%n_samples_,model%n_components_],dp))
  call matrix(model%get_covariance());call matrix(z);call matrix(train)
  call matrix(model%inverse_transform(z));call matrix(model%inverse_transform(train))
  if(allocated(model%mle_log_likelihood_)) then
    values=model%mle_log_likelihood_;allocate(kinds(size(values)));kinds=0
    do i=1,size(values)
      if(ieee_is_finite(values(i))) cycle
      kinds(i)=-1;values(i)=0
    end do
  else
    allocate(values(0),kinds(0))
  end if
  call vector(values);call vector(kinds)
  probability=0
  if(model%noise_variance_>0.or. &
    (size(model%components_,1)==p.and.all(model%explained_variance_>0))) probability=1
  call vector(real([probability],dp))
  if(probability==1) then
    call matrix(model%get_precision());call vector(model%score_samples(q));call vector([model%score(q)])
  else
    allocate(empty(0,0));call matrix(empty);call vector([real(dp)::]);call vector([real(dp)::])
  end if
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
    real(dp) :: data(6,2),wide(3,4),tied(4,2)
    data(:,1)=[1._dp,2._dp,3._dp,4._dp,5._dp,6._dp]
    data(:,2)=[1._dp,0._dp,2._dp,1._dp,3._dp,2._dp];wide=1
    bad=ieee_value(0._dp,ieee_quiet_nan)
    select case(name)
    case('mode');model%dimension_mode='unsupported'
    case('fraction_zero');model%dimension_mode='variance';model%variance_fraction=0
    case('fraction_one');model%dimension_mode='variance';model%variance_fraction=1
    case('fraction_nan');model%dimension_mode='variance';model%variance_fraction=bad
    case('mle_wide');model%dimension_mode='mle';call model%fit(wide);return
    case('mle_tied')
      tied=reshape([-1._dp,1._dp,0._dp,0._dp,0._dp,0._dp,-1._dp,1._dp],[4,2])
      model%dimension_mode='mle';call model%fit(tied);return
    case('components');model%n_components=-1
    case('large_components');model%n_components=3
    case('one_sample');call model%fit(data(:1,:));return
    case('nan_x');data(1,1)=bad
    case('covariance_unfitted');z=model%get_covariance();return
    case('precision_unfitted');z=model%get_precision();return
    case('score_unfitted');value=model%score(data);return
    case('query_features');call model%fit(data);values=model%score_samples(data(:,:1));return
    case('query_nan');call model%fit(data);data(1,1)=bad;values=model%score_samples(data);return
    case('singular_precision');data=1;call model%fit(data);z=model%get_precision();return
    case('singular_score');data=1;call model%fit(data);value=model%score(data);return
    case('wide_density');call model%fit(wide);z=model%get_precision();return
    case('inverse_zero_shape')
      model%dimension_mode='zero';call model%fit(data);z=model%inverse_transform(data(:,:1));return
    case default;stop 2
    end select
    call model%fit(data)
  end subroutine
end program
