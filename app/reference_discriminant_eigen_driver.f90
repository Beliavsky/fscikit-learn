program reference_discriminant_eigen_driver
  use sklearn_common, only: dp
  use sklearn_discriminant_analysis, only: linear_discriminant_analysis
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  implicit none
  type(linear_discriminant_analysis) :: model
  real(dp), allocatable :: x(:,:),q(:,:),z(:,:)
  integer, allocatable :: y(:)
  integer :: n,p,m,k,operation,store
  real(dp) :: tol,shrinkage,bad
  character(80) :: arg
  model%solver='eigen'
  if(command_argument_count()>0) then
    call get_command_argument(1,arg);call reject(trim(arg));stop 2
  end if
  read(*,*) n,p,m,k,operation,store,tol,shrinkage
  allocate(x(n,p),q(m,p),y(n));read(*,*) x;read(*,*) y;read(*,*) q
  model%n_components=k;model%tol=tol;model%shrinkage=shrinkage;model%store_covariance=store==1
  if(operation==0) then
    z=model%fit_transform(x,y)
  else
    call model%fit(x,y);z=model%transform(x)
  end if
  call vector(real(model%classes_,dp));call matrix(model%means_);call vector(model%priors_)
  call vector(model%eigenvalues_);call vector(model%explained_variance_ratio_);call matrix(model%scalings_)
  call matrix(z);call matrix(model%transform(x));call matrix(model%transform(q))
  call matrix(model%coef_);call vector(model%intercept_);call matrix(model%decision_function(q))
  call matrix(model%predict_proba(q));call vector(real(model%predict(q),dp))
  call matrix(model%covariance_(:,:,1));call matrix(model%precision_(:,:,1));call vector(model%logdet_)
  call vector([real(model%n_components_,dp)]);call vector([model%score(x,y)])
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
    real(dp) :: data(6,3)
    integer :: labels(6)
    data=reshape([1._dp,2._dp,0._dp,1._dp,3._dp,1._dp,0._dp,1._dp,2._dp,0._dp,2._dp,1._dp, &
      1._dp,0._dp,1._dp,2._dp,3._dp,0._dp],[6,3]);labels=[1,1,1,2,2,2]
    model%shrinkage=.2_dp;bad=ieee_value(0._dp,ieee_quiet_nan)
    select case(name)
    case('solver');model%solver='unknown'
    case('tol');model%tol=-1
    case('tol_nan');model%tol=bad
    case('shrink_nan');model%shrinkage=bad
    case('shrink_range');model%shrinkage=2
    case('shrink_negative');model%shrinkage=-.2_dp
    case('components');model%n_components=-1
    case('components_high');model%n_components=2
    case('one_class');labels=1
    case('samples');labels=[1,2,3,4,5,6]
    case('singular');model%shrinkage=0;data(:,3)=0
    case('zero_cov');data=0
    case('nan_x');data(1,1)=bad
    case('labels');call model%fit(data,labels(:5));return
    case('unfitted');z=model%transform(data);return
    case('width');call model%fit(data,labels);z=model%transform(data(:,:2));return
    case('nan_query');call model%fit(data,labels);data(1,1)=bad;z=model%transform(data);return
    case('empty_query');call model%fit(data,labels);z=model%transform(data(:0,:));return
    case('lsqr_transform');model%solver='lsqr';call model%fit(data,labels);z=model%transform(data);return
    case('refit_lsqr')
      call model%fit(data,labels);model%solver='lsqr';call model%fit(data,labels)
      model%solver='eigen';z=model%transform(data);return
    case default;stop 3
    end select
    call model%fit(data,labels)
  end subroutine
end program
