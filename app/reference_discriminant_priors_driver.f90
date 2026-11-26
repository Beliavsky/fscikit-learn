program reference_discriminant_priors_driver
  use sklearn_common, only: dp
  use sklearn_discriminant_analysis, only: linear_discriminant_analysis,quadratic_discriminant_analysis
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(linear_discriminant_analysis), target :: linear
  type(quadratic_discriminant_analysis), target :: quadratic
  class(quadratic_discriminant_analysis), pointer :: model
  real(dp), allocatable :: x(:,:),q(:,:),z(:,:)
  integer, allocatable :: y(:)
  integer :: n,p,m,k,mode,components,operation
  character(80) :: arg
  if(command_argument_count()>0) then
    call get_command_argument(1,arg);call reject(trim(arg));stop
  end if
  read(*,*) n,p,m,k,mode,components,operation
  allocate(x(n,p),q(m,p),y(n));read(*,*) x;read(*,*) y;read(*,*) q
  if(mode==5) then
    model=>quadratic;quadratic%reg_param=.2_dp
  else
    model=>linear;linear%n_components=components;linear%store_covariance=.true.;linear%shrinkage=.2_dp
    if(mode==1.or.mode==4) linear%solver='eigen'
    if(mode==2) then
      linear%solver='svd';linear%shrinkage=0
    end if
    if(mode==3.or.mode==4) linear%shrinkage_mode='auto'
  end if
  allocate(model%priors(k));read(*,*) model%priors
  if(any(mode==[1,2,4]).and.operation==0) then
    z=linear%fit_transform(x,y)
  else
    call model%fit(x,y)
    if(any(mode==[1,2,4])) z=linear%transform(x)
  end if
  call vector(real(model%classes_,dp));call matrix(model%means_);call vector(model%priors_);call vector(model%priors)
  call vector([merge(1._dp,0._dp,model%priors_normalized_)]);call cube(model%covariance_)
  call matrix(model%decision_function(q));call matrix(model%predict_proba(q));call matrix(model%predict_log_proba(q));call vector(real(model%predict(q),dp))
  call vector([model%score(x,y)])
  if(mode/=5) then
    call matrix(linear%coef_);call vector(linear%intercept_);call vector([real(linear%reference_class_,dp)])
  end if
  if(any(mode==[1,2,4])) then
    call matrix(linear%scalings_);call vector(linear%explained_variance_ratio_)
    call matrix(z);call matrix(linear%transform(x));call matrix(linear%transform(q))
    call vector([real(linear%n_components_,dp)])
    if(mode==2) then
      call vector(linear%xbar_)
    else
      call vector(linear%eigenvalues_)
    end if
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
  subroutine cube(a)
    real(dp), intent(in) :: a(:,:,:)
    print *,shape(a);print '(*(es26.17e3,1x))',reshape(a,[size(a)])
  end subroutine
  subroutine reject(name)
    character(*), intent(in) :: name
    real(dp) :: data(9,2),bad,wide(4,6)
    integer :: labels(9),i
    bad=ieee_value(0._dp,ieee_quiet_nan)
    do i=1,9
      labels(i)=mod(i,3)+1;data(i,:)=[sin(real(i,dp)),cos(.7_dp*i)]
    end do
    if(index(name,'qda_')==1) then
      model=>quadratic;quadratic%reg_param=.2_dp
    else
      model=>linear;linear%shrinkage=.2_dp
    end if
    model%priors=[.2_dp,.3_dp,.5_dp]
    select case(name)
    case('lda_length','qda_length');model%priors=[.5_dp,.5_dp]
    case('lda_empty','qda_empty');model%priors=[real(dp)::]
    case('lda_negative','qda_negative');model%priors(1)=-1
    case('lda_nan','qda_nan');model%priors(1)=bad
    case('lda_inf','qda_inf');model%priors(1)=ieee_value(0._dp,ieee_positive_inf)
    case('lda_zero','qda_zero');model%priors=0
    case('lda_overflow');model%priors=huge(1._dp)
    case('lda_samples');call linear%fit(data(:3,:),[1,2,3]);return
    case('eigen_singleton')
      linear%solver='eigen';labels=[1,2,2,2,2,3,3,3,3];linear%priors=[1._dp,0._dp,0._dp]
    case('svd_zero');linear%solver='svd';linear%shrinkage=0;data=0
    case('qda_wide_zero')
      wide=reshape([(real(i,dp),i=1,24)],[4,6]);quadratic%reg_param=0;quadratic%priors=[.5_dp,.5_dp]
      call quadratic%fit(wide,[1,1,2,2]);return
    case default;stop 3
    end select
    call model%fit(data,labels)
  end subroutine
end program
