program reference_regression_metrics_driver
  use sklearn_common, only: dp
  use sklearn_metrics, only: mean_pinball_loss,mean_tweedie_deviance,d2_tweedie_score,d2_pinball_score, &
    d2_absolute_error_score,root_mean_squared_log_error
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  real(dp), allocatable :: y(:,:),pred(:,:),weights(:),outputs(:),raw(:),vp(:),vd(:),va(:),vr(:)
  integer :: cases,index,kind,n,q,weighted,output_weighted,rmsle,d2valid,pin_valid,absolute_valid,i,j
  real(dp) :: alpha,power,value,nan,inf
  character(60) :: mode
  nan=ieee_value(0._dp,ieee_quiet_nan);inf=ieee_value(0._dp,ieee_positive_inf)
  call get_command_argument(1,mode)
  if(trim(mode)=='pinball_stdin') then
    read(*,*) n,q,weighted,alpha
    allocate(y(n,q),pred(n,q))
    do i=1,n
      read(*,*) y(i,:)
      read(*,*) pred(i,:)
    end do
    if(weighted==1) then
      allocate(weights(n));read(*,*) weights
    end if
    value=d2_pinball_score(y,pred,weights,alpha)
    error stop 'source pinball rejection accepted'
  end if
  if(trim(mode)=='d2_stdin') then
    read(*,*) n,weighted,power
    allocate(y(n,1),pred(n,1))
    do i=1,n
      read(*,*) y(i,1),pred(i,1)
    end do
    if(weighted==1) then
      allocate(weights(n));read(*,*) weights
    end if
    value=d2_tweedie_score(y(:,1),pred(:,1),weights,power)
    error stop 'source D2 Tweedie rejection accepted'
  end if
  if(len_trim(mode)>0) then
    call reject(trim(mode));error stop 'invalid regression metric input accepted'
  end if
  read(*,*) cases
  do index=1,cases
    read(*,*) kind,n,q,weighted,output_weighted,rmsle,d2valid,pin_valid,absolute_valid,alpha,power
    allocate(y(n,q),pred(n,q))
    do i=1,n
      read(*,*) y(i,:)
      read(*,*) pred(i,:)
    end do
    if(weighted==1) then
      allocate(weights(n));read(*,*) weights
    end if
    if(output_weighted==1) then
      allocate(outputs(q));read(*,*) outputs
    end if
    write(*,'(a,1x,i0)') 'BEGIN',index
    if(kind==0) then
      value=mean_pinball_loss(y,pred,weights,alpha,outputs,raw)
      call vector('pinball',[value]);call vector('pinball_raw',raw)
      if(pin_valid==1) then
        value=d2_pinball_score(y,pred,weights,alpha,outputs,raw)
        call vector('d2_pinball',[value]);call vector('d2_pinball_raw',raw)
      end if
      if(absolute_valid==1) then
        value=d2_absolute_error_score(y,pred,weights,outputs,raw)
        call vector('d2_absolute',[value]);call vector('d2_absolute_raw',raw)
      end if
      allocate(vp(q),vd(q),va(q),vr(q))
      do j=1,q
        vp(j)=mean_pinball_loss(y(:,j),pred(:,j),weights,alpha)
        if(pin_valid==1) vd(j)=d2_pinball_score(y(:,j),pred(:,j),weights,alpha)
        if(absolute_valid==1) va(j)=d2_absolute_error_score(y(:,j),pred(:,j),weights)
      end do
      call vector('vector_pinball',vp)
      if(pin_valid==1) call vector('vector_d2_pinball',vd)
      if(absolute_valid==1) call vector('vector_d2_absolute',va)
      if(rmsle==1) then
        value=root_mean_squared_log_error(y,pred,weights,outputs,raw)
        call vector('rmsle',[value]);call vector('rmsle_raw',raw)
        do j=1,q
          vr(j)=root_mean_squared_log_error(y(:,j),pred(:,j),weights)
        end do
        call vector('vector_rmsle',vr)
      end if
      if(alpha==.5_dp) then
        value=mean_pinball_loss(y,pred,sample_weight=weights,output_weights=outputs)
        call vector('default_pinball',[value])
        if(pin_valid==1) then
          value=d2_pinball_score(y,pred,sample_weight=weights,output_weights=outputs)
          call vector('default_d2_pinball',[value])
        end if
      end if
      deallocate(vp,vd,va,vr)
    else
      call vector('tweedie',[mean_tweedie_deviance(y(:,1),pred(:,1),weights,power)])
      if(d2valid==1) call vector('d2_tweedie',[d2_tweedie_score(y(:,1),pred(:,1),weights,power)])
      if(power==0) then
        call vector('default_tweedie',[mean_tweedie_deviance(y(:,1),pred(:,1),weights)])
        if(d2valid==1) call vector('default_d2_tweedie',[d2_tweedie_score(y(:,1),pred(:,1),weights)])
      end if
    end if
    write(*,'(a)') 'END'
    deallocate(y,pred)
    if(allocated(weights)) deallocate(weights)
    if(allocated(outputs)) deallocate(outputs)
  end do
contains
  subroutine vector(name,x)
    character(*), intent(in) :: name
    real(dp), intent(in) :: x(:)
    integer :: j
    write(*,'(a,1x,i0)',advance='no') name,size(x)
    do j=1,size(x)
      write(*,'(1x,es25.17e3)',advance='no') x(j)
    end do
    write(*,*)
  end subroutine
  subroutine reject(name)
    character(*), intent(in) :: name
    real(dp) :: a(2,2),b(2,2),w(2),ow(2)
    a=1;b=2;w=1;ow=1
    select case(name)
    case('empty');value=mean_pinball_loss([real(dp)::],[real(dp)::])
    case('shape');value=d2_pinball_score(a,b(:,:1))
    case('target_nan');a(1,1)=nan;value=mean_pinball_loss(a,b)
    case('pred_inf');b(1,1)=inf;value=mean_pinball_loss(a,b)
    case('alpha_low');value=mean_pinball_loss(a,b,alpha=-.1_dp)
    case('alpha_high');value=d2_pinball_score(a,b,alpha=1.1_dp)
    case('alpha_nan');value=mean_pinball_loss(a,b,alpha=nan)
    case('alpha_inf');value=d2_pinball_score(a,b,alpha=inf)
    case('weight_length');value=mean_pinball_loss(a,b,sample_weight=w(:1))
    case('weight_nan');w(1)=nan;value=d2_absolute_error_score(a,b,sample_weight=w)
    case('weight_inf');w(1)=inf;value=mean_tweedie_deviance(a(:,1),b(:,1),w)
    case('weight_zero');w=0;value=mean_pinball_loss(a,b,w)
    case('weight_cancel');w=[-1._dp,1._dp];value=d2_pinball_score(a,b,w)
    case('output_length');value=mean_pinball_loss(a,b,output_weights=ow(:1))
    case('output_nan');ow(1)=nan;value=d2_absolute_error_score(a,b,output_weights=ow)
    case('output_inf');ow(1)=inf;value=root_mean_squared_log_error(a,b,output_weights=ow)
    case('output_zero');ow=0;value=mean_pinball_loss(a,b,output_weights=ow)
    case('rmsle_target');a(1,1)=-1;value=root_mean_squared_log_error(a,b)
    case('rmsle_prediction');b(1,1)=-2;value=root_mean_squared_log_error(a,b)
    case('tweedie_constant');value=d2_tweedie_score(a(:,1),b(:,1))
    case('power_gap');value=mean_tweedie_deviance(a(:,1),b(:,1),power=.5_dp)
    case('power_nan');value=d2_tweedie_score(a(:,1),b(:,1),power=nan)
    case('power_inf');value=mean_tweedie_deviance(a(:,1),b(:,1),power=inf)
    case('positive_prediction');b=0;value=mean_tweedie_deviance(a(:,1),b(:,1),power=-1._dp)
    case('nonnegative_target');a=-1;value=mean_tweedie_deviance(a(:,1),b(:,1),power=1._dp)
    case('positive_target');a=0;value=d2_tweedie_score(a(:,1),b(:,1),power=2._dp)
    end select
  end subroutine
end program
