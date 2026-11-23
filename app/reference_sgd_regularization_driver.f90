program reference_sgd_regularization_driver
  use sklearn_common, only: dp,require
  use sklearn_sgd, only: sgd_regressor,sgd_classifier
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  use, intrinsic :: iso_fortran_env, only: output_unit
  implicit none
  type(sgd_regressor) :: r
  type(sgd_classifier) :: c
  real(dp), allocatable :: x(:,:),y(:),w(:),before(:,:),coef(:,:),bias(:),prediction(:,:),prob(:,:),trace(:,:)
  integer, allocatable :: labels(:),classes(:),chunks(:)
  integer :: n,p,kind,epochs,average,shuffle,intercept,seed,patience,stream,i,lo,hi,s,channel,ios
  real(dp) :: alpha,eta,power,ratio,tol,eps
  character(32) :: penalty,rate,loss,rejection
  call get_command_argument(1,rejection)
  if(len_trim(rejection)>0.and.rejection/='batch') then
    x=reshape([1._dp,2._dp,3._dp,4._dp],[2,2]);y=[1._dp,2._dp]
    select case(trim(rejection))
    case('ratio_negative');r%l1_ratio=-.1_dp
    case('ratio_large');r%l1_ratio=1.1_dp
    case('ratio_nan');r%l1_ratio=ieee_value(0._dp,ieee_quiet_nan)
    case('optimal_alpha');r%learning_rate='optimal';r%alpha=0
    case('seed_negative');r%random_state=-1
    case('classifier_ratio');c%l1_ratio=-1;call c%fit(x,[1,2]);error stop 'accepted classifier ratio'
    case('classifier_optimal');c%learning_rate='optimal';c%alpha=0
      call c%fit(x,[1,2]);error stop 'accepted classifier alpha'
    case default;error stop 'unknown rejection'
    end select
    call r%fit(x,y);error stop 'accepted invalid controls'
  end if
  do
  read(*,*,iostat=ios) n,p,kind,epochs,average,shuffle,intercept,seed,patience,stream
  if(ios<0) exit
  call require(ios==0,'invalid SGD reference header')
  r=sgd_regressor();c=sgd_classifier()
  if(allocated(x)) deallocate(x,y,w,trace,chunks)
  read(*,*) loss,penalty,rate
  read(*,*) alpha,eta,power,ratio,tol,eps
  allocate(x(n,p),y(n),w(n));read(*,*) x;read(*,*) y;read(*,*) w;before=x
  if(stream>0) then
    allocate(chunks(stream));read(*,*) chunks
  else
    chunks=[n]
  end if
  r%loss=loss;r%penalty=penalty;r%learning_rate=rate;r%max_iter=epochs;r%average=average;r%shuffle=shuffle/=0
  r%fit_intercept=intercept/=0;r%random_state=seed;r%n_iter_no_change=patience
  r%alpha=alpha;r%eta0=eta;r%power_t=power;r%l1_ratio=ratio;r%tol=tol;r%epsilon=eps
  c%loss=loss;c%penalty=penalty;c%learning_rate=rate;c%max_iter=epochs;c%average=average;c%shuffle=shuffle/=0
  c%fit_intercept=intercept/=0;c%random_state=seed;c%n_iter_no_change=patience
  c%alpha=alpha;c%eta0=eta;c%power_t=power;c%l1_ratio=ratio;c%tol=tol
  channel=max(1,kind);if(kind==2) channel=1
  allocate(trace(size(chunks),channel*p+channel+2));lo=1
  if(kind>0) then
    classes=[(-4+7*i,i=1,kind)]
  end if
  do s=1,size(chunks)
    hi=lo+chunks(s)-1
    if(kind==0) then
      if(stream>0) then
        call r%partial_fit(x(lo:hi,:),y(lo:hi),w(lo:hi))
      else
        call r%fit(x,y,w)
      end if
      coef=reshape(r%coef_,[1,p]);bias=[r%intercept_];prediction=reshape(r%predict(x),[n,1])
      prob=reshape([real(dp)::],[0,0]);labels=[integer::]
      trace(s,:)=[reshape(coef,[p]),bias,real(r%n_iter_,dp),r%t_]
    else
      if(stream>0) then
        call c%partial_fit(x(lo:hi,:),int(y(lo:hi)),classes,w(lo:hi))
      else
        call c%fit(x,int(y),w)
      end if
      coef=c%coef_;bias=c%intercept_;prediction=c%decision_function(x);labels=c%predict(x)
      if(loss=='log_loss'.or.loss=='modified_huber') then
        prob=c%predict_proba(x)
      else
        prob=reshape([real(dp)::],[0,0])
      end if
      trace(s,:)=[reshape(coef,[channel*p]),bias,real(c%n_iter_,dp),c%t_]
    end if
    lo=hi+1
  end do
  call matrix(coef);call matrix(reshape(bias,[size(bias),1]));call matrix(prediction);call matrix(prob)
  call matrix(reshape(real(labels,dp),[size(labels),1]));call matrix(trace)
  call require(all(x==before),'SGD feature input changed')
  if(rejection/='batch') exit
  print *, 'END'
  flush(output_unit)
  end do
contains
  subroutine matrix(a)
    real(dp), intent(in) :: a(:,:)
    print *, 'F',shape(a)
    print '(8(es25.16e3,1x))',a
  end subroutine
end program
