program reference_sgd_fit_driver
  use sklearn_common, only: dp,require
  use sklearn_sgd, only: sgd_regressor,sgd_classifier
  use, intrinsic :: iso_fortran_env, only: output_unit
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(sgd_regressor) :: r
  type(sgd_classifier) :: c
  real(dp), allocatable :: x(:,:),y(:),weights(:),initial(:,:),bias(:),before(:,:),old(:,:),old_bias(:)
  real(dp), allocatable :: coef(:,:),intercepts(:),prediction(:,:),prob(:,:),logs(:,:),seed_matrix(:,:)
  integer, allocatable :: classes(:),labels(:)
  integer :: n,p,kind,stages,seed,ios,s,channel,op,epochs,average,shuffle,intercept,warm,mask,i
  real(dp) :: alpha,eta,power,ratio,epsilon,tol
  character(32) :: loss,penalty,rate,argument
  call get_command_argument(1,argument)
  if(len_trim(argument)>0.and.argument/='batch') then
    call reject(trim(argument));error stop 'accepted invalid SGD fit'
  end if
  do
    read(*,*,iostat=ios) n,p,kind,stages,seed
    if(ios<0) exit
    call require(ios==0,'invalid SGD fit header')
    if(allocated(x)) deallocate(x,y,weights,initial,bias)
    r=sgd_regressor();c=sgd_classifier()
    read(*,*) loss,penalty,rate
    read(*,*) alpha,eta,power,ratio,epsilon,tol
    allocate(x(n,p),y(n),weights(n));read(*,*) x;read(*,*) y;read(*,*) weights;before=x
    channel=max(1,kind);if(kind==2) channel=1
    allocate(initial(channel,p),bias(channel))
    r%loss=loss;r%penalty=penalty;r%learning_rate=rate;r%alpha=alpha;r%eta0=eta;r%power_t=power
    r%l1_ratio=ratio;r%epsilon=epsilon;r%tol=tol;r%random_state=seed;r%n_iter_no_change=2
    c%loss=loss;c%penalty=penalty;c%learning_rate=rate;c%alpha=alpha;c%eta0=eta;c%power_t=power
    c%l1_ratio=ratio;c%epsilon=epsilon;c%tol=tol;c%random_state=seed;c%n_iter_no_change=2
    classes=[(3+7*i,i=0,kind-1)]
    do s=1,stages
      read(*,*) op,epochs,average,shuffle,intercept,warm,mask
      read(*,*) initial;read(*,*) bias;old=initial;old_bias=bias
      r%max_iter=epochs;r%average=average;r%shuffle=shuffle/=0;r%fit_intercept=intercept/=0;r%warm_start=warm/=0
      c%max_iter=epochs;c%average=average;c%shuffle=shuffle/=0;c%fit_intercept=intercept/=0;c%warm_start=warm/=0
      if(op==2) then
        if(kind==0) then
          r%coef_=initial(1,:);r%intercept_=bias(1)
        else
          c%coef_=initial;c%intercept_=bias
        end if
      end if
      if(kind==0) then
        if(op==1) then
          call r%partial_fit(x,y,weights)
        else
          select case(mask)
          case(0);call r%fit(x,y,weights)
          case(1);call r%fit(x,y,weights,coef_init=initial(1,:))
          case(2);call r%fit(x,y,weights,intercept_init=bias(1))
          case(3);call r%fit(x,y,weights,initial(1,:),bias(1))
          end select
        end if
        coef=reshape(r%coef_,[1,p]);intercepts=[r%intercept_];prediction=reshape(r%predict(x),[n,1])
        labels=[integer::];prob=reshape([real(dp)::],[0,0]);logs=prob
        call matrix(coef);call matrix(reshape(intercepts,[channel,1]));call matrix(prediction)
        call matrix(reshape(real(labels,dp),[0,1]));call matrix(prob);call matrix(logs)
        call matrix(reshape([real(r%n_iter_,dp),r%t_,real(r%n_features_in_,dp)],[3,1]))
      else
        if(op==1) then
          call c%partial_fit(x,int(y),classes,weights)
        else
          seed_matrix=initial
          if(btest(mask,4)) seed_matrix=reshape(initial,[2,p/2])
          select case(iand(mask,3))
          case(0);call c%fit(x,int(y),weights)
          case(1)
            if(btest(mask,2)) then
              call c%fit(x,int(y),weights,coef_init=reshape(transpose(initial),[p]))
            else
              call c%fit(x,int(y),weights,coef_init=seed_matrix)
            end if
          case(2)
            if(btest(mask,3)) then
              call c%fit(x,int(y),weights,intercept_init=bias(1))
            else
              call c%fit(x,int(y),weights,intercept_init=bias)
            end if
          case(3)
            if(btest(mask,2)) then
              if(btest(mask,3)) then
                call c%fit(x,int(y),weights,coef_init=reshape(transpose(initial),[p]),intercept_init=bias(1))
              else
                call c%fit(x,int(y),weights,coef_init=reshape(transpose(initial),[p]),intercept_init=bias)
              end if
            else if(btest(mask,3)) then
              call c%fit(x,int(y),weights,coef_init=seed_matrix,intercept_init=bias(1))
            else
              call c%fit(x,int(y),weights,coef_init=seed_matrix,intercept_init=bias)
            end if
          end select
        end if
        coef=c%coef_;intercepts=c%intercept_;prediction=c%decision_function(x);labels=c%predict(x)
        prob=reshape([real(dp)::],[0,0]);logs=prob
        if(loss=='log_loss'.or.loss=='modified_huber') then
          prob=c%predict_proba(x);logs=c%predict_log_proba(x)
        end if
        call matrix(coef);call matrix(reshape(intercepts,[channel,1]));call matrix(prediction)
        call matrix(reshape(real(labels,dp),[n,1]));call matrix(prob);call matrix(logs)
        call matrix(reshape([real(c%n_iter_,dp),c%t_,real(c%n_features_in_,dp)],[3,1]))
      end if
      call require(all(initial==old).and.all(bias==old_bias).and.all(x==before),'SGD initialization input changed')
    end do
    if(argument/='batch') exit
    print *, 'END';flush(output_unit)
  end do
contains
  subroutine matrix(a)
    real(dp), intent(in) :: a(:,:)
    print *, 'F',shape(a)
    print '(8(es25.16e3,1x))',a
  end subroutine
  subroutine reject(name)
    character(*), intent(in) :: name
    real(dp) :: a(4,2),target(4),initial(2),b,nan,inf
    real(dp), allocatable :: z(:,:)
    a=0;target=1;initial=0;b=0;nan=ieee_value(0._dp,ieee_quiet_nan);inf=ieee_value(0._dp,ieee_positive_inf)
    select case(name)
    case('regressor_coef_shape');call r%fit(a,target,coef_init=[0._dp]);return
    case('regressor_coef_nan');initial(1)=nan
    case('regressor_coef_inf');initial(1)=inf
    case('regressor_bias_nan');b=nan
    case('regressor_bias_inf');b=inf
    case('regressor_warm_width');r%warm_start=.true.;call r%fit(a,target);call r%fit(a(:,:1),target);return
    case('regressor_warm_average');r%warm_start=.true.;call r%fit(a,target);r%average=1;call r%fit(a,target);return
    case('classifier_coef_shape');call c%fit(a,[1,2,1,2],coef_init=reshape([0._dp],[1,1]));return
    case('classifier_bias_shape');call c%fit(a,[1,2,1,2],intercept_init=[0._dp,0._dp]);return
    case('classifier_multi_vector');call c%fit(a,[1,2,3,2],coef_init=[0._dp,0._dp,0._dp,0._dp,0._dp,0._dp]);return
    case('classifier_multi_scalar');call c%fit(a,[1,2,3,2],intercept_init=0._dp);return
    case('classifier_coef_nan');call c%fit(a,[1,2,1,2],coef_init=reshape([nan,0._dp],[1,2]));return
    case('classifier_coef_inf');call c%fit(a,[1,2,1,2],coef_init=reshape([inf,0._dp],[1,2]));return
    case('classifier_bias_nan');call c%fit(a,[1,2,1,2],intercept_init=[nan]);return
    case('classifier_bias_inf');call c%fit(a,[1,2,1,2],intercept_init=[inf]);return
    case('classifier_warm_width');c%warm_start=.true.;call c%fit(a,[1,2,1,2]);call c%fit(a(:,:1),[1,2,1,2]);return
    case('classifier_warm_classes');c%warm_start=.true.;call c%fit(a,[1,2,1,2]);call c%fit(a,[1,2,3,2]);return
    case('classifier_epsilon_negative');c%epsilon=-1;call c%fit(a,[1,2,1,2]);return
    case('classifier_epsilon_nan');c%epsilon=nan;call c%fit(a,[1,2,1,2]);return
    case('log_unfitted');z=c%predict_log_proba(a);return
    case('log_loss_unsupported');call c%fit(a,[1,2,1,2]);z=c%predict_log_proba(a);return
    case default;error stop 'unknown SGD fit rejection'
    end select
    call r%fit(a,target,coef_init=initial,intercept_init=b)
  end subroutine
end program
