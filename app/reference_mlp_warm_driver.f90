program reference_mlp_warm_driver
  use sklearn_neural_network, only: mlp_classifier,mlp_regressor
  use sklearn_common, only: dp,require
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  implicit none
  type(mlp_classifier) :: c
  type(mlp_regressor) :: r
  real(dp), allocatable :: x(:,:),q(:,:),y(:),original(:,:)
  integer, allocatable :: labels(:),hidden(:)
  integer :: cases,k,n,p,kind,h,nq,epochs,batch,seed,shuf,nest,patience,maxfun,reset,warm,mutate,i,l
  real(dp) :: eta,alpha,tol,momentum,power,beta1,beta2,eps
  character(16) :: solver,activation,rate
  character(64) :: mode,name
  call get_command_argument(1,mode)
  if(len_trim(mode)>0) then
    call rejection(trim(mode));error stop 'MLP rejection unexpectedly returned'
  end if
  read(*,*) cases
  do k=1,cases
    read(*,*) n,p,kind,h,nq,epochs,batch,seed,shuf,nest,patience,maxfun,reset,warm,mutate
    read(*,*) solver,activation,rate
    read(*,*) eta,alpha,tol,momentum,power,beta1,beta2,eps
    allocate(hidden(h),x(n,p),q(nq,p),y(n),labels(n))
    if(h>0) read(*,*) hidden
    do i=1,n
      read(*,*) x(i,:)
    end do
    read(*,*) y
    labels=int(y)
    do i=1,nq
      read(*,*) q(i,:)
    end do
    original=x
    write(*,'(a,1x,i0)') 'BEGIN',k
    if(kind==0) then
      if(reset==1) r=mlp_regressor()
      r%hidden_layer_sizes=hidden;r%max_iter=epochs;r%batch_size=batch;r%random_state=seed
      r%shuffle=shuf==1;r%nesterovs_momentum=nest==1;r%n_iter_no_change=patience
      r%warm_start=warm==1;r%max_fun=maxfun;r%solver=solver;r%activation=activation;r%learning_rate=rate
      r%learning_rate_init=eta;r%alpha=alpha;r%tol=tol;r%momentum=momentum
      r%power_t=power;r%beta_1=beta1;r%beta_2=beta2;r%epsilon=eps
      if(mutate==2) r%t_=2147483640_int64
      if(mutate==3) then
        r%layers_(1)%weights=spread(spread(.1_dp,1,p),2,hidden(1))
        r%layers_(1)%bias=spread(.05_dp,1,hidden(1))
        r%layers_(2)%weights=spread(spread(-.1_dp,1,hidden(1)),2,size(r%layers_(2)%bias))
      end if
      if(mutate==1) then
        do l=1,size(r%layers_)
          r%layers_(l)%weights=.9_dp*r%layers_(l)%weights+.002_dp
          r%layers_(l)%bias=r%layers_(l)%bias-.015_dp*l
        end do
      end if
      call r%fit(x,y)
      do l=1,size(r%layers_)
        write(name,'(a,i0)') 'weight_',l;call matrix(trim(name),r%layers_(l)%weights)
        write(name,'(a,i0)') 'bias_',l;call vector(trim(name),r%layers_(l)%bias)
        write(name,'(a,i0)') 'moment_',l;call matrix(trim(name),r%layers_(l)%m)
        write(name,'(a,i0)') 'variance_',l;call matrix(trim(name),r%layers_(l)%v)
        write(name,'(a,i0)') 'bias_moment_',l;call vector(trim(name),r%layers_(l)%mb)
        write(name,'(a,i0)') 'bias_variance_',l;call vector(trim(name),r%layers_(l)%vb)
      end do
      if(allocated(r%loss_curve_)) then
        call vector('loss_curve',r%loss_curve_)
        call vector('training_state',[r%best_loss_,real(r%no_improvement_count_,dp)])
      else
        call vector('loss_curve',[real(dp)::]);call vector('training_state',[real(dp)::])
      end if
      call vector('metadata',[real(r%n_iter_,dp),real(r%t_,dp),real(r%n_features_in_,dp), &
        real(r%n_outputs_,dp),real(r%n_layers_,dp),r%loss_])
      call vector('prediction',r%predict(q))
    else
      if(reset==1) c=mlp_classifier()
      c%hidden_layer_sizes=hidden;c%max_iter=epochs;c%batch_size=batch;c%random_state=seed
      c%shuffle=shuf==1;c%nesterovs_momentum=nest==1;c%n_iter_no_change=patience
      c%warm_start=warm==1;c%max_fun=maxfun;c%solver=solver;c%activation=activation;c%learning_rate=rate
      c%learning_rate_init=eta;c%alpha=alpha;c%tol=tol;c%momentum=momentum
      c%power_t=power;c%beta_1=beta1;c%beta_2=beta2;c%epsilon=eps
      if(mutate==2) c%t_=2147483640_int64
      if(mutate==3) then
        c%layers_(1)%weights=spread(spread(.1_dp,1,p),2,hidden(1))
        c%layers_(1)%bias=spread(.05_dp,1,hidden(1))
        c%layers_(2)%weights=spread(spread(-.1_dp,1,hidden(1)),2,size(c%layers_(2)%bias))
      end if
      if(mutate==1) then
        do l=1,size(c%layers_)
          c%layers_(l)%weights=.9_dp*c%layers_(l)%weights+.002_dp
          c%layers_(l)%bias=c%layers_(l)%bias-.015_dp*l
        end do
      end if
      call c%fit(x,labels)
      do l=1,size(c%layers_)
        write(name,'(a,i0)') 'weight_',l;call matrix(trim(name),c%layers_(l)%weights)
        write(name,'(a,i0)') 'bias_',l;call vector(trim(name),c%layers_(l)%bias)
        write(name,'(a,i0)') 'moment_',l;call matrix(trim(name),c%layers_(l)%m)
        write(name,'(a,i0)') 'variance_',l;call matrix(trim(name),c%layers_(l)%v)
        write(name,'(a,i0)') 'bias_moment_',l;call vector(trim(name),c%layers_(l)%mb)
        write(name,'(a,i0)') 'bias_variance_',l;call vector(trim(name),c%layers_(l)%vb)
      end do
      if(allocated(c%loss_curve_)) then
        call vector('loss_curve',c%loss_curve_)
        call vector('training_state',[c%best_loss_,real(c%no_improvement_count_,dp)])
      else
        call vector('loss_curve',[real(dp)::]);call vector('training_state',[real(dp)::])
      end if
      call vector('metadata',[real(c%n_iter_,dp),real(c%t_,dp),real(c%n_features_in_,dp), &
        real(c%n_outputs_,dp),real(c%n_layers_,dp),c%loss_])
      call matrix('probability',c%predict_proba(q));call matrix('log_probability',c%predict_log_proba(q))
      call vector('prediction',real(c%predict(q),dp));call vector('classes',real(c%classes_,dp))
    end if
    call require(all(x==original),'MLP modified its input')
    write(*,'(a)') 'END';flush(6)
    deallocate(hidden,x,q,y,labels)
  end do
contains
  subroutine vector(name,a)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:)
    write(*,'(a,1x,*(es25.16e3,1x))') trim(name),a
  end subroutine
  subroutine matrix(name,a)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:,:)
    call vector(name,reshape(a,[size(a)]))
  end subroutine
  subroutine rejection(mode)
    character(*),intent(in)::mode
    real(dp)::a(2,2),target(2),nan
    real(dp),allocatable::v(:),prob(:,:)
    a=1;target=[0._dp,1._dp];nan=ieee_value(0._dp,ieee_quiet_nan)
    select case(mode)
    case('solver');r%solver='unknown'
    case('reg_warm_width');call r%fit(a,target);r%warm_start=.true.;v=r%predict(a);call r%fit(a(:,1:1),target);return
    case('cls_warm_width');call c%fit(a,[3,7]);c%warm_start=.true.;call c%fit(a(:,1:1),[3,7]);return
    case('warm_classes');call c%fit(a,[3,7]);c%warm_start=.true.;call c%fit(a,[3,9]);return
    case('warm_class_count');call c%fit(a,[3,7]);c%warm_start=.true.;call c%fit(a,[3,3]);return
    case('warm_layers');r%hidden_layer_sizes=[2];call r%fit(a,target);r%warm_start=.true.
      r%hidden_layer_sizes=[2,2];call r%fit(a,target);return
    case('warm_lbfgs_width');r%hidden_layer_sizes=[2];r%solver='lbfgs';call r%fit(a,target);r%warm_start=.true.
      r%hidden_layer_sizes=[3];call r%fit(a,target);return
    case('warm_history');r%solver='lbfgs';call r%fit(a,target);r%solver='adam';r%warm_start=.true.
      call r%fit(a,target);return
    case('warm_controls');call r%fit(a,target);r%warm_start=.true.;r%max_iter=0;call r%fit(a,target);return

    case('activation');r%activation='unknown'
    case('rate');r%learning_rate='unknown'
    case('epochs');r%max_iter=0
    case('eta');r%learning_rate_init=0
    case('alpha');r%alpha=-1
    case('tol');r%tol=-1
    case('patience');r%n_iter_no_change=0
    case('batch');r%batch_size=-1
    case('seed');r%random_state=-1
    case('momentum');r%momentum=1.1_dp
    case('power');r%power_t=-1
    case('beta1');r%beta_1=1
    case('beta2');r%beta_2=-.1_dp
    case('epsilon');r%epsilon=0
    case('nonfinite');r%alpha=nan
    case('hidden');r%hidden_layer_sizes=[0]
    case('data');a(1,1)=nan
    case('target');target(1)=nan
    case('target_count');call r%fit(a,[1._dp]);return
    case('empty_batch');call c%fit(a(:0,:),[integer::]);return
    case('reg_unfitted');v=r%predict(a);return
    case('cls_unfitted');prob=c%predict_proba(a);return
    case('reg_width');call r%fit(a,target);v=r%predict(a(:,1:1));return
    case('cls_width');call c%fit(a,[3,7]);prob=c%predict_proba(a(:,1:1));return
    case default;error stop 'unknown MLP rejection'
    end select
    call r%fit(a,target)
  end subroutine
end program
