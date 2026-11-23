program reference_mlp_stochastic_driver
  use sklearn_neural_network, only: mlp_classifier,mlp_regressor
  use sklearn_common, only: dp,require
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  implicit none
  type(mlp_classifier) :: c
  type(mlp_regressor) :: r
  real(dp), allocatable :: x(:,:),q(:,:),y(:),original(:,:)
  integer, allocatable :: labels(:),hidden(:)
  integer :: cases,k,n,p,kind,h,nq,epochs,batch,seed,shuf,nest,patience,i,l
  real(dp) :: eta,alpha,tol,momentum,power,beta1,beta2,eps
  character(16) :: solver,activation,rate
  character(64) :: mode,name
  call get_command_argument(1,mode)
  if(len_trim(mode)>0) then
    call rejection(trim(mode));error stop 'MLP rejection unexpectedly returned'
  end if
  read(*,*) cases
  do k=1,cases
    read(*,*) n,p,kind,h,nq,epochs,batch,seed,shuf,nest,patience
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
      r=mlp_regressor()
      r%hidden_layer_sizes=hidden;r%max_iter=epochs;r%batch_size=batch;r%random_state=seed
      r%shuffle=shuf==1;r%nesterovs_momentum=nest==1;r%n_iter_no_change=patience
      r%solver=solver;r%activation=activation;r%learning_rate=rate
      r%learning_rate_init=eta;r%alpha=alpha;r%tol=tol;r%momentum=momentum
      r%power_t=power;r%beta_1=beta1;r%beta_2=beta2;r%epsilon=eps
      call r%fit(x,y)
      do l=1,size(r%layers_)
        write(name,'(a,i0)') 'weight_',l;call matrix(trim(name),r%layers_(l)%weights)
        write(name,'(a,i0)') 'bias_',l;call vector(trim(name),r%layers_(l)%bias)
        write(name,'(a,i0)') 'moment_',l;call matrix(trim(name),r%layers_(l)%m)
        write(name,'(a,i0)') 'variance_',l;call matrix(trim(name),r%layers_(l)%v)
        write(name,'(a,i0)') 'bias_moment_',l;call vector(trim(name),r%layers_(l)%mb)
        write(name,'(a,i0)') 'bias_variance_',l;call vector(trim(name),r%layers_(l)%vb)
      end do
      call vector('loss_curve',r%loss_curve_)
      call vector('metadata',[real(r%n_iter_,dp),real(r%t_,dp),real(r%n_features_in_,dp), &
        real(r%n_outputs_,dp),real(r%n_layers_,dp),r%loss_,r%best_loss_])
      call vector('prediction',r%predict(q))
    else
      c=mlp_classifier()
      c%hidden_layer_sizes=hidden;c%max_iter=epochs;c%batch_size=batch;c%random_state=seed
      c%shuffle=shuf==1;c%nesterovs_momentum=nest==1;c%n_iter_no_change=patience
      c%solver=solver;c%activation=activation;c%learning_rate=rate
      c%learning_rate_init=eta;c%alpha=alpha;c%tol=tol;c%momentum=momentum
      c%power_t=power;c%beta_1=beta1;c%beta_2=beta2;c%epsilon=eps
      call c%fit(x,labels)
      do l=1,size(c%layers_)
        write(name,'(a,i0)') 'weight_',l;call matrix(trim(name),c%layers_(l)%weights)
        write(name,'(a,i0)') 'bias_',l;call vector(trim(name),c%layers_(l)%bias)
        write(name,'(a,i0)') 'moment_',l;call matrix(trim(name),c%layers_(l)%m)
        write(name,'(a,i0)') 'variance_',l;call matrix(trim(name),c%layers_(l)%v)
        write(name,'(a,i0)') 'bias_moment_',l;call vector(trim(name),c%layers_(l)%mb)
        write(name,'(a,i0)') 'bias_variance_',l;call vector(trim(name),c%layers_(l)%vb)
      end do
      call vector('loss_curve',c%loss_curve_)
      call vector('metadata',[real(c%n_iter_,dp),real(c%t_,dp),real(c%n_features_in_,dp), &
        real(c%n_outputs_,dp),real(c%n_layers_,dp),c%loss_,c%best_loss_])
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
    case('one_class');call c%fit(a,[3,3]);return
    case('reg_unfitted');v=r%predict(a);return
    case('cls_unfitted');prob=c%predict_proba(a);return
    case('reg_width');call r%fit(a,target);v=r%predict(a(:,1:1));return
    case('cls_width');call c%fit(a,[3,7]);prob=c%predict_proba(a(:,1:1));return
    case default;error stop 'unknown MLP rejection'
    end select
    call r%fit(a,target)
  end subroutine
end program
