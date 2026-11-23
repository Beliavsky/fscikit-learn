program reference_mlp_targets_driver
  use sklearn_neural_network, only: mlp_classifier,mlp_regressor
  use sklearn_common, only: dp,require
  implicit none
  type(mlp_classifier) :: c
  type(mlp_regressor) :: r
  real(dp), allocatable :: x(:,:),q(:,:),y(:,:),qy(:,:),original(:,:)
  integer, allocatable :: hidden(:)
  integer :: cases,k,n,p,kind,outputs,h,nq,epochs,batch,seed,shuf,patience,warm,reset,i,l
  real(dp) :: eta,alpha,tol,momentum,power,beta1,beta2,eps
  character(16) :: solver,activation,rate
  character(64) :: name,mode
  call get_command_argument(1,mode)
  if(len_trim(mode)>0) then
    call rejection(trim(mode));error stop 'MLP target rejection unexpectedly returned'
  end if
  read(*,*) cases
  do k=1,cases
    read(*,*) n,p,kind,outputs,h,nq,epochs,batch,seed,shuf,patience,warm,reset
    read(*,*) solver,activation,rate
    read(*,*) eta,alpha,tol,momentum,power,beta1,beta2,eps
    allocate(hidden(h),x(n,p),q(nq,p),y(n,outputs),qy(nq,outputs))
    if(h>0) read(*,*) hidden
    do i=1,n
      read(*,*) x(i,:)
    end do
    do i=1,n
      read(*,*) y(i,:)
    end do
    do i=1,nq
      read(*,*) q(i,:)
    end do
    do i=1,nq
      read(*,*) qy(i,:)
    end do
    original=x
    write(*,'(a,1x,i0)') 'BEGIN',k
    if(kind==0) then
      if(reset==1) r=mlp_regressor()
      r%hidden_layer_sizes=hidden;r%max_iter=epochs;r%batch_size=batch;r%random_state=seed
      r%shuffle=shuf==1;r%n_iter_no_change=patience;r%warm_start=warm==1
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
      if(allocated(r%loss_curve_)) then
        call vector('loss_curve',r%loss_curve_)
        call vector('training_state',[r%best_loss_,real(r%no_improvement_count_,dp)])
      else
        call vector('loss_curve',[real(dp)::]);call vector('training_state',[real(dp)::])
      end if
      if(allocated(r%gradient_)) then
        call vector('gradient',r%gradient_)
      else
        call vector('gradient',[real(dp)::])
      end if
      call vector('metadata',[real(r%n_iter_,dp),real(r%t_,dp),real(r%n_features_in_,dp), &
        real(r%n_outputs_,dp),real(r%n_layers_,dp),r%loss_,real(merge(1,0,r%multilabel_),dp)])
      call matrix('prediction',r%predict_multioutput(q));call vector('score',[r%score(q,qy)])
    else
      if(reset==1) c=mlp_classifier()
      c%hidden_layer_sizes=hidden;c%max_iter=epochs;c%batch_size=batch;c%random_state=seed
      c%shuffle=shuf==1;c%n_iter_no_change=patience;c%warm_start=warm==1
      c%solver=solver;c%activation=activation;c%learning_rate=rate
      c%learning_rate_init=eta;c%alpha=alpha;c%tol=tol;c%momentum=momentum
      c%power_t=power;c%beta_1=beta1;c%beta_2=beta2;c%epsilon=eps
      call c%fit(x,int(y))
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
      if(allocated(c%gradient_)) then
        call vector('gradient',c%gradient_)
      else
        call vector('gradient',[real(dp)::])
      end if
      call vector('metadata',[real(c%n_iter_,dp),real(c%t_,dp),real(c%n_features_in_,dp), &
        real(c%n_outputs_,dp),real(c%n_layers_,dp),c%loss_,real(merge(1,0,c%multilabel_),dp)])
      call matrix('probability',c%predict_proba(q));call matrix('log_probability',c%predict_log_proba(q))
      if(c%multilabel_) then
        call matrix('prediction',real(c%predict_indicator(q),dp))
      else
        call matrix('prediction',reshape(real(c%predict(q),dp),[nq,1]))
      end if
      call vector('classes',real(c%classes_,dp));call vector('score',[c%score(q,int(qy))])
    end if
    call require(all(x==original),'MLP changed its input')
    write(*,'(a)') 'END';flush(6)
    deallocate(hidden,x,q,y,qy)
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
    use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan
    character(*), intent(in) :: mode
    real(dp)::a(2,2),target(2,2),nan
    integer::indicator(2,2)
    real(dp),allocatable::v(:),z(:,:)
    integer,allocatable::labels(:),bits(:,:)
    a=1;target=1;indicator=reshape([0,1,1,0],[2,2]);nan=ieee_value(0._dp,ieee_quiet_nan)
    select case(mode)
    case('reg_empty');call r%fit(a,target(:,:0))
    case('reg_count');call r%fit(a,target(:1,:))
    case('reg_nan');target(1,1)=nan;call r%fit(a,target)
    case('reg_unfitted');z=r%predict_multioutput(a)
    case('reg_vector');call r%fit(a,target);v=r%predict(a)
    case('reg_score_width');call r%fit(a,target);nan=r%score(a,target(:,1:1))
    case('reg_query_width');call r%fit(a,target);z=r%predict_multioutput(a(:,1:1))
    case('reg_warm_width');call r%fit(a,target);r%warm_start=.true.;call r%fit(a,target(:,1:1))
    case('cls_empty');call c%fit(a,indicator(:,:0))
    case('cls_count');call c%fit(a,indicator(:1,:))
    case('cls_values');indicator(1,1)=2;call c%fit(a,indicator)
    case('cls_unfitted');bits=c%predict_indicator(a)
    case('cls_vector');call c%fit(a,indicator);labels=c%predict(a)
    case('cls_scalar_indicator');call c%fit(a,[0,1]);bits=c%predict_indicator(a)
    case('cls_score_width');call c%fit(a,indicator);nan=c%score(a,indicator(:,1:1))
    case('cls_score_values');call c%fit(a,indicator);indicator(1,1)=-1;nan=c%score(a,indicator)
    case('cls_warm_width');call c%fit(a,indicator);c%warm_start=.true.
      call c%fit(a,reshape([0,1,1,0,0,1],[2,3]))
    case('cls_warm_kind');call c%fit(a,[0,1]);c%warm_start=.true.;call c%fit(a,indicator)
    case('cls_warm_vector');call c%fit(a,indicator);c%warm_start=.true.;call c%fit(a,[0,1])
    case default;error stop 'unknown MLP target rejection'
    end select
  end subroutine
end program
