program reference_mlp_partial_driver
  use sklearn_neural_network, only: mlp_classifier,mlp_regressor
  use sklearn_common, only: dp,require
  implicit none
  type(mlp_classifier) :: c
  type(mlp_regressor) :: r
  real(dp), allocatable :: x(:,:),q(:,:),y(:,:),qy(:,:),original(:,:)
  integer, allocatable :: hidden(:),declared(:)
  integer :: cases,k,n,p,kind,outputs,h,nq,epochs,batch,seed,shuf,patience,nest,warm,reset,op,nclasses,form,i,l
  real(dp) :: eta,alpha,tol,momentum,power,beta1,beta2,eps
  character(16) :: solver,activation,rate
  character(64) :: name,mode
  call get_command_argument(1,mode)
  if(len_trim(mode)>0) then
    call rejection(trim(mode));error stop 'MLP target rejection unexpectedly returned'
  end if
  read(*,*) cases
  do k=1,cases
    read(*,*) n,p,kind,outputs,h,nq,epochs,batch,seed,shuf,patience,nest,warm,reset,op,nclasses,form
    read(*,*) solver,activation,rate
    read(*,*) eta,alpha,tol,momentum,power,beta1,beta2,eps
    allocate(hidden(h),declared(nclasses),x(n,p),q(nq,p),y(n,outputs),qy(nq,outputs))
    if(h>0) read(*,*) hidden
    if(nclasses>0) read(*,*) declared
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
      r%shuffle=shuf==1;r%nesterovs_momentum=nest==1;r%n_iter_no_change=patience;r%warm_start=warm==1
      r%solver=solver;r%activation=activation;r%learning_rate=rate
      r%learning_rate_init=eta;r%alpha=alpha;r%tol=tol;r%momentum=momentum
      r%power_t=power;r%beta_1=beta1;r%beta_2=beta2;r%epsilon=eps
      if(op==0) then
        if(form==0.and.outputs==1) then
          call r%fit(x,y(:,1))
        else
          call r%fit(x,y)
        end if
      else
        if(form==0.and.outputs==1) then
          call r%partial_fit(x,y(:,1))
        else
          call r%partial_fit(x,y)
        end if
      end if
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
      if(.not.r%optimizer_ready_) then
        call vector('optimizer',[real(dp)::])
      else if(r%optimizer_kind_=='adam') then
        call vector('optimizer',[1._dp,r%optimizer_rate_,r%optimizer_eta0_,r%optimizer_beta1_, &
          r%optimizer_beta2_,r%optimizer_epsilon_,real(r%optimizer_step_,dp)])
      else
        call vector('optimizer',[2._dp,r%optimizer_rate_,r%optimizer_eta0_,r%optimizer_momentum_, &
          r%optimizer_power_,real(merge(1,0,r%optimizer_nesterov_),dp), &
          real(schedule(r%optimizer_schedule_),dp)])
      end if
      call vector('metadata',[real(r%n_iter_,dp),real(r%t_,dp),real(r%n_features_in_,dp), &
        real(r%n_outputs_,dp),real(r%n_layers_,dp),r%loss_,real(merge(1,0,r%multilabel_),dp)])
      call matrix('prediction',r%predict_multioutput(q));call vector('score',[r%score(q,qy)])
    else
      if(reset==1) c=mlp_classifier()
      c%hidden_layer_sizes=hidden;c%max_iter=epochs;c%batch_size=batch;c%random_state=seed
      c%shuffle=shuf==1;c%nesterovs_momentum=nest==1;c%n_iter_no_change=patience;c%warm_start=warm==1
      c%solver=solver;c%activation=activation;c%learning_rate=rate
      c%learning_rate_init=eta;c%alpha=alpha;c%tol=tol;c%momentum=momentum
      c%power_t=power;c%beta_1=beta1;c%beta_2=beta2;c%epsilon=eps
      if(op==0) then
        if(kind==1) then
          call c%fit(x,int(y(:,1)))
        else
          call c%fit(x,int(y))
        end if
      else
        if(kind==1) then
          if(nclasses>0) then
            call c%partial_fit(x,int(y(:,1)),declared)
          else
            call c%partial_fit(x,int(y(:,1)))
          end if
        else
          if(nclasses>0) then
            call c%partial_fit(x,int(y),declared)
          else
            call c%partial_fit(x,int(y))
          end if
        end if
      end if
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
      if(.not.c%optimizer_ready_) then
        call vector('optimizer',[real(dp)::])
      else if(c%optimizer_kind_=='adam') then
        call vector('optimizer',[1._dp,c%optimizer_rate_,c%optimizer_eta0_,c%optimizer_beta1_, &
          c%optimizer_beta2_,c%optimizer_epsilon_,real(c%optimizer_step_,dp)])
      else
        call vector('optimizer',[2._dp,c%optimizer_rate_,c%optimizer_eta0_,c%optimizer_momentum_, &
          c%optimizer_power_,real(merge(1,0,c%optimizer_nesterov_),dp), &
          real(schedule(c%optimizer_schedule_),dp)])
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
    deallocate(hidden,declared,x,q,y,qy)
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
  integer function schedule(name) result(value)
    character(*), intent(in) :: name
    value=0
    if(name=='invscaling') value=1
    if(name=='adaptive') value=2
  end function
  subroutine rejection(mode)
    use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan
    character(*), intent(in) :: mode
    real(dp)::a(2,2),target(2,2),nan
    integer::indicator(2,2)
    real(dp),allocatable::v(:),z(:,:)
    integer,allocatable::labels(:),bits(:,:)
    a=1;target=1;indicator=reshape([0,1,1,0],[2,2]);nan=ieee_value(0._dp,ieee_quiet_nan)
    select case(mode)
    case('missing_classes');call c%partial_fit(a,[0,1])
    case('empty_classes');call c%partial_fit(a,[0,1],classes=[integer::])
    case('one_class');call c%partial_fit(a,[0,0],classes=[0])
    case('unknown_class');call c%partial_fit(a,[0,2],classes=[0,1])
    case('changed_classes');call c%partial_fit(a,[0,1],classes=[0,1]);call c%partial_fit(a,[0,1],classes=[0,2])
    case('changed_class_count');call c%partial_fit(a,[0,1],classes=[0,1]);call c%partial_fit(a,[0,1],classes=[0,1,2])
    case('warm_subset');c%warm_start=.true.;call c%partial_fit(a,[0,0],classes=[0,1])
    case('cls_solver');c%solver='lbfgs';call c%partial_fit(a,[0,1],classes=[0,1])
    case('reg_width');call r%partial_fit(a,target);call r%partial_fit(a(:,1:1),target)
    case('cls_width');call c%partial_fit(a,[0,1],classes=[0,1]);call c%partial_fit(a(:,1:1),[0,1])
    case('reg_outputs');call r%partial_fit(a,target);call r%partial_fit(a,target(:,1:1))
    case('indicator_width');call c%partial_fit(a,indicator,classes=[0,1])
      call c%partial_fit(a,reshape([0,1,1,0,0,1],[2,3]))
    case('indicator_values');indicator(1,1)=2;call c%partial_fit(a,indicator,classes=[0,1])
    case('indicator_classes');call c%partial_fit(a,indicator,classes=[3,7])
    case('indicator_mode');call c%partial_fit(a,[0,1],classes=[0,1]);call c%partial_fit(a,indicator)
    case('vector_mode');call c%partial_fit(a,indicator,classes=[0,1]);call c%partial_fit(a,[0,1])
    case('reg_nan');target(1,1)=nan;call r%partial_fit(a,target)
    case('reg_count');call r%partial_fit(a,target(:1,:))
    case('cls_count');call c%partial_fit(a,[0],classes=[0,1])
    case('controls');call r%partial_fit(a,target);r%max_iter=0;call r%partial_fit(a,target)
    case('history');r%solver='lbfgs';call r%partial_fit(a,target);r%solver='adam';call r%partial_fit(a,target)
    case default;error stop 'unknown MLP partial rejection'
    end select
  end subroutine
end program
