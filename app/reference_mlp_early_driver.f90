program reference_mlp_early_driver
  use sklearn_neural_network, only: mlp_classifier,mlp_regressor
  use sklearn_common, only: dp,require
  implicit none
  type(mlp_classifier) :: c
  type(mlp_regressor) :: r
  real(dp), allocatable :: x(:,:),q(:,:),y(:,:),qy(:,:),original(:,:)
  integer, allocatable :: hidden(:)
  integer :: cases,k,n,p,kind,outputs,h,nq,epochs,batch,seed,shuf,patience,warm,reset,i,l,nest,early
  real(dp) :: eta,alpha,tol,momentum,power,beta1,beta2,eps,fraction
  character(16) :: solver,activation,rate
  character(64) :: name,mode
  call get_command_argument(1,mode)
  if(len_trim(mode)>0) then
    call rejection(trim(mode));error stop 'MLP target rejection unexpectedly returned'
  end if
  read(*,*) cases
  do k=1,cases
    read(*,*) n,p,kind,outputs,h,nq,epochs,batch,seed,shuf,patience,warm,reset,nest,early
    read(*,*) solver,activation,rate
    read(*,*) eta,alpha,tol,momentum,power,beta1,beta2,eps,fraction
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
      r%early_stopping=early==1;r%validation_fraction=fraction;r%nesterovs_momentum=nest==1
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
      if(allocated(r%validation_scores_)) then
        call vector('validation_scores',r%validation_scores_)
      else
        call vector('validation_scores',[real(dp)::])
      end if
      call vector('best_validation_score',[r%best_validation_score_])
      if(allocated(r%training_indices_)) then
        call vector('training_indices',real(r%training_indices_,dp))
        call vector('validation_indices',real(r%validation_indices_,dp))
      else
        call vector('training_indices',[real(dp)::]);call vector('validation_indices',[real(dp)::])
      end if
      do l=1,size(r%best_layers_)
        write(name,'(a,i0)') 'best_weight_',l;call matrix(trim(name),r%best_layers_(l)%weights)
        write(name,'(a,i0)') 'best_bias_',l;call vector(trim(name),r%best_layers_(l)%bias)
      end do
      call vector('optimizer',[r%optimizer_rate_,real(merge(r%optimizer_step_,0_8,r%optimizer_kind_=='adam'),dp)])
      call vector('metadata',[real(r%n_iter_,dp),real(r%t_,dp),real(r%n_features_in_,dp), &
        real(r%n_outputs_,dp),real(r%n_layers_,dp),r%loss_,real(merge(1,0,r%multilabel_),dp)])
      call matrix('prediction',r%predict_multioutput(q));call vector('score',[r%score(q,qy)])
    else
      if(reset==1) c=mlp_classifier()
      c%hidden_layer_sizes=hidden;c%max_iter=epochs;c%batch_size=batch;c%random_state=seed
      c%early_stopping=early==1;c%validation_fraction=fraction;c%nesterovs_momentum=nest==1
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
      if(allocated(c%validation_scores_)) then
        call vector('validation_scores',c%validation_scores_)
      else
        call vector('validation_scores',[real(dp)::])
      end if
      call vector('best_validation_score',[c%best_validation_score_])
      if(allocated(c%training_indices_)) then
        call vector('training_indices',real(c%training_indices_,dp))
        call vector('validation_indices',real(c%validation_indices_,dp))
      else
        call vector('training_indices',[real(dp)::]);call vector('validation_indices',[real(dp)::])
      end if
      do l=1,size(c%best_layers_)
        write(name,'(a,i0)') 'best_weight_',l;call matrix(trim(name),c%best_layers_(l)%weights)
        write(name,'(a,i0)') 'best_bias_',l;call vector(trim(name),c%best_layers_(l)%bias)
      end do
      call vector('optimizer',[c%optimizer_rate_,real(merge(c%optimizer_step_,0_8,c%optimizer_kind_=='adam'),dp)])
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
    real(dp) :: x(4,1),y(4)
    x(:,1)=[0._dp,1._dp,2._dp,3._dp];y=x(:,1)
    r%hidden_layer_sizes=[integer::];r%max_iter=1;r%early_stopping=.true.
    c%hidden_layer_sizes=[integer::];c%max_iter=1;c%early_stopping=.true.;c%validation_fraction=.5_dp
    select case(mode)
    case('negative_fraction');r%validation_fraction=-.1_dp;call r%fit(x,y)
    case('unit_fraction');r%validation_fraction=1;call r%fit(x,y)
    case('nan_fraction');r%validation_fraction=ieee_value(0._dp,ieee_quiet_nan);call r%fit(x,y)
    case('zero_fraction');r%validation_fraction=0;call r%fit(x,y)
    case('empty_train');r%validation_fraction=.9_dp;call r%fit(x,y)
    case('singleton_class');call c%fit(x,[0,0,0,1])
    case('small_test');c%validation_fraction=.1_dp;call c%fit(x,[0,0,1,1])
    case('small_train');c%validation_fraction=.6_dp;call c%fit(x,[0,0,1,1])
    case('partial_reg');call r%partial_fit(x,y)
    case('partial_cls');call c%partial_fit(x,[0,0,1,1],classes=[0,1])
    case('warm_on');r%early_stopping=.false.;call r%fit(x,y)
      r%warm_start=.true.;r%early_stopping=.true.;call r%fit(x,y)
    case('warm_off');call r%fit(x,y);r%warm_start=.true.;r%early_stopping=.false.;call r%fit(x,y)
    case default;error stop 'unknown MLP early rejection'
    end select
  end subroutine
end program
