program reference_mlp_single_driver
  use sklearn_neural_network, only: mlp_classifier,mlp_regressor
  use sklearn_common, only: dp,require
  use sklearn_sparse, only: csr_matrix
  implicit none
  type(csr_matrix) :: sx,sq,sx_original,sq_original
  type(mlp_classifier) :: c
  type(mlp_regressor) :: r
  real(dp), allocatable :: x(:,:),q(:,:),y(:,:),qy(:,:),original(:,:)
  integer, allocatable :: hidden(:),declared(:)
  integer :: cases,k,n,p,kind,outputs,h,nq,epochs,batch,seed,shuf,patience,nest,warm,reset,op,nclasses,form,i,l
  integer :: sparse_train,sparse_query,early
  real(dp) :: eta,alpha,tol,momentum,power,beta1,beta2,eps,fraction
  character(16) :: solver,activation,rate
  character(64) :: name,mode
  call get_command_argument(1,mode)
  if(len_trim(mode)>0) then
    call rejection(trim(mode));error stop 'MLP target rejection unexpectedly returned'
  end if
  read(*,*) cases
  do k=1,cases
    read(*,*) n,p,kind,outputs,h,nq,epochs,batch,seed,shuf,patience,nest,warm,reset,op,nclasses,form,sparse_train,sparse_query,early
    read(*,*) solver,activation,rate
    read(*,*) eta,alpha,tol,momentum,power,beta1,beta2,eps,fraction
    allocate(hidden(h),declared(nclasses),y(n,outputs),qy(nq,outputs))
    if(h>0) read(*,*) hidden
    if(nclasses>0) read(*,*) declared
    if(sparse_train==1) then
      call read_csr(sx,n,p);allocate(x(0,0))
    else
      allocate(x(n,p))
      do i=1,n
        read(*,*) x(i,:)
      end do
    end if
    do i=1,n
      read(*,*) y(i,:)
    end do
    if(sparse_query==1) then
      call read_csr(sq,nq,p);allocate(q(0,0))
    else
      allocate(q(nq,p))
      do i=1,nq
        read(*,*) q(i,:)
      end do
    end if
    do i=1,nq
      read(*,*) qy(i,:)
    end do
    original=x;sx_original=sx;sq_original=sq
    write(*,'(a,1x,i0)') 'BEGIN',k
    if(kind==0) then
      if(reset==1) r=mlp_regressor()
      r%hidden_layer_sizes=hidden;r%max_iter=epochs;r%batch_size=batch;r%random_state=seed
      r%early_stopping=early==1;r%validation_fraction=fraction
      r%shuffle=shuf==1;r%nesterovs_momentum=nest==1;r%n_iter_no_change=patience;r%warm_start=warm==1
      r%solver=solver;r%activation=activation;r%learning_rate=rate
      r%learning_rate_init=eta;r%alpha=alpha;r%tol=tol;r%momentum=momentum
      r%power_t=power;r%beta_1=beta1;r%beta_2=beta2;r%epsilon=eps
      if(op==0) then
        if(form==0.and.outputs==1) then
          if(sparse_train==1) then
          call r%fit(sx,y(:,1))
          else
          call r%fit(x,y(:,1))
          end if
        else
          if(sparse_train==1) then
          call r%fit(sx,y)
          else
          call r%fit(x,y)
          end if
        end if
      else
        if(form==0.and.outputs==1) then
          if(sparse_train==1) then
          call r%partial_fit(sx,y(:,1))
          else
          call r%partial_fit(x,y(:,1))
          end if
        else
          if(sparse_train==1) then
          call r%partial_fit(sx,y)
          else
          call r%partial_fit(x,y)
          end if
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
      call vector('single_class',[real(merge(1,0,r%single_class_),dp)])
      call vector('metadata',[real(r%n_iter_,dp),real(r%t_,dp),real(r%n_features_in_,dp), &
        real(r%n_outputs_,dp),real(r%n_layers_,dp),r%loss_,real(merge(1,0,r%multilabel_),dp)])
      if(sparse_query==1) then
      call matrix('prediction',r%predict_multioutput(sq));call vector('score',[r%score(sq,qy)])
      else
      call matrix('prediction',r%predict_multioutput(q));call vector('score',[r%score(q,qy)])
      end if
    else
      if(reset==1) c=mlp_classifier()
      c%hidden_layer_sizes=hidden;c%max_iter=epochs;c%batch_size=batch;c%random_state=seed
      c%early_stopping=early==1;c%validation_fraction=fraction
      c%shuffle=shuf==1;c%nesterovs_momentum=nest==1;c%n_iter_no_change=patience;c%warm_start=warm==1
      c%solver=solver;c%activation=activation;c%learning_rate=rate
      c%learning_rate_init=eta;c%alpha=alpha;c%tol=tol;c%momentum=momentum
      c%power_t=power;c%beta_1=beta1;c%beta_2=beta2;c%epsilon=eps
      if(op==0) then
        if(kind==1.and.form==0) then
          if(sparse_train==1) then
          call c%fit(sx,int(y(:,1)))
          else
          call c%fit(x,int(y(:,1)))
          end if
        else
          if(sparse_train==1) then
          call c%fit(sx,int(y))
          else
          call c%fit(x,int(y))
          end if
        end if
      else
        if(kind==1.and.form==0) then
          if(nclasses>0) then
            if(sparse_train==1) then
            call c%partial_fit(sx,int(y(:,1)),declared)
            else
            call c%partial_fit(x,int(y(:,1)),declared)
            end if
          else
            if(sparse_train==1) then
            call c%partial_fit(sx,int(y(:,1)))
            else
            call c%partial_fit(x,int(y(:,1)))
            end if
          end if
        else
          if(nclasses>0) then
            if(sparse_train==1) then
            call c%partial_fit(sx,int(y),declared)
            else
            call c%partial_fit(x,int(y),declared)
            end if
          else
            if(sparse_train==1) then
            call c%partial_fit(sx,int(y))
            else
            call c%partial_fit(x,int(y))
            end if
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
      call vector('single_class',[real(merge(1,0,c%single_class_),dp)])
      call vector('metadata',[real(c%n_iter_,dp),real(c%t_,dp),real(c%n_features_in_,dp), &
        real(c%n_outputs_,dp),real(c%n_layers_,dp),c%loss_,real(merge(1,0,c%multilabel_),dp)])
      if(sparse_query==1) then
      call matrix('probability',c%predict_proba(sq));call matrix('log_probability',c%predict_log_proba(sq))
      if(c%multilabel_) then
        call matrix('prediction',real(c%predict_indicator(sq),dp))
      else
        call matrix('prediction',reshape(real(c%predict(sq),dp),[nq,1]))
      end if
      call vector('classes',real(c%classes_,dp));call vector('score',[c%score(sq,int(qy))])
      else
      call matrix('probability',c%predict_proba(q));call matrix('log_probability',c%predict_log_proba(q))
      if(c%multilabel_) then
        call matrix('prediction',real(c%predict_indicator(q),dp))
      else
        call matrix('prediction',reshape(real(c%predict(q),dp),[nq,1]))
      end if
      call vector('classes',real(c%classes_,dp));call vector('score',[c%score(q,int(qy))])
      end if
    end if
    call require(all(x==original),'MLP changed its input')
    if(sparse_train==1) call unchanged(sx,sx_original)
    if(sparse_query==1) call unchanged(sq,sq_original)
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
  subroutine read_csr(a,n,p)
    type(csr_matrix), intent(out) :: a
    integer, intent(in) :: n,p
    integer :: nnz
    read(*,*) nnz
    a%n_rows=n;a%n_cols=p;allocate(a%indptr(n+1),a%indices(nnz),a%data(nnz))
    read(*,*) a%indptr
    if(nnz>0) then
      read(*,*) a%indices;read(*,*) a%data
    end if
  end subroutine
  subroutine unchanged(a,b)
    type(csr_matrix), intent(in) :: a,b
    call require(a%n_rows==b%n_rows.and.a%n_cols==b%n_cols,'MLP changed CSR shape')
    call require(all(a%indptr==b%indptr).and.all(a%indices==b%indices).and.all(a%data==b%data), &
      'MLP changed CSR storage')
  end subroutine
  subroutine rejection(mode)
    use sklearn_sparse, only: csr_from_dense
    character(*), intent(in) :: mode
    type(csr_matrix) :: a,b
    real(dp) :: dense(4,1)
    integer,allocatable :: labels(:)
    real(dp),allocatable :: prediction(:)
    dense(:,1)=[0._dp,1._dp,2._dp,3._dp];a=csr_from_dense(dense);b=a
    c%hidden_layer_sizes=[integer::];c%max_iter=1
    r%hidden_layer_sizes=[integer::];r%max_iter=1
    select case(mode)
    case('empty_classes');call c%partial_fit(a,[7,7,7,7],classes=[integer::])
    case('unknown_class');call c%partial_fit(a,[7,7,7,8],classes=[7])
    case('changed_count');call c%partial_fit(a,[7,7,7,7],classes=[7]);call c%partial_fit(a,[7,7,7,7],classes=[7,8])
    case('changed_value');call c%partial_fit(a,[7,7,7,7],classes=[7]);call c%partial_fit(a,[8,8,8,8],classes=[8])
    case('warm_classes');call c%fit(a,[7,7,7,7]);c%warm_start=.true.;call c%fit(a,[7,8,7,8])
    case('warm_kind');call c%fit(a,[7,7,7,7]);c%warm_start=.true.;call c%fit(a, &
      reshape([0,1,0,1,1,0,1,0],[4,2]))
    case('partial_early');c%early_stopping=.true.;call c%partial_fit(a,[7,7,7,7],classes=[7])
    case('empty_partition');c%early_stopping=.true.;call c%fit(dense(:1,:),[7])
    case('huge_classifier_query');call c%fit(a,[7,7,7,7]);b%n_rows=huge(1)-1;labels=c%predict(b)
    case('huge_regressor_query');call r%fit(a,dense(:,1));b%n_rows=huge(1)-1;prediction=r%predict(b)
    case default;error stop 'unknown MLP single-class rejection'
    end select
  end subroutine
end program
