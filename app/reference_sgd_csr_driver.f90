program reference_sgd_csr_driver
  use sklearn_common, only: dp,require
  use sklearn_sgd, only: sgd_regressor,sgd_classifier
  use sklearn_sparse, only: csr_matrix,csr_from_arrays
  use, intrinsic :: iso_fortran_env, only: output_unit
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(sgd_regressor) :: r
  type(sgd_classifier) :: c
  type(csr_matrix) :: graph,original
  real(dp), allocatable :: x(:,:),y(:),weights(:),initial(:,:),bias(:),before(:,:),old(:,:),old_bias(:)
  real(dp), allocatable :: coef(:,:),intercepts(:),prediction(:,:),prob(:,:),logs(:,:),seed_matrix(:,:)
  integer, allocatable :: classes(:),labels(:)
  integer :: n,p,kind,stages,seed,ios,s,channel,op,epochs,average,shuffle,intercept,warm,mask,i,nnz,train_format,query_format
  real(dp) :: alpha,eta,power,ratio,epsilon,tol
  character(32) :: loss,penalty,rate,argument
  call get_command_argument(1,argument)
  if(len_trim(argument)>0.and.argument/='batch') then
    call reject(trim(argument));error stop 'accepted invalid SGD fit'
  end if
  do
    read(*,*,iostat=ios) n,p,kind,stages,seed,nnz
    if(ios<0) exit
    call require(ios==0,'invalid SGD fit header')
    if(allocated(x)) deallocate(x,y,weights,initial,bias)
    r=sgd_regressor();c=sgd_classifier()
    read(*,*) loss,penalty,rate
    read(*,*) alpha,eta,power,ratio,epsilon,tol
    allocate(x(0,0),y(n),weights(n))
    graph%n_rows=n;graph%n_cols=p
    if(allocated(graph%indptr)) deallocate(graph%indptr,graph%indices,graph%data)
    allocate(graph%indptr(n+1),graph%indices(nnz),graph%data(nnz))
    read(*,*) graph%indptr;read(*,*) graph%indices;read(*,*) graph%data
    read(*,*) y;read(*,*) weights;original=graph
    channel=max(1,kind);if(kind==2) channel=1
    allocate(initial(channel,p),bias(channel))
    r%loss=loss;r%penalty=penalty;r%learning_rate=rate;r%alpha=alpha;r%eta0=eta;r%power_t=power
    r%l1_ratio=ratio;r%epsilon=epsilon;r%tol=tol;r%random_state=seed;r%n_iter_no_change=2
    c%loss=loss;c%penalty=penalty;c%learning_rate=rate;c%alpha=alpha;c%eta0=eta;c%power_t=power
    c%l1_ratio=ratio;c%epsilon=epsilon;c%tol=tol;c%random_state=seed;c%n_iter_no_change=2
    classes=[(3+7*i,i=0,kind-1)]
    do s=1,stages
      read(*,*) op,epochs,average,shuffle,intercept,warm,mask,train_format,query_format
      if((train_format==0.or.query_format==0).and.size(x,1)==0) x=graph%to_dense()
      before=x
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
        if(train_format==1) then
        if(op==1) then
          call r%partial_fit(graph,y,weights)
        else
          select case(mask)
          case(0);call r%fit(graph,y,weights)
          case(1);call r%fit(graph,y,weights,coef_init=initial(1,:))
          case(2);call r%fit(graph,y,weights,intercept_init=bias(1))
          case(3);call r%fit(graph,y,weights,initial(1,:),bias(1))
          end select
        end if
        else
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
        end if
        coef=reshape(r%coef_,[1,p]);intercepts=[r%intercept_]
        if(query_format==1) then
          prediction=reshape(r%predict(graph),[n,1])
        else
          prediction=reshape(r%predict(x),[n,1])
        end if
        labels=[integer::];prob=reshape([real(dp)::],[0,0]);logs=prob
        call matrix(coef);call matrix(reshape(intercepts,[channel,1]));call matrix(prediction)
        call matrix(reshape(real(labels,dp),[0,1]));call matrix(prob);call matrix(logs)
        call matrix(reshape([real(r%n_iter_,dp),r%t_,real(r%n_features_in_,dp)],[3,1]))
      else
        if(train_format==1) then
        if(op==1) then
          call c%partial_fit(graph,int(y),classes,weights)
        else
          seed_matrix=initial
          if(btest(mask,4)) seed_matrix=reshape(initial,[2,p/2])
          select case(iand(mask,3))
          case(0);call c%fit(graph,int(y),weights)
          case(1)
            if(btest(mask,2)) then
              call c%fit(graph,int(y),weights,coef_init=reshape(transpose(initial),[p]))
            else
              call c%fit(graph,int(y),weights,coef_init=seed_matrix)
            end if
          case(2)
            if(btest(mask,3)) then
              call c%fit(graph,int(y),weights,intercept_init=bias(1))
            else
              call c%fit(graph,int(y),weights,intercept_init=bias)
            end if
          case(3)
            if(btest(mask,2)) then
              if(btest(mask,3)) then
                call c%fit(graph,int(y),weights,coef_init=reshape(transpose(initial),[p]),intercept_init=bias(1))
              else
                call c%fit(graph,int(y),weights,coef_init=reshape(transpose(initial),[p]),intercept_init=bias)
              end if
            else if(btest(mask,3)) then
              call c%fit(graph,int(y),weights,coef_init=seed_matrix,intercept_init=bias(1))
            else
              call c%fit(graph,int(y),weights,coef_init=seed_matrix,intercept_init=bias)
            end if
          end select
        end if
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
        end if
        coef=c%coef_;intercepts=c%intercept_
        if(query_format==1) then
          prediction=c%decision_function(graph);labels=c%predict(graph)
        else
          prediction=c%decision_function(x);labels=c%predict(x)
        end if
        prob=reshape([real(dp)::],[0,0]);logs=prob
        if(loss=='log_loss'.or.loss=='modified_huber') then
          if(query_format==1) then
            prob=c%predict_proba(graph);logs=c%predict_log_proba(graph)
          else
            prob=c%predict_proba(x);logs=c%predict_log_proba(x)
          end if
        end if
        call matrix(coef);call matrix(reshape(intercepts,[channel,1]));call matrix(prediction)
        call matrix(reshape(real(labels,dp),[n,1]));call matrix(prob);call matrix(logs)
        call matrix(reshape([real(c%n_iter_,dp),c%t_,real(c%n_features_in_,dp)],[3,1]))
      end if
      call require(all(initial==old).and.all(bias==old_bias).and.all(x==before),'SGD initialization input changed')
      call require(all(graph%data==original%data).and.all(graph%indices==original%indices).and. &
        all(graph%indptr==original%indptr),'SGD CSR input changed')
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
    type(csr_matrix) :: a,q
    real(dp), allocatable :: pred(:),scores(:,:)
    real(dp) :: nan,inf
    nan=ieee_value(0._dp,ieee_quiet_nan);inf=ieee_value(0._dp,ieee_positive_inf)
    a=csr_from_arrays(2,2,[1,2,3],[1,2],[1._dp,1._dp]);q=a
    select case(name)
    case('unallocated');deallocate(a%indptr)
    case('empty');a=csr_from_arrays(0,2,[1],[integer::],[real(dp)::])
    case('zero_width');a=csr_from_arrays(2,0,[1,1,1],[integer::],[real(dp)::])
    case('data_nan');a%data(1)=nan
    case('data_inf');a%data(1)=inf
    case('pointer');a%indptr(1)=0
    case('column');a%indices(1)=3
    case('target_count');call r%fit(a,[1._dp]);return
    case('target_nan');call r%fit(a,[nan,1._dp]);return
    case('weight_count');call r%fit(a,[1._dp,2._dp],sample_weight=[1._dp]);return
    case('weight_nan');call r%fit(a,[1._dp,2._dp],sample_weight=[nan,1._dp]);return
    case('reg_query_unfitted');pred=r%predict(a);return
    case('cls_query_unfitted');scores=c%decision_function(a);return
    case('reg_width');call r%fit(a,[1._dp,2._dp]);q%n_cols=3;pred=r%predict(q);return
    case('cls_width');call c%fit(a,[0,1]);q%n_cols=3;scores=c%decision_function(q);return
    case('reg_partial_width');call r%fit(a,[1._dp,2._dp]);q%n_cols=3;call r%partial_fit(q,[1._dp,2._dp]);return
    case('cls_partial_width');call c%fit(a,[0,1]);q%n_cols=3;call c%partial_fit(q,[0,1]);return
    case('cls_target_count');call c%fit(a,[0]);return
    case('cls_initial_shape');call c%fit(a,[0,1],coef_init=[1._dp]);return
    case('reg_initial_shape');call r%fit(a,[1._dp,2._dp],coef_init=[1._dp]);return
    case('proba_loss');call c%fit(a,[0,1]);scores=c%predict_proba(a);return
    case('reg_query_nan');call r%fit(a,[1._dp,2._dp]);q%data(1)=nan;pred=r%predict(q);return
    case('cls_query_nan');call c%fit(a,[0,1]);q%data(1)=nan;scores=c%predict_log_proba(q);return
    case default;error stop 'unknown CSR SGD rejection'
    end select
    call r%fit(a,[1._dp,2._dp])
  end subroutine
end program
