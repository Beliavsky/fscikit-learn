program reference_multitask_cv_driver
  use sklearn_multitask, only: multi_task_elastic_net_cv,multi_task_lasso_cv,multi_task_elastic_net,multi_task_lasso
  use sklearn_common, only: dp
  implicit none
  type(multi_task_elastic_net_cv), target :: net
  type(multi_task_lasso_cv), target :: sparse
  class(multi_task_elastic_net_cv), pointer :: model
  type(multi_task_elastic_net), target :: base_net
  type(multi_task_lasso), target :: base_sparse
  class(multi_task_elastic_net), pointer :: base
  real(dp), allocatable :: x(:,:),y(:,:),prediction(:,:)
  real(dp) :: value
  integer :: input,output,kind,nops,op,n,p,t,i,k,na,nr,custom
  character(512) :: path
  character(32) :: prefix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  read(input,*) kind,nops
  if(kind==1) model=>net
  if(kind==2) model=>sparse
  if(kind==3) base=>base_net
  if(kind==4) base=>base_sparse
  do k=1,nops
    read(input,*) op,n,p,t
    if(kind<=2) then
      read(input,*) model%cv,model%n_alphas,model%max_iter,model%random_state,na,nr,custom
      read(input,*) model%eps,model%tol,model%l1_ratio,model%fit_intercept,model%warm_start
      read(input,*) model%selection
      if(allocated(model%alphas)) deallocate(model%alphas)
      if(allocated(model%l1_ratios)) deallocate(model%l1_ratios)
      if(allocated(model%test_fold)) deallocate(model%test_fold)
      if(na>0) then
        allocate(model%alphas(na)); read(input,*) model%alphas
      end if
      if(nr>0) then
        allocate(model%l1_ratios(nr)); read(input,*) model%l1_ratios
      end if
      if(custom/=0) then
        allocate(model%test_fold(n)); read(input,*) model%test_fold
      end if
    else
      read(input,*) base%alpha,base%l1_ratio,base%tol,base%max_iter,base%random_state, &
        base%fit_intercept,base%warm_start
      read(input,*) base%selection
    end if
    if(allocated(x)) deallocate(x,y)
    allocate(x(n,p),y(n,t))
    do i=1,n
      read(input,*) x(i,:),y(i,:)
    end do
    write(prefix,'(i0,a)') k,'_'
    if(kind<=2) then
      if(op==1) call model%fit(x,y)
      if(op==2) then
        prediction=model%predict(x); call matrix('result',prediction)
      end if
      if(op==3) then
        value=model%score(x,y); call vector('result',[value])
      end if
      call matrix('coef',model%coef_); call vector('intercept',model%intercept_)
      call matrix('alphas',model%alphas_); call tensor('mse',model%mse_path_)
      call tensor('fold_iterations',real(model%fold_n_iter_,dp)); call tensor('fold_gaps',model%fold_dual_gaps_)
      call vector('fold_labels',real(model%fold_labels_,dp)); call vector('fold_sizes',real(model%fold_test_sizes_,dp))
      call vector('selected',[model%alpha_,model%l1_ratio_,model%best_mse_])
      call vector('fit',[model%dual_gap_,model%eps_])
      call vector('meta',real([model%n_features_in_,model%n_iter_,model%n_splits_],dp))
    else
      if(op==1) call base%fit(x,y)
      if(op==2) then
        prediction=base%predict(x); call matrix('result',prediction)
      end if
      if(op==3) then
        value=base%score(x,y); call vector('result',[value])
      end if
      call matrix('coef',base%coef_); call vector('intercept',base%intercept_)
      call vector('fit',[base%dual_gap_,base%eps_]); call vector('meta',real([base%n_features_in_,base%n_iter_],dp))
    end if
  end do
  close(input); close(output)
contains
  subroutine matrix(name,a)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:,:)
    call vector(name,reshape(a,[size(a)]))
  end subroutine
  subroutine tensor(name,a)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:,:,:)
    call vector(name,reshape(a,[size(a)]))
  end subroutine
  subroutine vector(name,a)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:)
    write(output,'(a,1x,*(es25.16e3,1x))') trim(prefix)//name,a
  end subroutine
end program
