program reference_minibatch_sparse_driver
  use sklearn_minibatch_nmf, only: mini_batch_nmf
  use sklearn_common, only: dp
  use sklearn_sparse,only:csr_matrix
  implicit none
  type(mini_batch_nmf) :: model
  type(csr_matrix)::sparse,before
  real(dp), allocatable :: x(:,:),values(:,:),w0(:,:),h0(:,:)
  integer :: input,output,nops,op,n,p,i,k,initial_k,count,kind
  character(512) :: path
  character(32) :: prefix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  read(input,*) nops
  do k=1,nops
    read(input,*) op,n,p,count,kind
    read(input,*) model%n_components,model%max_iter,model%batch_size,model%random_state,model%max_no_improvement, &
      model%fresh_restarts_max_iter,model%transform_max_iter
    read(input,*) model%beta_loss,model%tol,model%alpha_w,model%alpha_h,model%l1_ratio,model%forget_factor
    read(input,*) model%init,model%fresh_restarts,initial_k
    if(allocated(w0)) deallocate(w0,h0)
    if(initial_k>0) then
      allocate(w0(n,initial_k),h0(initial_k,p))
      do i=1,n
        read(input,*) w0(i,:)
      end do
      do i=1,initial_k
        read(input,*) h0(i,:)
      end do
    end if
    if(kind==0)then
      sparse%n_rows=n;sparse%n_cols=p
      if(allocated(sparse%data))deallocate(sparse%indptr,sparse%indices,sparse%data)
      allocate(sparse%indptr(n+1),sparse%indices(count),sparse%data(count))
      read(input,*)sparse%indptr;read(input,*)sparse%indices;read(input,*)sparse%data;before=sparse
    else
      if(allocated(x))deallocate(x)
      allocate(x(n,p))
      do i=1,n
        read(input,*)x(i,:)
      end do
    end if
    print '(a,i0)','minibatch_stage ',k
    write(prefix,'(i0,a)') k,'_'
    if(kind==0)then
      select case(op)
      case(1);call model%fit(sparse,w0,h0)
      case(2);call model%partial_fit(sparse,w0,h0)
      case(3);values=model%fit_transform(sparse,w0,h0);call matrix('result',values)
      case(4);values=model%transform(sparse);call matrix('result',values)
      end select
      call vector('input_preserved',[real(merge(1,0,all(sparse%indptr==before%indptr).and. &
        all(sparse%indices==before%indices).and.all(sparse%data==before%data)),dp)])
    else
      select case(op)
      case(1);call model%fit(x,w0,h0)
      case(2);call model%partial_fit(x,w0,h0)
      case(3);values=model%fit_transform(x,w0,h0);call matrix('result',values)
      case(4);values=model%transform(x);call matrix('result',values)
      case(5);values=model%inverse_transform(x);call matrix('result',values)
      end select
    end if
    call matrix('components',model%components_)
    call matrix('numerator',model%components_numerator_); call matrix('denominator',model%components_denominator_)
    call vector('meta',[real(model%n_features_in_,dp),real(model%n_components_,dp),real(model%batch_size_,dp), &
      real(model%transform_max_iter_,dp),real(model%n_steps_,dp),real(model%n_iter_,dp)])
    call vector('resolved',[model%beta_loss_,model%gamma_,model%rho_])
    call vector('convergence',[model%ewa_cost_,model%ewa_cost_min_,real(model%no_improvement_,dp)])
    call vector('error',[model%reconstruction_err_])
  end do
  close(input); close(output)
contains
  subroutine matrix(name,a)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:,:)
    call vector(name,reshape(a,[size(a)]))
  end subroutine
  subroutine vector(name,a)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:)
    write(output,'(a,1x,*(es25.16e3,1x))') trim(prefix)//name,a
  end subroutine
end program
