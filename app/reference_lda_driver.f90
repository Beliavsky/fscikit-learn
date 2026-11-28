program reference_lda_driver
  use sklearn_lda, only: latent_dirichlet_allocation
  use sklearn_sparse, only: csr_matrix,csr_from_dense
  use sklearn_common, only: dp
  implicit none
  type(latent_dirichlet_allocation) :: model
  type(csr_matrix) :: sparse
  character(16) :: storage
  real(dp), allocatable :: x(:,:),values(:,:)
  real(dp) :: scalar
  integer :: input,output,nops,op,n,p,i,k,nz
  character(512) :: path
  character(32) :: prefix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  call get_command_argument(3,storage)
  read(input,*) nops
  do k=1,nops
    read(input,*) op,n,p
    read(input,*) model%n_components,model%max_iter,model%batch_size,model%evaluate_every, &
      model%max_doc_update_iter,model%random_state
    read(input,*) model%doc_topic_prior,model%topic_word_prior,model%learning_decay,model%learning_offset, &
      model%total_samples,model%perp_tol,model%mean_change_tol
    read(input,*) model%learning_method
    if(storage=='raw') then
      read(input,*) nz
      sparse%n_rows=n; sparse%n_cols=p
      if(allocated(sparse%data)) deallocate(sparse%data,sparse%indices,sparse%indptr)
      allocate(sparse%indptr(n+1),sparse%indices(nz),sparse%data(nz))
      read(input,*) sparse%indptr
      read(input,*) sparse%indices
      read(input,*) sparse%data
    else
      if(allocated(x)) deallocate(x)
      allocate(x(n,p))
      do i=1,n
        read(input,*) x(i,:)
      end do
      if(storage=='csr') sparse=csr_from_dense(x)
    end if
    write(prefix,'(i0,a)') k,'_'
    if(storage=='csr'.or.storage=='raw') then
    select case(op)
    case(1); call model%fit(sparse)
    case(2); call model%partial_fit(sparse)
    case(3); values=model%fit_transform(sparse); call matrix('result',values)
    case(4); values=model%transform(sparse); call matrix('result',values)
    case(5); values=model%transform(sparse,normalize=.false.); call matrix('result',values)
    case(6); scalar=model%score(sparse); call vector('result',[scalar])
    case(7); scalar=model%perplexity(sparse); call vector('result',[scalar])
    case(8); scalar=model%perplexity(sparse,sub_sampling=.true.); call vector('result',[scalar])
    case(9); values=model%fit_transform(sparse,normalize=.false.); call matrix('result',values)
    end select
    else
    select case(op)
    case(1); call model%fit(x)
    case(2); call model%partial_fit(x)
    case(3); values=model%fit_transform(x); call matrix('result',values)
    case(4); values=model%transform(x); call matrix('result',values)
    case(5); values=model%transform(x,normalize=.false.); call matrix('result',values)
    case(6); scalar=model%score(x); call vector('result',[scalar])
    case(7); scalar=model%perplexity(x); call vector('result',[scalar])
    case(8); scalar=model%perplexity(x,sub_sampling=.true.); call vector('result',[scalar])
    case(9); values=model%fit_transform(x,normalize=.false.); call matrix('result',values)
    end select
    end if
    call matrix('components',model%components_); call matrix('exp_components',model%exp_dirichlet_component_)
    call vector('meta',[real(model%n_features_in_,dp),real(model%n_iter_,dp),real(model%n_batch_iter_,dp)])
    call vector('priors',[model%doc_topic_prior_,model%topic_word_prior_]); call vector('bound',[model%bound_])
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
