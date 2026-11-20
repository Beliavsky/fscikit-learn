program reference_rbm_driver
  use sklearn_rbm, only: bernoulli_rbm
  use sklearn_common, only: dp
  implicit none
  type(bernoulli_rbm) :: model
  real(dp), allocatable :: x(:,:),values(:,:),scores(:)
  integer :: input,output,nops,op,n,p,i,k
  character(512) :: path
  character(32) :: prefix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  read(input,*) nops
  do k=1,nops
    read(input,*) op,n,p
    read(input,*) model%n_components,model%batch_size,model%n_iter,model%random_state,model%learning_rate
    if(allocated(x)) deallocate(x)
    allocate(x(n,p))
    do i=1,n
      read(input,*) x(i,:)
    end do
    write(prefix,'(i0,a)') k,'_'
    select case(op)
    case(1); call model%fit(x)
    case(2); call model%partial_fit(x)
    case(3); values=model%fit_transform(x); call matrix('result',values)
    case(4); values=model%transform(x); call matrix('result',values)
    case(5); values=model%gibbs(x); call matrix('result',values)
    case(6); scores=model%score_samples(x); call vector('result',scores)
    end select
    call matrix('components',model%components_)
    call vector('hidden',model%intercept_hidden_); call vector('visible',model%intercept_visible_)
    call matrix('chain',model%h_samples_); call vector('features',[real(model%n_features_in_,dp)])
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
