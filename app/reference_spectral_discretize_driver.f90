program reference_spectral_discretize_driver
  use sklearn_common,only:dp
  use sklearn_spectral,only:spectral_discretize
  implicit none
  real(dp),allocatable::x(:,:)
  integer,allocatable::labels(:)
  integer::n,k,seed,iterations,restarts,input,output,i
  character(512)::path
  call get_command_argument(1,path);open(newunit=input,file=trim(path),status='old')
  read(input,*)n,k,seed,iterations,restarts
  allocate(x(n,k))
  do i=1,n
    read(input,*)x(i,:)
  end do
  close(input)
  labels=spectral_discretize(x,random_state=seed,n_iter_max=iterations,max_svd_restarts=restarts)
  call get_command_argument(2,path);open(newunit=output,file=trim(path),status='replace')
  write(output,'(*(i0,1x))')labels
  close(output)
end program
