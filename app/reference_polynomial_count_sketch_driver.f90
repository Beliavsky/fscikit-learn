program reference_polynomial_count_sketch_driver
  use sklearn_polynomial_count_sketch, only: polynomial_count_sketch
  use sklearn_common, only: dp
  implicit none
  type(polynomial_count_sketch) :: model
  real(dp), allocatable :: x(:,:),z(:,:)
  integer :: input,output,n,p,nops,i,k,op
  character(512) :: file
  character(24) :: prefix
  call get_command_argument(1,file); open(newunit=input,file=trim(file),status='old')
  call get_command_argument(2,file); open(newunit=output,file=trim(file),status='replace')
  read(input,*) nops
  do k=1,nops
    read(input,*) op,n,p,model%degree,model%n_components,model%random_state,model%gamma,model%coef0
    if(allocated(x)) deallocate(x)
    allocate(x(n,p))
    do i=1,n
      read(input,*) x(i,:)
    end do
    if(op==1) then
      call model%fit(x); z=model%transform(x)
    else if(op==2) then
      z=model%fit_transform(x)
    else
      z=model%transform(x)
    end if
    write(prefix,'(i0,a)') k,'_'
    call emit('hash',real(reshape(model%index_hash_,[size(model%index_hash_)]),dp))
    call emit('bits',real(reshape(model%bit_hash_,[size(model%bit_hash_)]),dp))
    call emit('features',reshape(z,[size(z)]))
    call emit('gram',reshape(matmul(z,transpose(z)),[size(z,1)**2]))
    call emit('meta',real([model%n_features_in_,model%n_features_out_,size(z,1),size(z,2)],dp))
  end do
  close(input); close(output)
contains
  subroutine emit(name,values)
    character(*), intent(in) :: name
    real(dp), intent(in) :: values(:)
    write(output,'(a,*(1x,es26.17e3))') trim(prefix)//name,values
  end subroutine
end program
