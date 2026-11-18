program reference_ica_driver
  use sklearn
  implicit none
  type(fast_ica) :: model
  real(dp), allocatable :: x(:,:),z(:,:),q(:,:),raw(:,:),prewhite(:,:)
  integer :: n,p,i,j,a,f,b,c,k,output,input
  character(1024) :: path
  character(24) :: algorithms(2)=[character(24)::'parallel','deflation']
  character(24) :: functions(3)=[character(24)::'logcosh','exp','cube']
  character(24) :: whitening(3)=[character(24)::'unit-variance','arbitrary-variance','none']
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p; allocate(x(n,p))
  do i=1,n
    read(input,*) x(i,:)
  end do
  allocate(prewhite(n,p)); raw=x
  do i=1,n
    read(input,*) prewhite(i,:)
  end do
  close(input)
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  do a=1,2
    do f=1,3
      do b=1,3
        x=raw
        if(b==3) x=prewhite
        q=x+0.031_dp
        do c=1,2
          k=p
          if(c==2.and.b/=3) k=p-1
          model%n_components=k; model%algorithm=algorithms(a); model%fun=functions(f)
          model%whiten=whitening(b); model%tol=1.e-7_dp; model%max_iter=2000
          if(c==2) model%max_iter=1
          if(allocated(model%w_init)) deallocate(model%w_init)
          allocate(model%w_init(k,k)); model%w_init=0.13_dp
          do j=1,k
            model%w_init(j,j)=1+0.1_dp*j
          end do
          z=model%fit_transform(x)
          write(output,*) a,f,b,c
          call matrix(model%components_); call matrix(model%mixing_)
          call matrix(z); call matrix(model%transform(q)); call matrix(model%inverse_transform(z))
          write(output,*) model%n_iter_
        end do
      end do
    end do
  end do
  close(output)
contains
  subroutine matrix(values)
    real(dp), intent(in) :: values(:,:)
    write(output,'(*(es25.16e3,1x))') values
  end subroutine
end program
