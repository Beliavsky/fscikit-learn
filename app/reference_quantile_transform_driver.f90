program reference_quantile_transform_driver
  use sklearn
  implicit none
  type(quantile_transformer) :: model
  real(dp), allocatable :: x(:,:),query(:,:),probe(:,:),z(:,:)
  integer :: n,p,i,j,input,output
  character(1024) :: path
  character(32) :: prefix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p; allocate(x(n,p),query(n,p),probe(n,p))
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  do j=1,12
    do i=1,n
      read(input,*) x(i,:)
    end do
    do i=1,n
      read(input,*) query(i,:)
    end do
    do i=1,n
      read(input,*) probe(i,:)
    end do
    model=quantile_transformer(); model%n_quantiles=9; model%subsample=0
    if(mod(j,2)==0) model%output_distribution='normal'
    if(j==3) model%n_quantiles=n+10
    if(j==4) model%n_quantiles=n
    if(j==5.or.j==6) model%n_quantiles=1
    if(j==7) model%n_quantiles=3
    z=model%fit_transform(x); write(prefix,'(a,i0)') 'quantile',j
    call vector('_train',reshape(z,[size(z)]))
    z=model%inverse_transform(z); call vector('_roundtrip',reshape(z,[size(z)]))
    call vector('_references',model%references_)
    call vector('_quantiles',reshape(model%quantiles_,[size(model%quantiles_)]))
    call vector('_metadata',real([model%n_quantiles_,model%n_features_in_],dp))
    z=model%transform(query); call vector('_query',reshape(z,[size(z)]))
    z=model%inverse_transform(probe); call vector('_inverse',reshape(z,[size(z)]))
  end do
  close(input); close(output)
contains
  subroutine vector(suffix,v)
    character(*), intent(in) :: suffix
    real(dp), intent(in) :: v(:)
    write(output,'(a,*(1x,es25.16e3))') trim(prefix)//suffix,v
  end subroutine
end program
