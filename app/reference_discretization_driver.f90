program reference_discretization_driver
  use sklearn
  implicit none
  type(kbins_discretizer) :: bins
  real(dp), allocatable :: x(:,:),train(:,:),query(:,:),z(:,:),back(:,:),w(:)
  integer, allocatable :: counts(:)
  integer :: input,output,n,p,i,j,h
  character(1024) :: path
  character(32) :: prefix,suffix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p; allocate(x(n,p),w(n),counts(p))
  do i=1,n
    read(input,*) x(i,:),w(i)
  end do
  close(input)
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  do j=1,12
    train=x; query=x+0.137_dp; bins%n_bins=4; bins%encode='ordinal'
    bins%strategy='uniform'
    if(j==2.or.j==4.or.j==6.or.j==8.or.j==11.or.j==12) bins%encode='onehot-dense'
    if(j==3.or.j==4.or.j==7.or.j==10.or.j==11) bins%strategy='quantile'
    if(j==5.or.j==6.or.j==8.or.j==12) bins%strategy='kmeans'
    if(j==10) then
      train=real(floor(x),dp); query=train+0.137_dp
    end if
    if(j==11) then
      train(:,p)=3; query(:,p)=3.137_dp
    end if
    if(j==7.or.j==8) then
      call bins%fit(train,sample_weight=w)
    else if(j==9.or.j==12) then
      counts=[(2+mod(i,4),i=1,p)]; call bins%fit(train,n_bins=counts)
    else
      call bins%fit(train)
    end if
    if(.not.bins%converged_) error stop 'discretization reference fit did not converge'
    z=bins%transform(train); back=bins%inverse_transform(z)
    write(prefix,'(a,i0)') 'bins',j
    call vector(trim(prefix)//'_counts',real(bins%n_bins_,dp))
    call vector(trim(prefix)//'_width',[real(bins%n_features_out_,dp)])
    do h=1,p
      write(suffix,'(a,i0)') '_edges',h
      call vector(trim(prefix)//trim(suffix),bins%bin_edges_(h)%values)
    end do
    call vector(trim(prefix)//'_train',reshape(z,[size(z)]))
    z=bins%transform(query); call vector(trim(prefix)//'_query',reshape(z,[size(z)]))
    call vector(trim(prefix)//'_inverse',reshape(back,[size(back)]))
  end do
  close(output)
contains
  subroutine vector(name,v)
    character(*), intent(in) :: name
    real(dp), intent(in) :: v(:)
    write(output,'(a,*(1x,es25.16e3))') name,v
  end subroutine
end program
