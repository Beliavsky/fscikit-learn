program reference_spectral_qr_driver
  use sklearn_common,only:dp
  use sklearn_spectral_qr,only:spectral_cluster_qr
  implicit none
  real(dp),allocatable::x(:,:)
  integer,allocatable::labels(:),pivots(:)
  integer::n,k,i,input,output
  character(512)::path
  call get_command_argument(1,path);open(newunit=input,file=trim(path),status='old')
  read(input,*)n,k;allocate(x(n,k))
  do i=1,n
    read(input,*)x(i,:)
  end do
  close(input);labels=spectral_cluster_qr(x,pivots)
  call get_command_argument(2,path);open(newunit=output,file=trim(path),status='replace')
  write(output,'(a,*(1x,i0))')'labels',labels
  write(output,'(a,*(1x,i0))')'pivots',pivots
  close(output)
end program
