program reference_numpy_nan_ordering_driver
 use sklearn_common,only:dp
 use sklearn_numpy_ordering,only:numpy_argsort,numpy_argpartition
 implicit none
 real(dp),allocatable::x(:)
 integer,allocatable::ids(:)
 integer::n,k,status
 do
  read(*,*,iostat=status)n,k
  if(status/=0)exit
  allocate(x(n));read(*,*)x
  ids=numpy_argsort(x);write(*,'(*(i0,1x))')ids
  ids=numpy_argpartition(x,k);write(*,'(*(i0,1x))')ids
  deallocate(x)
 end do
end program
