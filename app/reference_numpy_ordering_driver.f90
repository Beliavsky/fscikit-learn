program reference_numpy_ordering_driver
 use sklearn_common,only:dp
 use sklearn_numpy_ordering,only:numpy_argsort,numpy_argpartition
 implicit none
 real(dp),allocatable::x(:)
 integer,allocatable::ids(:)
 integer::cases,n,k,i
 read(*,*) cases
 do i=1,cases
  read(*,*) n,k
  if(allocated(x)) deallocate(x)
  allocate(x(n));read(*,*) x
  ids=numpy_argsort(x);write(*,'(*(i0,1x))') ids
  ids=numpy_argpartition(x,k);write(*,'(*(i0,1x))') ids
 end do
end program
