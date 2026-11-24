program reference_integer_sets_driver
 use sklearn_integer_sets,only:integer_set_order
 implicit none
 integer::count,i,n
 integer,allocatable::keys(:),values(:)
 read(*,*) count
 do i=1,count
  read(*,*) n;allocate(keys(n));read(*,*) keys
  values=integer_set_order(keys);print *, i,values;deallocate(keys)
 end do
end program
