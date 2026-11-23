program fixed_component_factorization
  use sklearn, only: dp,non_negative_factorization
  implicit none
  real(dp) :: x(3,3),components(2,3)
  real(dp), allocatable :: w(:,:),h(:,:)
  integer :: iterations,i
  components(1,:)=[1._dp,0._dp,1._dp]
  components(2,:)=[0._dp,1._dp,1._dp]
  x(1,:)=[2._dp,1._dp,3._dp]
  x(2,:)=[1._dp,3._dp,4._dp]
  x(3,:)=[4._dp,2._dp,6._dp]
  call non_negative_factorization(x,w,h,iterations,h0=components,update_h=.false.,tol=1.e-10_dp)
  print *, 'Codes for a fixed component dictionary:'
  do i=1,size(w,1)
    print '(2f12.6)',w(i,:)
  end do
  print '(a,es12.3)', 'Reconstruction error: ',sqrt(sum((x-matmul(w,h))**2))
end program
