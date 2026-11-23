program test_non_negative_factorization
  use sklearn, only: dp,non_negative_factorization
  use sklearn_common, only: require
  implicit none
  real(dp) :: x(4,3),h0(1,3),w0(4,1)
  real(dp), allocatable :: w(:,:),h(:,:),z(:,:),g(:,:)
  integer :: i,it,jt
  do i=1,4
    x(i,:)=real(i,dp)*[1._dp,2._dp,3._dp]
  end do
  h0(1,:)=[1._dp,2._dp,3._dp];w0=1
  call non_negative_factorization(x,w,h,it,init='custom',w0=w0,h0=h0)
  call require(size(h,1)==1.and.it<=3,'custom automatic rank and stopping')
  call require(maxval(abs(matmul(w,h)-x))<1.e-12_dp,'rank-one factorization')
  call require(all(w0==1).and.all(h0(1,:)==[1._dp,2._dp,3._dp]),'initializer ownership')
  call non_negative_factorization(x,w,h,it,h0=h0,update_h=.false.)
  call require(all(h==h0).and.maxval(abs(w(:,1)-[1._dp,2._dp,3._dp,4._dp]))<1.e-12_dp, &
    'fixed H coordinate solution')
  call non_negative_factorization(x,z,g,jt,h0=h0,w0=10*w0,update_h=.false.)
  call require(all(w==z).and.it==jt,'fixed H ignores supplied W')
  call non_negative_factorization(x,w,h,it,h0=h0,update_h=.false.,solver='mu',max_iter=30,tol=0._dp)
  call require(it==30.and.all(h==h0),'fixed H MU iteration count and ownership')
  call require(maxval(abs(matmul(w,h)-x))<1.e-12_dp,'fixed H MU solution')
  call non_negative_factorization(x,w,h,it,h0=h0,update_h=.false.,alpha_w=.1_dp,l1_ratio=1._dp)
  call require(maxval(abs(w(:,1)-([1._dp,2._dp,3._dp,4._dp]-.3_dp/14)))<1.e-12_dp, &
    'fixed H L1 regularization feature scaling')
  call non_negative_factorization(x,w,h,it,n_components=2,init='random',random_state=42,max_iter=3,tol=0._dp)
  call non_negative_factorization(x,z,g,jt,n_components=2,init='random',random_state=42,max_iter=3,tol=0._dp)
  call require(all(w==z).and.all(h==g).and.it==jt,'integer seed reproducibility')
  print *, 'non_negative_factorization: PASS'
end program
