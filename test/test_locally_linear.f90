program test_locally_linear
  use sklearn
  use sklearn_common, only: require
  implicit none
  type(locally_linear_embedding) :: model,changed
  real(dp) :: x(40,3),t,gram(2,2),objective
  real(dp), allocatable :: z(:,:),again(:,:),reconstructed(:,:),query(:,:),actual(:,:)
  integer :: i
  do i=1,40
    t=real(i,dp)/10; x(i,:)=[sin(t),cos(t),0.2_dp*t+0.05_dp*sin(7*t)]
  end do
  model%n_neighbors=8; z=model%fit_transform(x)
  call require(maxval(abs(sum(model%weights_,dim=2)-1))<1.e-12_dp,'LLE affine weights sum to one')
  gram=matmul(transpose(z),z); gram(1,1)=gram(1,1)-1; gram(2,2)=gram(2,2)-1
  call require(maxval(abs(gram))<1.e-12_dp,'LLE embedding orthonormality')
  call require(maxval(abs(sum(z,dim=1)))<1.e-8_dp,'LLE skips constant mode')
  reconstructed=z
  do i=1,40
    reconstructed(i,:)=z(i,:)-matmul(model%weights_(i,:),z(model%neighbors_(i,:),:))
    call require(all(model%neighbors_(i,:)/=i),'LLE excludes each training sample from its neighborhood')
  end do
  objective=sum(reconstructed**2)
  call require(abs(objective-model%reconstruction_error_)<1.e-10_dp,'LLE eigenvalue reconstruction objective')
  changed=model; again=changed%fit_transform(3*x+100)
  call require(maxval(abs(matmul(z,transpose(z))-matmul(again,transpose(again))))<1.e-8_dp, &
    'LLE invariant under translation and uniform scaling')
  query=x+0.017_dp; actual=model%transform(query)
  again=changed%transform(3*query+100)
  call require(maxval(abs(matmul(actual,transpose(actual))-matmul(again,transpose(again))))<1.e-8_dp, &
    'LLE out-of-sample geometry invariant under scaling')
  model%method='ltsa'; model%n_components=1; z=model%fit_transform(x)
  call require(abs(sum(z*z)-1)<1.e-12_dp,'LTSA unit embedding norm')
  call require(model%reconstruction_error_>=0,'LTSA nonnegative alignment objective')
  print *, 'locally linear embedding: PASS'
end program
