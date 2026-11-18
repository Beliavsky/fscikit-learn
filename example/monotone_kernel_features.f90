program monotone_kernel_features
  use sklearn
  implicit none
  type(isotonic_regression) :: monotone
  type(rbf_sampler) :: fourier
  type(nystroem) :: landmarks
  real(dp) :: x(40,2),y(40),t(40)
  real(dp), allocatable :: fitted(:),z(:,:),approximate(:,:),exact(:,:)
  integer :: i
  do i=1,40
    t(i)=real(i,dp)/10; y(i)=t(i)+0.4_dp*sin(3*i*1._dp)
    x(i,:)=[sin(t(i)),cos(t(i))]
  end do
  fitted=monotone%fit_transform(t,y)
  print *, 'Monotone training R2: ',monotone%score(t,y)
  fourier%n_components=2000; fourier%random_state=42
  z=fourier%fit_transform(x); approximate=matmul(z,transpose(z)); exact=rbf_kernel(x,x,1._dp)
  print *, 'Fourier kernel maximum error: ',maxval(abs(approximate-exact))
  landmarks%n_components=40; landmarks%gamma=1; landmarks%random_state=42
  z=landmarks%fit_transform(x); approximate=matmul(z,transpose(z))
  print *, 'Full-landmark kernel maximum error: ',maxval(abs(approximate-exact))
end program
