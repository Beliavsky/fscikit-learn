program paired_latent_components
  use sklearn
  implicit none
  type(pls_regression) :: model
  type(cca) :: correlation
  real(dp) :: x(100,4),y(100,2),t,rho
  real(dp), allocatable :: z(:,:),u(:,:)
  integer :: i
  do i=1,100
    t=real(i,dp)/100
    x(i,:)=[sin(17*t),cos(23*t),sin(41*t),t]
    y(i,:)=[2*x(i,1)-x(i,2)+3*x(i,4)+1,x(i,2)+2*x(i,3)-2]
  end do
  model%n_components=4; z=model%fit_transform(x,y)
  print *, 'Multiresponse PLS training R2: ',model%score(x,y)
  call correlation%fit(x,y); z=correlation%transform(x); u=correlation%transform_y(y)
  rho=sum(z(:,1)*u(:,1))/sqrt(sum(z(:,1)**2)*sum(u(:,1)**2))
  print *, 'First canonical correlation: ',rho
end program
