program reference_cross_decomposition_driver
  use sklearn
  implicit none
  type(pls_regression) :: reg
  type(pls_canonical) :: canonical
  type(cca) :: correlation
  type(pls_svd) :: svd
  real(dp), allocatable :: x(:,:),y(:,:),z(:,:),u(:,:),xc(:,:),yc(:,:)
  integer :: n,p,q,i,j,input,output
  character(1024) :: path
  character(32) :: prefix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p,q; allocate(x(n,p),y(n,q))
  do i=1,n
    read(input,*) x(i,:),y(i,:)
  end do
  close(input)
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  do j=1,4
    reg%n_components=2; reg%scale=j/=2; reg%tol=1.e-10_dp; reg%max_iter=2000
    xc=x; yc=y
    if(j==3) reg%max_iter=1
    if(j==4) yc=2
    z=reg%fit_transform(xc,yc); u=reg%transform_y(yc)
    write(prefix,'(a,i0)') 'reg',j
    call results(trim(prefix),reg%x_weights_,reg%y_weights_,reg%x_loadings_,reg%y_loadings_, &
      reg%x_rotations_,reg%y_rotations_,reg%coef_,reg%intercept_,reg%x_scores_,reg%y_scores_, &
      z,u,reg%predict(x+0.031_dp),reg%inverse_transform(z),reg%inverse_transform_y(u),reg%n_iter_)
    call vector(trim(prefix)//'_score',[reg%score(xc,yc)])
  end do
  do j=1,4
    canonical%n_components=min(2,q); canonical%scale=j/=2
    canonical%tol=1.e-10_dp; canonical%max_iter=2000; canonical%algorithm='nipals'
    if(j==3) canonical%algorithm='svd'
    if(j==4) canonical%max_iter=1
    z=canonical%fit_transform(x,y); u=canonical%transform_y(y)
    write(prefix,'(a,i0)') 'canonical',j
    call results(trim(prefix),canonical%x_weights_,canonical%y_weights_,canonical%x_loadings_,canonical%y_loadings_, &
      canonical%x_rotations_,canonical%y_rotations_,canonical%coef_,canonical%intercept_, &
      canonical%x_scores_,canonical%y_scores_,z,u,canonical%predict(x+0.031_dp), &
      canonical%inverse_transform(z),canonical%inverse_transform_y(u),canonical%n_iter_)
  end do
  do j=1,3
    correlation%n_components=min(2,q); correlation%scale=j/=2
    correlation%tol=1.e-10_dp; correlation%max_iter=2000
    if(j==3) correlation%max_iter=1
    z=correlation%fit_transform(x,y); u=correlation%transform_y(y)
    write(prefix,'(a,i0)') 'cca',j
    call results(trim(prefix),correlation%x_weights_,correlation%y_weights_,correlation%x_loadings_, &
      correlation%y_loadings_,correlation%x_rotations_,correlation%y_rotations_,correlation%coef_, &
      correlation%intercept_,correlation%x_scores_,correlation%y_scores_,z,u,correlation%predict(x+0.031_dp), &
      correlation%inverse_transform(z),correlation%inverse_transform_y(u),correlation%n_iter_)
  end do
  do j=1,2
    svd%scale=j==1; svd%n_components=min(2,q); z=svd%fit_transform(x,y); u=svd%transform_y(y)
    write(prefix,'(a,i0)') 'svd',j
    call matrix(trim(prefix)//'_x_weights',svd%x_weights_); call matrix(trim(prefix)//'_y_weights',svd%y_weights_)
    call matrix(trim(prefix)//'_x_transform',z); call matrix(trim(prefix)//'_y_transform',u)
    call matrix(trim(prefix)//'_query',svd%transform(x+0.031_dp))
  end do
  close(output)
contains
  subroutine results(name,xw,yw,xl,yl,xr,yr,coef,intercept,xs,ys,xt,yt,pred,xi,yi,iterations)
    character(*), intent(in) :: name
    real(dp), intent(in) :: xw(:,:),yw(:,:),xl(:,:),yl(:,:),xr(:,:),yr(:,:),coef(:,:),intercept(:)
    real(dp), intent(in) :: xs(:,:),ys(:,:),xt(:,:),yt(:,:),pred(:,:),xi(:,:),yi(:,:)
    integer, intent(in) :: iterations(:)
    call matrix(name//'_x_weights',xw); call matrix(name//'_y_weights',yw)
    call matrix(name//'_x_loadings',xl); call matrix(name//'_y_loadings',yl)
    call matrix(name//'_x_rotations',xr); call matrix(name//'_y_rotations',yr)
    call matrix(name//'_coef',coef); call vector(name//'_intercept',intercept)
    call matrix(name//'_x_scores',xs); call matrix(name//'_y_scores',ys)
    call matrix(name//'_x_transform',xt); call matrix(name//'_y_transform',yt)
    call matrix(name//'_predict',pred); call matrix(name//'_x_inverse',xi); call matrix(name//'_y_inverse',yi)
    call vector(name//'_iterations',real(iterations,dp))
  end subroutine
  subroutine vector(name,values)
    character(*), intent(in) :: name
    real(dp), intent(in) :: values(:)
    write(output,'(a,1x,*(es25.16e3,1x))') name,values
  end subroutine
  subroutine matrix(name,values)
    character(*), intent(in) :: name
    real(dp), intent(in) :: values(:,:)
    call vector(name,reshape(values,[size(values)]))
  end subroutine
end program
