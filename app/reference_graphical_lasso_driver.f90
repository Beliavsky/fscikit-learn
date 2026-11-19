program reference_graphical_lasso_driver
  use sklearn
  implicit none
  type(graphical_lasso) :: model
  type(empirical_covariance) :: empirical
  real(dp), allocatable :: x(:,:),train(:,:),query(:,:)
  integer :: input,output,n,p,i,j
  character(1024) :: path
  character(32) :: prefix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p; allocate(x(n,p))
  do i=1,n
    read(input,*) x(i,:)
  end do
  close(input)
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  do j=1,8
    train=x; query=x+0.137_dp; model%alpha=0.08_dp; model%covariance='empirical'; model%assume_centered=.false.
    model%max_iter=300; model%tol=1.e-9_dp; model%enet_tol=1.e-11_dp
    select case(j)
    case(2); model%alpha=0.2_dp
    case(3); model%alpha=10
    case(4); model%alpha=0
    case(5); model%assume_centered=.true.
    case(6)
      call empirical%fit(x); train=empirical%covariance_; model%covariance='precomputed'
    case(7)
      train(:,p)=x(:,1)+0.3_dp*x(:,2); query=train+0.137_dp; model%alpha=0.2_dp
    case(8); model%max_iter=2; model%tol=1.e-14_dp
    end select
    call model%fit(train)
    if(j/=8.and..not.model%converged_) error stop 'GraphicalLasso reference fit did not converge'
    write(prefix,'(a,i0)') 'graph',j
    call vector(trim(prefix)//'_location',model%location_)
    call vector(trim(prefix)//'_covariance',reshape(model%covariance_,[p*p]))
    call vector(trim(prefix)//'_precision',reshape(model%precision_,[p*p]))
    call vector(trim(prefix)//'_cost',[model%objective_,model%dual_gap_])
    call vector(trim(prefix)//'_mahalanobis',model%mahalanobis(query))
    call vector(trim(prefix)//'_score',[model%score(query)])
  end do
  close(output)
contains
  subroutine vector(name,v)
    character(*), intent(in) :: name
    real(dp), intent(in) :: v(:)
    write(output,'(a,*(1x,es25.16e3))') name,v
  end subroutine
end program
