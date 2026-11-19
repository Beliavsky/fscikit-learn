program reference_theil_sen_driver
  use sklearn
  implicit none
  type(theil_sen_regressor) :: model
  real(dp), allocatable :: x(:,:),y(:),train(:,:),query(:,:)
  integer :: input,output,n,p,i,j
  character(1024) :: path
  character(32) :: prefix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p; allocate(x(n,p),y(n))
  do i=1,n
    read(input,*) x(i,:),y(i)
  end do
  close(input)
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  do j=1,8
    train=x; model%fit_intercept=.true.; model%n_subsamples=0; model%max_subpopulation=50000
    model%max_iter=2000; model%tol=1.e-8_dp; model%random_state=31
    select case(j)
    case(2); model%fit_intercept=.false.
    case(3); model%n_subsamples=p+2
    case(4); model%n_subsamples=n
    case(5); train(:,p)=train(:,1); model%n_subsamples=n
    case(6); model%max_iter=2; model%tol=1.e-14_dp
    case(7); train=x(:,1:1); model%fit_intercept=.false.
    case(8); model%max_subpopulation=17
    end select
    query=train+0.137_dp; call model%fit(train,y)
    if(j/=6.and..not.model%converged_) error stop 'TheilSen reference fit did not converge'
    write(prefix,'(a,i0)') 'theil',j
    call vector(trim(prefix)//'_coef',model%coef_)
    call vector(trim(prefix)//'_parameters',[model%intercept_,model%breakdown_,real(model%n_subpopulation_,dp), &
      real(model%n_subsamples_,dp),real(model%n_iter_,dp)])
    call vector(trim(prefix)//'_train',model%predict(train))
    call vector(trim(prefix)//'_query',model%predict(query))
    call vector(trim(prefix)//'_score',[model%score(train,y)])
    if(j==8) call vector('sampled_subsets',reshape(real(model%subsets_,dp),[size(model%subsets_)]))
  end do
  close(output)
contains
  subroutine vector(name,v)
    character(*), intent(in) :: name
    real(dp), intent(in) :: v(:)
    write(output,'(a,*(1x,es25.16e3))') name,v
  end subroutine
end program
