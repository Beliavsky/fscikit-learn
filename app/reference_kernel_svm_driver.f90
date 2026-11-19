program reference_kernel_svm_driver
  use sklearn
  implicit none
  type(svr) :: reg
  type(nu_svr) :: nureg
  type(one_class_svm) :: novelty
  real(dp), allocatable :: x(:,:),y(:),w(:),train(:,:),query(:,:),target(:),beta(:),g(:,:),gq(:,:)
  integer :: n,p,i,j,input,output
  character(1024) :: path
  character(32) :: prefix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p; allocate(x(n,p),y(n),w(n),beta(n))
  do i=1,n
    read(input,*) x(i,:),y(i),w(i)
  end do
  close(input); g=rbf_kernel(x,x,0.7_dp); gq=rbf_kernel(x+0.137_dp,x,0.7_dp)
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  do j=1,8
    reg%kernel='rbf'; reg%gamma=-1; reg%gamma_rule='scale'; reg%c=2; reg%epsilon=0.1_dp
    reg%tol=1.e-8_dp; reg%max_iter=200000; reg%degree=2; reg%coef0=0.4_dp
    train=x; query=x+0.137_dp; target=y
    select case(j)
    case(2); reg%gamma_rule='auto'; reg%c=1; reg%epsilon=0.2_dp
    case(3); reg%gamma=0.7_dp; reg%epsilon=0.05_dp
    case(4); reg%kernel='linear'; reg%c=0.7_dp
    case(5); reg%kernel='poly'; reg%gamma=0.3_dp; reg%c=0.7_dp
    case(6); reg%kernel='precomputed'; train=g; query=gq
    case(7); target=2
    case(8); reg%epsilon=1000
    end select
    if(j==3) then
      call reg%fit(train,target,w)
    else
      call reg%fit(train,target)
    end if
    if(.not.reg%converged_) error stop 'SVR reference fit did not converge'
    write(prefix,'(a,i0)') 'svr',j
    call vector(trim(prefix)//'_train',reg%predict(train)); call vector(trim(prefix)//'_query',reg%predict(query))
    call vector(trim(prefix)//'_parameters',[reg%intercept_,reg%gamma_])
    call vector(trim(prefix)//'_score',[reg%score(train,target)])
    if(j/=4.and.j/=5) then
      beta=0; beta(reg%support_)=reg%dual_coef_; call vector(trim(prefix)//'_dual',beta)
    end if
    if(j==4) call vector(trim(prefix)//'_coef',reg%coef_)
  end do
  do j=1,7
    nureg%kernel='rbf'; nureg%gamma=-1; nureg%gamma_rule='scale'; nureg%c=2; nureg%nu=0.4_dp
    nureg%tol=1.e-8_dp; nureg%max_iter=200000; nureg%degree=2; nureg%coef0=0.4_dp
    train=x; query=x+0.137_dp; target=y
    select case(j)
    case(2); nureg%gamma_rule='auto'; nureg%c=1; nureg%nu=0.7_dp
    case(3); nureg%gamma=0.7_dp; nureg%nu=0.3_dp
    case(4); nureg%kernel='linear'; nureg%c=0.7_dp; nureg%nu=0.5_dp
    case(5); nureg%kernel='poly'; nureg%gamma=0.3_dp; nureg%c=0.7_dp; nureg%nu=0.8_dp
    case(6); nureg%kernel='precomputed'; nureg%nu=0.5_dp; train=g; query=gq
    case(7); target=2
    end select
    if(j==3) then
      call nureg%fit(train,target,w)
    else
      call nureg%fit(train,target)
    end if
    if(.not.nureg%converged_) error stop 'NuSVR reference fit did not converge'
    write(prefix,'(a,i0)') 'nu',j
    call vector(trim(prefix)//'_train',nureg%predict(train)); call vector(trim(prefix)//'_query',nureg%predict(query))
    call vector(trim(prefix)//'_parameters',[nureg%intercept_,nureg%gamma_])
    call vector(trim(prefix)//'_score',[nureg%score(train,target)])
    if(j/=4.and.j/=5) then
      beta=0; beta(nureg%support_)=nureg%dual_coef_; call vector(trim(prefix)//'_dual',beta)
    end if
    if(j==4) call vector(trim(prefix)//'_coef',nureg%coef_)
  end do
  do j=1,7
    novelty%kernel='rbf'; novelty%gamma=-1; novelty%gamma_rule='scale'; novelty%nu=0.3_dp
    novelty%tol=1.e-8_dp; novelty%max_iter=200000; novelty%degree=2; novelty%coef0=0.4_dp
    train=x; query=x+0.137_dp
    select case(j)
    case(2); novelty%gamma_rule='auto'; novelty%nu=0.5_dp
    case(3); novelty%gamma=0.7_dp; novelty%nu=0.4_dp
    case(4)
      novelty%kernel='linear'; novelty%nu=0.23_dp; train=abs(x)+1; query=abs(x+0.137_dp)+1
    case(5); novelty%kernel='poly'; novelty%gamma=0.3_dp
    case(6); novelty%kernel='precomputed'; novelty%nu=0.5_dp; train=g; query=gq
    case(7); novelty%gamma=0; novelty%nu=0.4_dp
    end select
    if(j==3) then
      call novelty%fit(train,w)
    else
      call novelty%fit(train)
    end if
    if(.not.novelty%converged_) error stop 'OneClassSVM reference fit did not converge'
    write(prefix,'(a,i0)') 'one',j
    call vector(trim(prefix)//'_train',novelty%decision_function(train))
    call vector(trim(prefix)//'_query',novelty%decision_function(query))
    call vector(trim(prefix)//'_raw',novelty%score_samples(query))
    call vector(trim(prefix)//'_parameters',[novelty%intercept_,novelty%offset_,novelty%gamma_])
    if(j/=4.and.j/=5.and.j/=7) then
      beta=0; beta(novelty%support_)=novelty%dual_coef_; call vector(trim(prefix)//'_dual',beta)
    end if
    if(j==4) then
      call vector(trim(prefix)//'_coef',novelty%coef_)
      beta=0; beta(novelty%support_)=novelty%dual_coef_; call vector('one4_dual_meta',beta)
    end if
  end do
  close(output)
contains
  subroutine vector(name,values)
    character(*), intent(in) :: name
    real(dp), intent(in) :: values(:)
    write(output,'(a,1x,*(es25.16e3,1x))') name,values
  end subroutine
end program
