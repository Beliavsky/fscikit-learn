program reference_passive_aggressive_driver
  use sklearn
  implicit none
  type(passive_aggressive_classifier) :: classifier
  type(passive_aggressive_regressor) :: reg
  real(dp), allocatable :: x(:,:),y(:),train(:,:),query(:,:),z(:,:)
  integer, allocatable :: labels(:),target(:)
  integer :: input,output,n,p,i,j
  character(1024) :: path
  character(32) :: prefix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p; allocate(x(n,p),y(n),labels(n))
  do i=1,n
    read(input,*) x(i,:),y(i),labels(i)
  end do
  close(input)
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  do j=1,8
    train=x; target=labels; classifier%shuffle=.false.; classifier%loss='hinge'; classifier%c=0.5_dp
    classifier%tol=-1; classifier%max_iter=8; classifier%n_iter_no_change=3
    classifier%fit_intercept=.true.; classifier%average=0; classifier%class_weight_balanced=.false.
    select case(j)
    case(2); classifier%loss='squared_hinge'; classifier%c=0.3_dp; classifier%fit_intercept=.false.
    case(3); classifier%class_weight_balanced=.true.
    case(4); classifier%average=1
    case(5); classifier%average=17
    case(6); classifier%tol=1.e-3_dp; classifier%max_iter=100; classifier%c=0.2_dp
    case(7)
      target=merge(-5,11,labels==minval(labels)); classifier%loss='squared_hinge'
      classifier%class_weight_balanced=.true.; classifier%average=3
    case(8); train(2,:)=0; train(5,:)=0; classifier%average=7
    end select
    query=train+0.137_dp; call classifier%fit(train,target); write(prefix,'(a,i0)') 'pac',j
    call vector(trim(prefix)//'_coef',reshape(classifier%coef_,[size(classifier%coef_)]))
    call vector(trim(prefix)//'_intercept',classifier%intercept_)
    call vector(trim(prefix)//'_classes',real(classifier%classes_,dp))
    z=classifier%decision_function(train); call vector(trim(prefix)//'_train',reshape(z,[size(z)]))
    z=classifier%decision_function(query); call vector(trim(prefix)//'_query',reshape(z,[size(z)]))
    call vector(trim(prefix)//'_predict',real(classifier%predict(query),dp))
    call vector(trim(prefix)//'_parameters',[real(classifier%n_iter_,dp),classifier%t_])
    call vector(trim(prefix)//'_score',[classifier%score(train,target)])
  end do
  do j=1,8
    train=x; reg%shuffle=.false.; reg%loss='epsilon_insensitive'; reg%c=0.5_dp; reg%epsilon=0.1_dp
    reg%tol=-1; reg%max_iter=8; reg%n_iter_no_change=3; reg%fit_intercept=.true.; reg%average=0
    select case(j)
    case(2); reg%loss='squared_epsilon_insensitive'; reg%c=0.3_dp; reg%epsilon=0.05_dp; reg%fit_intercept=.false.
    case(3); reg%average=1
    case(4); reg%average=100000
    case(5); reg%tol=1.e-3_dp; reg%max_iter=100
    case(6); train(2,:)=0; train(5,:)=0; reg%average=5
    case(7); train(2,:)=0; train(5,:)=0; reg%average=5; reg%loss='squared_epsilon_insensitive'
    case(8); reg%epsilon=10000; reg%tol=1.e-3_dp; reg%max_iter=100
    end select
    query=train+0.137_dp; call reg%fit(train,y); write(prefix,'(a,i0)') 'par',j
    call vector(trim(prefix)//'_coef',reg%coef_); call vector(trim(prefix)//'_intercept',[reg%intercept_])
    call vector(trim(prefix)//'_train',reg%predict(train)); call vector(trim(prefix)//'_query',reg%predict(query))
    call vector(trim(prefix)//'_parameters',[real(reg%n_iter_,dp),reg%t_])
    call vector(trim(prefix)//'_score',[reg%score(train,y)])
  end do
  close(output)
contains
  subroutine vector(name,v)
    character(*), intent(in) :: name
    real(dp), intent(in) :: v(:)
    write(output,'(a,*(1x,es25.16e3))') name,v
  end subroutine
end program
