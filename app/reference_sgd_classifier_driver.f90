program reference_sgd_classifier_driver
  use sklearn, only: dp,sgd_classifier
  implicit none
  type(sgd_classifier) :: model
  real(dp), allocatable :: x(:,:),w(:),cw(:)
  integer, allocatable :: binary(:),multi(:),y(:)
  integer :: input,output,n,p,i,j
  character(512) :: path
  character(24) :: prefix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p; allocate(x(n,p),w(n),binary(n),multi(n))
  do i=1,n
    read(input,*) x(i,:),w(i),binary(i),multi(i)
  end do
  close(input); call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  do j=1,12
    model=sgd_classifier(); model%max_iter=7; model%tol=-1; model%eta0=.03_dp; model%alpha=.01_dp
    y=binary; cw=[1._dp,1._dp]
    select case(j)
    case(2); y=multi; cw=[1._dp,1._dp,1._dp]
    case(3); model%loss='log_loss'
    case(4); model%loss='log_loss'; y=multi; cw=[.5_dp,2._dp,1.5_dp]
    case(5); model%loss='modified_huber'; model%average=1
    case(6); model%loss='modified_huber'; model%average=10; y=multi; cw=[1._dp,1._dp,1._dp]
    case(7); model%loss='squared_hinge'; model%class_weight_balanced=.true.; y=multi
    case(8); model%loss='perceptron'; model%penalty='none'; model%learning_rate='constant'
    case(9); model%loss='log_loss'; model%average=1; model%fit_intercept=.false.; y=multi
    case(10); cw=[.7_dp,2._dp]; model%fit_intercept=.false.
    case(11); model%max_iter=300; model%tol=1.e-3_dp; y=multi; cw=[1._dp,1._dp,1._dp]
    case(12); model%loss='log_loss'; model%average=10; model%fit_intercept=.false.
    end select
    if(j==9) then
      call model%partial_fit(x(:9,:),y(:9),classes=[-4,7,21],sample_weight=w(:9))
      call model%partial_fit(x(10:10,:),y(10:10),sample_weight=w(10:10))
      call model%partial_fit(x(11:,:),y(11:),sample_weight=w(11:))
    else if(j==7) then
      call model%fit(x,y,sample_weight=w)
    else
      call model%fit(x,y,sample_weight=w,class_weight=cw)
      if(j==12) call model%partial_fit(x(:9,:),y(:9),sample_weight=w(:9))
    end if
    write(prefix,'(a,i0)') 'case',j
    call matrix('_coef',model%coef_); call vector('_intercept',model%intercept_)
    call vector('_classes',real(model%classes_,dp)); call matrix('_train',model%decision_function(x))
    call matrix('_query',model%decision_function(x+.137_dp)); call vector('_predict',real(model%predict(x+.137_dp),dp))
    call vector('_score',[model%score(x,y)]); call vector('_clock',[real(model%n_iter_,dp),model%t_])
    if(model%loss=='log_loss'.or.model%loss=='modified_huber') call matrix('_proba',model%predict_proba(x+.137_dp))
  end do
  close(output)
contains
  subroutine vector(suffix,values)
    character(*), intent(in) :: suffix
    real(dp), intent(in) :: values(:)
    write(output,'(a,*(1x,es25.16e3))') trim(prefix)//suffix,values
  end subroutine
  subroutine matrix(suffix,values)
    character(*), intent(in) :: suffix
    real(dp), intent(in) :: values(:,:)
    call vector(suffix,reshape(values,[size(values)]))
  end subroutine
end program
