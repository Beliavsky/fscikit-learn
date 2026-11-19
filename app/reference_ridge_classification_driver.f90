program reference_ridge_classification_driver
  use sklearn
  implicit none
  type(ridge_classifier) :: model
  type(ridge_classifier_cv) :: cv
  real(dp), allocatable :: x(:,:),w(:),weights(:),train(:,:),query(:,:),z(:,:),cw(:)
  integer, allocatable :: binary(:),multi(:),target(:)
  integer :: n,p,i,j,input,output
  character(1024) :: path
  character(32) :: prefix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p; allocate(x(n,p),w(n),binary(n),multi(n))
  do i=1,n
    read(input,*) x(i,:),w(i),binary(i),multi(i)
  end do
  close(input)
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  do j=1,10
    model=ridge_classifier(); model%alpha=.7_dp; train=x; target=multi; weights=1+0*w
    select case(j)
    case(2); target=binary
    case(3); model%fit_intercept=.false.
    case(4); weights=w
    case(5); weights=w; model%class_weight_balanced=.true.
    case(6); weights=w
    case(7); target=binary; weights=w; model%fit_intercept=.false.
    case(8)
      model%alpha=0
      if(n<=p) model%fit_intercept=.false.
    case(9); train(:,p)=train(:,1); model%alpha=.03_dp
    case(10); target=binary; model%class_weight_balanced=.true.; model%alpha=10
    end select
    if(j==6) then
      call model%fit(train,target,sample_weight=weights,class_weight=[.5_dp,2._dp,1.5_dp])
    else
      call model%fit(train,target,sample_weight=weights)
    end if
    query=train+.137_dp; write(prefix,'(a,i0)') 'ridge',j
    call matrix('_coef',model%coef_); call vector('_intercept',model%intercept_)
    call vector('_classes',real(model%classes_,dp)); call matrix('_train',model%decision_function(train))
    call matrix('_query',model%decision_function(query)); call vector('_predict',real(model%predict(query),dp))
    call vector('_score',[model%score(train,target)])
  end do
  do j=1,7
    cv=ridge_classifier_cv(); cv%alphas=[.03_dp,.3_dp,3._dp,30._dp]; cv%store_cv_results=.true.
    target=multi; weights=1+0*w
    select case(j)
    case(2); target=binary
    case(3); cv%fit_intercept=.false.
    case(4); weights=w
    case(5); weights=w; cv%class_weight_balanced=.true.
    case(6); weights=w
    case(7); target=binary; cv%fit_intercept=.false.; weights=w
    end select
    if(j==6) then
      call cv%fit(x,target,sample_weight=weights,class_weight=[.5_dp,2._dp,1.5_dp])
    else
      call cv%fit(x,target,sample_weight=weights)
    end if
    write(prefix,'(a,i0)') 'ridgecv',j
    call matrix('_coef',cv%coef_); call vector('_intercept',cv%intercept_)
    call vector('_classes',real(cv%classes_,dp)); call matrix('_train',cv%decision_function(x))
    call matrix('_query',cv%decision_function(x+.137_dp)); call vector('_predict',real(cv%predict(x+.137_dp),dp))
    call vector('_score',[cv%score(x,target)]); call vector('_selected',[cv%alpha_,cv%best_score_])
    call vector('_cv_scores',cv%cv_scores_); call vector('_cv_results',reshape(cv%cv_results_,[size(cv%cv_results_)]))
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
