program reference_sgd_one_class_driver
  use sklearn_sgd_one_class, only: sgd_one_class_svm
  use sklearn_common, only: dp
  implicit none
  type(sgd_one_class_svm) :: model
  real(dp), allocatable :: x(:,:),w(:)
  integer, allocatable :: fit_labels(:)
  integer :: n,p,i,input,output,shuf,intercept,operation,part,first,last
  character(512) :: path
  character(24) :: prefix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p,operation,shuf,intercept
  read(input,*) model%learning_rate,model%nu,model%eta0,model%power_t,model%tol
  read(input,*) model%max_iter,model%n_iter_no_change,model%average,model%random_state
  allocate(x(n,p),w(n))
  do i=1,n
    read(input,*) x(i,:),w(i)
  end do
  close(input); model%shuffle=shuf==1; model%fit_intercept=intercept==1
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  if(operation==2) then
    do part=1,3
      first=1+(part-1)*n/3; last=part*n/3
      call model%partial_fit(x(first:last,:),w(first:last)); write(prefix,'(a,i0)') 'partial',part
      call emit()
    end do
  else
    if(operation==6) then
      fit_labels=model%fit_predict(x,w)
    else if(operation==4) then
      call model%fit(x,w,coef_init=spread(.1_dp,1,p),offset_init=.2_dp)
    else
      call model%fit(x,w)
    end if
    prefix='fit'; call emit()
    if(operation==3) then
      model%warm_start=.true.; call model%fit(x+.02_dp,w); prefix='warm'; call emit()
    else if(operation==5) then
      call model%partial_fit(x,w); prefix='after'; call emit()
    end if
  end if
  close(output)
contains
  subroutine emit()
    call vector('_coef',model%coef_); call vector('_offset',[model%offset_])
    call vector('_decision',model%decision_function(x+.137_dp)); call vector('_samples',model%score_samples(x+.137_dp))
    call vector('_predict',real(model%predict(x+.137_dp),dp))
    call vector('_state',[model%t_,real(model%n_iter_,dp),real(model%n_features_in_,dp)])
  end subroutine
  subroutine vector(suffix,a)
    character(*), intent(in) :: suffix
    real(dp), intent(in) :: a(:)
    write(output,'(a,1x,*(es25.16e3,1x))') trim(prefix)//suffix,a
  end subroutine
end program
