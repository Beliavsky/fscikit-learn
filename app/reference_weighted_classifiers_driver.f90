program reference_weighted_classifiers_driver
  use sklearn, only: dp,gaussian_nb,logistic_regression
  implicit none
  type(gaussian_nb) :: nb
  type(logistic_regression) :: lr
  integer :: n,p,m,nc,lc,batches,weighted,prior,do_log,intercept,refit,i,stage,first,last,unit,out
  integer, allocatable :: y(:),classes(:),cuts(:)
  real(dp), allocatable :: x(:,:),query(:,:),weights(:),priors(:),prob(:,:),logp(:,:),values(:,:)
  character(1024) :: input,output
  call get_command_argument(1,input); call get_command_argument(2,output)
  open(newunit=unit,file=trim(input),status='old')
  read(unit,*) n,p,m,nc,lc,batches,weighted,prior,do_log,intercept,refit,lr%c,nb%var_smoothing
  allocate(x(n,p),query(m,p),weights(n),priors(nc),classes(lc),y(n),cuts(batches))
  read(unit,*) y; read(unit,*) classes; read(unit,*) weights; read(unit,*) priors
  if(batches>0) read(unit,*) cuts
  do i=1,n
    read(unit,*) x(i,:)
  end do
  do i=1,m
    read(unit,*) query(i,:)
  end do
  close(unit)
  if(prior==1) nb%priors=priors
  lr%fit_intercept=intercept==1; lr%tol=1.e-10_dp; lr%max_iter=500
  open(newunit=out,file=trim(output),status='replace')
  if(batches==0) then
    call batch_fit(); call write_nb()
  else
    first=1
    do stage=1,batches
      last=cuts(stage)
      if(stage==1) then
        if(weighted==1) then
          call nb%partial_fit(x(first:last,:),y(first:last),classes,weights(first:last))
        else
          call nb%partial_fit(x(first:last,:),y(first:last),classes)
        end if
      else
        if(weighted==1) then
          call nb%partial_fit(x(first:last,:),y(first:last),sample_weight=weights(first:last))
        else
          call nb%partial_fit(x(first:last,:),y(first:last))
        end if
      end if
      call write_nb(); first=last+1
    end do
    if(refit==1) then
      call batch_fit(); call write_nb()
    end if
  end if
  if(do_log==1) then
    if(weighted==1) then
      call lr%fit(x,y,weights)
    else
      call lr%fit(x,y)
    end if
    write(out,*) size(lr%coef_,1); call matrix(lr%coef_)
    write(out,'(*(es25.16e3,1x))') lr%intercept_
    values=lr%decision_function(query); call matrix(values)
    prob=lr%predict_proba(query); call matrix(prob); write(out,*) lr%predict(query)
    write(out,*) lr%n_iter_,lr%converged_
  end if
  write(out,*) y; call matrix(x); call matrix(query)
  write(out,'(*(es25.16e3,1x))') weights
  close(out)
contains
  subroutine batch_fit()
    if(weighted==1) then
      call nb%fit(x,y,weights)
    else
      call nb%fit(x,y)
    end if
  end subroutine
  subroutine matrix(a)
    real(dp), intent(in) :: a(:,:)
    integer :: row
    do row=1,size(a,1)
      write(out,'(*(es25.16e3,1x))') a(row,:)
    end do
  end subroutine
  subroutine write_nb()
    write(out,*) size(nb%classes_),nb%n_features_in_; write(out,*) nb%classes_
    write(out,'(*(es25.16e3,1x))') nb%class_count_
    call matrix(nb%theta_); call matrix(nb%var_)
    write(out,'(*(es25.16e3,1x))') nb%class_prior_
    write(out,'(es25.16e3)') nb%epsilon_
    prob=nb%predict_proba(query); logp=nb%predict_log_proba(query)
    call matrix(prob); call matrix(logp); write(out,*) nb%predict(query)
  end subroutine
end program
