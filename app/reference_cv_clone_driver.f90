program reference_cv_clone_driver
  use sklearn_common,only:dp
  use sklearn_linear_model,only:linear_regression,huber_regressor,tweedie_regressor,poisson_regressor,gamma_regressor
  use sklearn_model_selection,only:cross_val_score
  implicit none
  class(linear_regression),allocatable::model
  real(dp),allocatable::x(:,:),y(:),other(:),w(:),scores(:),before(:)
  integer::n,p,family,iterations,warm,prefit,weighted,i,unit,out,learned_before
  real(dp)::intercept_before,change
  character(1024)::path
  call get_command_argument(1,path);open(newunit=unit,file=trim(path),status='old')
  read(unit,*)n,p,family,iterations,warm,prefit,weighted;allocate(x(n,p),y(n),other(n),w(n))
  do i=1,n
    read(unit,*)x(i,:),y(i),other(i),w(i)
  end do
  close(unit)
  select case(family)
  case(0);allocate(huber_regressor::model)
  case(1);allocate(poisson_regressor::model)
  case(2);allocate(gamma_regressor::model)
  case(3);allocate(tweedie_regressor::model)
  end select
  select type(model)
  type is(huber_regressor)
    model%max_iter=iterations;model%tol=1.e-7_dp;model%warm_start=warm==1
  class is(tweedie_regressor)
    model%max_iter=iterations;model%tol=1.e-8_dp;model%warm_start=warm==1;model%alpha=.1_dp
    if(family==3)model%power=1.5_dp
  end select
  if(prefit==1)call model%fit(x,other)
  learned_before=merge(1,0,allocated(model%coef_));intercept_before=model%intercept_
  if(allocated(model%coef_))before=model%coef_
  select case(weighted)
  case(0);scores=cross_val_score(model,x,y,cv=4)
  case(1);scores=cross_val_score(model,x,y,cv=4,sample_weight=w)
  case(2);scores=cross_val_score(model,x,y,cv=4,sample_weight=w(1))
  end select
  change=abs(model%intercept_-intercept_before)
  if(allocated(before))change=max(change,maxval(abs(model%coef_-before)))
  call get_command_argument(2,path);open(newunit=out,file=trim(path),status='replace')
  write(out,'(*(es26.17e3,1x))')scores
  write(out,'(*(es26.17e3,1x))')change,real(learned_before,dp),real(merge(1,0,allocated(model%coef_)),dp)
  close(out)
end program
