program reference_glm_controls_driver
  use sklearn
  implicit none
  class(tweedie_regressor),allocatable::model
  real(dp),allocatable::x(:,:),y(:),weight(:)
  integer::n,p,family,intercept,mode,steps,i,j,input,output,cap
  real(dp)::power,alpha,tol
  character(16)::link
  character(1024)::path
  call get_command_argument(1,path);open(newunit=input,file=trim(path),status='old')
  read(input,*)n,p,family,power,link,alpha,tol,cap,intercept,mode,steps
  select case(family)
  case(0);allocate(tweedie_regressor::model)
  case(1);allocate(poisson_regressor::model)
  case(2);allocate(gamma_regressor::model)
  end select
  model%solver='native-newton'
  model%power=power;model%link=link;model%alpha=alpha;model%tol=tol;model%max_iter=cap
  model%fit_intercept=intercept/=0;model%warm_start=steps>1
  allocate(x(n,p),y(n),weight(n))
  do i=1,n
    read(input,*)x(i,:),y(i),weight(i)
  end do
  close(input)
  call get_command_argument(2,path);open(newunit=output,file=trim(path),status='replace')
  do j=1,steps
    select case(mode)
    case(0);call model%fit(x,y)
    case(1);call model%fit(x,y,weight)
    case(2);call model%fit(x,y,weight(1))
    case(3);call model%fit(x,reshape(y,[n,1]),weight)
    case(4);call model%fit(x,reshape(y,[n,1]),weight(1))
    end select
    write(output,'(*(es25.16e3,1x))')model%coef_,model%intercept_
    write(output,'(*(es25.16e3,1x))')model%predict(x+0.017_dp)
    if(mode==0)then
      write(output,'(*(es25.16e3,1x))')model%score(x,y)
    else
      if(mode==2.or.mode==4)weight=weight(1)
      write(output,'(*(es25.16e3,1x))')model%score(x,y,weight)
    end if
    write(output,*)model%n_features_in_,model%n_iter_,merge(1,0,model%converged_)
    model%alpha=model%alpha*0.5_dp;y=y*1.1_dp
  end do
  close(output)
end program
