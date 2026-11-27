program reference_bayesian_ridge_driver
  use sklearn_linear_model,only:bayesian_ridge
  use sklearn_common,only:dp
  implicit none
  type(bayesian_ridge)::model
  real(dp),allocatable::x(:,:),y(:),weight(:)
  integer::n,p,mode,intercept,history,i,input,output
  character(1024)::path
  call get_command_argument(1,path);open(newunit=input,file=trim(path),status='old')
  read(input,*)n,p,mode,intercept,history,model%max_iter,model%tol,model%alpha_1,model%alpha_2, &
    model%lambda_1,model%lambda_2,model%alpha_init,model%lambda_init
  model%fit_intercept=intercept/=0;model%compute_score=history/=0
  allocate(x(n,p),y(n),weight(n))
  do i=1,n
    read(input,*)x(i,:),y(i),weight(i)
  end do
  close(input)
  call get_command_argument(2,path);open(newunit=output,file=trim(path),status='replace')
  select case(mode)
  case(0);call model%fit(x,y)
  case(1);call model%fit(x,y,weight)
  case(2);call model%fit(x,y,weight(1))
  case(3);call model%fit(x,reshape(y,[n,1]),weight)
  case(4);call model%fit(x,reshape(y,[n,1]),weight(1))
  end select
  write(output,'(*(es25.16e3,1x))')model%coef_
  write(output,'(*(es25.16e3,1x))')model%intercept_,model%alpha_,model%lambda_
  write(output,'(*(es25.16e3,1x))')reshape(model%sigma_,[p*p])
  write(output,'(*(es25.16e3,1x))')model%predict(x+.017_dp)
  write(output,'(*(es25.16e3,1x))')model%predict_std(x+.017_dp)
  write(output,'(*(es25.16e3,1x))')model%scores_
  write(output,'(*(es25.16e3,1x))')model%x_offset_
  write(output,*)model%n_features_in_,model%n_iter_,merge(1,0,model%converged_)
  close(output)
end program
