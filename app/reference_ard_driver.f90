program reference_ard_driver
  use sklearn_linear_model,only:ard_regression
  use sklearn_common,only:dp
  implicit none
  type(ard_regression)::model
  real(dp),allocatable::x(:,:),y(:)
  integer::n,p,column,intercept,history,i,unit,out
  character(1024)::input,output
  call get_command_argument(1,input);call get_command_argument(2,output)
  open(newunit=unit,file=trim(input),status='old')
  read(unit,*)n,p,column,intercept,history,model%max_iter,model%tol,model%alpha_1,model%alpha_2, &
    model%lambda_1,model%lambda_2,model%threshold_lambda
  allocate(x(n,p),y(n))
  do i=1,n
    read(unit,*)x(i,:),y(i)
  end do
  close(unit);model%fit_intercept=intercept==1;model%compute_score=history==1
  if(column==1)then
    call model%fit(x,reshape(y,[n,1]))
  else
    call model%fit(x,y)
  end if
  open(newunit=out,file=trim(output),status='replace')
  write(out,'(*(es26.17e3,1x))')model%coef_
  write(out,'(*(es26.17e3,1x))')model%intercept_,model%alpha_
  write(out,'(*(es26.17e3,1x))')model%lambda_
  if(size(model%sigma_)>0)then
    write(out,'(*(es26.17e3,1x))')model%sigma_
  else
    write(out,'(a)')''
  end if
  write(out,'(*(es26.17e3,1x))')model%predict(x+.017_dp)
  write(out,'(*(es26.17e3,1x))')model%predict_std(x+.017_dp)
  if(size(model%scores_)>0)then
    write(out,'(*(es26.17e3,1x))')model%scores_
  else
    write(out,'(a)')''
  end if
  write(out,'(*(es26.17e3,1x))')model%x_offset_
  write(out,*)model%n_features_in_,model%n_iter_,model%rank_
  close(out)
end program
