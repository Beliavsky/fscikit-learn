program reference_ols_extended_driver
 use sklearn_common,only:dp
 use sklearn_linear_model,only:linear_regression
 use sklearn_sparse,only:csr_from_dense
 implicit none
 type(linear_regression)::model
 real(dp),allocatable::x(:,:),y(:,:),q(:,:),w(:),pred(:,:),qy(:,:),qw(:)
 real(dp)::score
 integer::j,n,p,t,m,train_sparse,query_sparse,matrix_target,intercept,positive,weights,status,id,i
 id=0
 do
  read(*,*,iostat=status)n,p,t,m,train_sparse,query_sparse,matrix_target,intercept,positive,weights
  if(status<0)exit
  if(status/=0)error stop 'invalid OLS request'
  allocate(x(n,p),y(n,t),q(m,p),w(n),qy(m,t),qw(m))
  do i=1,n
   read(*,*)x(i,:);read(*,*)y(i,:)
  end do
  do i=1,m
   read(*,*)q(i,:)
  end do
  w=1;if(weights==1)read(*,*)w
  if(weights==2)then
   read(*,*)score;w=score
  end if
  model%fit_intercept=intercept/=0;model%positive=positive/=0
  if(train_sparse/=0)then
   if(matrix_target/=0)then
    if(weights==2)then
     call model%fit(csr_from_dense(x),y,sample_weight=score)
    else if(weights==1)then
     call model%fit(csr_from_dense(x),y,w)
    else
     call model%fit(csr_from_dense(x),y)
    end if
   else
    if(weights==2)then
     call model%fit(csr_from_dense(x),y(:,1),sample_weight=score)
    else if(weights==1)then
     call model%fit(csr_from_dense(x),y(:,1),w)
    else
     call model%fit(csr_from_dense(x),y(:,1))
    end if
   end if
  else
   if(matrix_target/=0)then
    if(weights==2)then
     call model%fit(x,y,sample_weight=score)
    else if(weights==1)then
     call model%fit(x,y,w)
    else
     call model%fit(x,y)
    end if
   else
    if(weights==2)then
     call model%fit(x,y(:,1),sample_weight=score)
    else if(weights==1)then
     call model%fit(x,y(:,1),w)
    else
     call model%fit(x,y(:,1))
    end if
   end if
  end if
  if(query_sparse/=0)then
   pred=model%predict_matrix(csr_from_dense(q))
  else
   pred=model%predict_matrix(q)
  end if
  do i=1,m
   qw(i)=.5_dp+real(i-1,dp)/max(1,m-1)
   qy(i,:)=[(.2_dp*i-.1_dp*j,j=1,t)]
  end do
  if(query_sparse/=0)then
   if(matrix_target/=0)then
    score=model%score(csr_from_dense(q),qy,qw)
   else
    score=model%score(csr_from_dense(q),qy(:,1),qw)
   end if
  else
   if(matrix_target/=0)then
    score=model%score(q,qy,qw)
   else
    score=model%score(q,qy(:,1),qw)
   end if
  end if
  id=id+1;write(*,'(a,i0)')'CASE ',id
  write(*,*)model%n_features_in_,model%n_targets_,model%rank_,merge(1,0,allocated(model%singular_))
  write(*,'(*(es26.17e3,1x))')model%coef_matrix_
  write(*,'(*(es26.17e3,1x))')model%intercepts_
  write(*,'(*(es26.17e3,1x))')pred
  if(allocated(model%singular_))then
   write(*,'(*(es26.17e3,1x))')model%singular_
  else
   write(*,'(a)')''
  end if
  write(*,'(es26.17e3)')score
  deallocate(x,y,q,w,qy,qw)
 end do
end program
