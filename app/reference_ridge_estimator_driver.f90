program reference_ridge_estimator_driver
 use sklearn_common,only:dp
 use sklearn_linear_model,only:ridge
 use sklearn_sparse,only:csr_matrix,csr_from_dense
 implicit none
 type(ridge)::model
 type(csr_matrix)::sx,sq
 real(dp),allocatable::x(:,:),y(:,:),q(:,:),w(:),qw(:),penalty(:),prediction(:,:),scalar_weight,qy(:,:)
 integer::n,p,t,m,train_sparse,query_sparse,matrix_target,intercept,positive,limit,seed,weight_mode,penalties,i,j,ios,score_valid,case_id
 character(32)::run_mode
 real(dp)::tol,alpha,score
 character(16)::solver
 case_id=0
 call get_command_argument(1,run_mode)
 do
  read(*,*,iostat=ios)n,p,t,m,train_sparse,query_sparse,matrix_target,intercept,positive,limit,seed, &
   weight_mode,penalties,tol,alpha,solver,score_valid
  if(ios/=0)exit
  case_id=case_id+1;write(*,'(a,1x,i0)')'CASE',case_id
  allocate(x(n,p),y(n,t),q(m,p),qw(m),qy(m,t))
  do j=1,t
   do i=1,m
    qy(i,j)=.1_dp*i-.2_dp*j
   end do
  end do
  do i=1,n
   read(*,*)x(i,:);read(*,*)y(i,:)
  end do
  do i=1,m
   read(*,*)q(i,:)
  end do
  read(*,*)qw
  if(weight_mode==1)then
   allocate(w(n));read(*,*)w
  else if(weight_mode==2)then
   allocate(scalar_weight);read(*,*)scalar_weight
  end if
  if(penalties==1)then
   allocate(penalty(t));read(*,*)penalty
  end if
  model%solver=solver;model%alpha=alpha;model%tol=tol;model%max_iter=limit
  model%fit_intercept=intercept==1;model%positive=positive==1;model%random_state=seed
  if(allocated(model%alpha_per_target))deallocate(model%alpha_per_target)
  if(penalties==1)model%alpha_per_target=penalty
  sx=csr_from_dense(x);sq=csr_from_dense(q)
  if(train_sparse==1)then
   if(matrix_target==1)then
    if(weight_mode==2)then
     call model%fit(sx,y,sample_weight=scalar_weight)
    else
     call model%fit(sx,y,sample_weight=w)
    end if
   else
    if(weight_mode==2)then
     call model%fit(sx,y(:,1),sample_weight=scalar_weight)
    else
     call model%fit(sx,y(:,1),sample_weight=w)
    end if
   end if
  else
   if(matrix_target==1)then
    if(weight_mode==2)then
     call model%fit(x,y,sample_weight=scalar_weight)
    else
     call model%fit(x,y,sample_weight=w)
    end if
   else
    if(weight_mode==2)then
     call model%fit(x,y(:,1),sample_weight=scalar_weight)
    else if(allocated(w))then
     call model%fit(x,y(:,1),sample_weight=w)
    else
     call model%fit(x,y(:,1))
    end if
   end if
  end if
  score=0
  if(query_sparse==1)then
   if(matrix_target==1)then
    prediction=model%predict_matrix(sq)
    if(score_valid==1.or.trim(run_mode)=='force-score')score=model%score(sq,qy,qw)
   else
    prediction=reshape(model%predict(sq),[m,1])
    if(score_valid==1.or.trim(run_mode)=='force-score')score=model%score(sq,qy(:,1),qw)
   end if
  else
   if(matrix_target==1)then
    prediction=model%predict_matrix(q)
    if(score_valid==1.or.trim(run_mode)=='force-score')score=model%score(q,qy,qw)
   else
    prediction=reshape(model%predict(q),[m,1])
    if(score_valid==1.or.trim(run_mode)=='force-score')score=model%score(q,qy(:,1),qw)
   end if
  end if
  write(*,*)model%n_features_in_,model%n_targets_,merge(1,0,model%matrix_target_)
  write(*,'(a)')trim(model%solver_)
  write(*,'(*(i0,1x))')model%n_iter_
  write(*,'(*(es26.17e3,1x))')model%coef_matrix_
  write(*,'(*(es26.17e3,1x))')model%intercepts_
  write(*,'(*(es26.17e3,1x))')prediction
  write(*,'(es26.17e3)')score
  deallocate(x,y,q,qw,qy)
  if(allocated(w))deallocate(w)
  if(allocated(penalty))deallocate(penalty)
  if(allocated(scalar_weight))deallocate(scalar_weight)
 end do
end program
