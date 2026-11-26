program reference_qda_svd_driver
 use sklearn_common,only:dp
 use sklearn_discriminant_analysis,only:quadratic_discriminant_analysis
 implicit none
 type(quadratic_discriminant_analysis)::model,fresh
 integer::n,p,m,np,i,c,k,ios,case_id
 real(dp)::reg,tol,score
 real(dp),allocatable::x(:,:),q(:,:),decision(:,:),prob(:,:),logs(:,:),projector(:,:,:),scalings(:)
 integer,allocatable::y(:),truth(:),pred(:),directions(:),ranks(:)
 case_id=0
 do
  read(*,*,iostat=ios)n,p,m,np,reg,tol
  if(ios/=0)exit
  model=fresh;model%reg_param=reg;model%tol=tol;allocate(x(n,p),q(m,p),y(n),truth(m))
  do i=1,n
   read(*,*)x(i,:),y(i)
  end do
  do i=1,m
   read(*,*)q(i,:),truth(i)
  end do
  if(np>=0)then
   allocate(model%priors(np))
   if(np>0)read(*,*)model%priors
  end if
  call model%fit(x,y)
  if(m>0)then
   decision=model%decision_function(q);logs=model%predict_log_proba(q)
   prob=model%predict_proba(q);pred=model%predict(q);score=model%score(q,truth)
  else
   allocate(decision(0,size(model%classes_)),logs(0,size(model%classes_)),prob(0,size(model%classes_)),pred(0))
   score=0
  end if
  k=size(model%classes_);allocate(projector(p,p,k),directions(k),ranks(k),scalings(0))
  do c=1,k
   directions(c)=size(model%subspaces_(c)%scalings);ranks(c)=model%subspaces_(c)%rank
   scalings=[scalings,model%subspaces_(c)%scalings]
   projector(:,:,c)=matmul(model%subspaces_(c)%rotations,transpose(model%subspaces_(c)%rotations))
  end do
  case_id=case_id+1;write(*,'(a,i0)')'CASE ',case_id
  write(*,*)model%n_features_in_,k
  write(*,*)model%classes_
  write(*,'(*(es25.16e3,1x))')model%means_
  write(*,'(*(es25.16e3,1x))')model%priors_
  write(*,'(*(es25.16e3,1x))')model%covariance_
  write(*,'(*(es25.16e3,1x))')scalings
  write(*,'(*(es25.16e3,1x))')projector
  write(*,*)directions
  write(*,*)ranks
  write(*,'(*(es25.16e3,1x))')model%logdet_
  write(*,'(*(es25.16e3,1x))')decision
  write(*,'(*(es25.16e3,1x))')logs
  write(*,'(*(es25.16e3,1x))')prob
  write(*,*)pred
  write(*,'(*(es25.16e3,1x))')score
  deallocate(x,q,y,truth,projector,directions,ranks,scalings,decision,logs,prob,pred)
 end do
end program
