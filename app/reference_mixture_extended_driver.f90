program reference_mixture_extended_driver
 use sklearn_mixture,only:gaussian_mixture
 use sklearn_common,only:dp
 implicit none
 type(gaussian_mixture)::model,fresh
 integer::n,p,k,m,cv,init,seed,limit,runs,warm,mask,draws,i,j,c,q,ios,case_id
 real(dp)::tol,reg
 real(dp),allocatable::x(:,:),query(:,:),prob(:,:),samples(:,:),scores(:)
 integer,allocatable::labels(:),fitlabels(:),samplelabels(:)
 character(16),parameter::covs(4)=[character(16)::'full','tied','diag','spherical']
 character(20),parameter::inits(4)=[character(20)::'kmeans','random','random_from_data','k-means++']
 case_id=0
 do
  read(*,*,iostat=ios)n,p,k,m,cv,init,seed,limit,runs,warm,mask,draws,tol,reg
  if(ios/=0)exit
  model=fresh;model%n_components=k;model%covariance_type=covs(cv);model%init_params=inits(init)
  model%random_state=seed;model%max_iter=limit;model%n_init=runs;model%tol=tol;model%reg_covar=reg
  model%warm_start=warm>0
  allocate(x(n,p),query(m,p))
  do i=1,n
   read(*,*)x(i,:)
  end do
  do i=1,m
   read(*,*)query(i,:)
  end do
  if(btest(mask,0))then
   allocate(model%weights_init(k));read(*,*)model%weights_init
  end if
  if(btest(mask,1))then
   allocate(model%means_init(k,p))
   do c=1,k
    read(*,*)model%means_init(c,:)
   end do
  end if
  if(btest(mask,2))then
   q=k;if(cv==2)q=1
   allocate(model%precisions_init(p,p,q))
   do c=1,q
    do i=1,p
     read(*,*)model%precisions_init(i,:,c)
    end do
   end do
  end if
  do i=0,warm
   fitlabels=model%fit_predict(x)
  end do
  scores=model%score_samples(query);prob=model%predict_proba(query);labels=model%predict(query)
  if(draws>0)then
   call model%sample(draws,samples,samplelabels)
  else
   allocate(samples(0,p),samplelabels(0))
  end if
  case_id=case_id+1;write(*,'(a,i0)')'CASE ',case_id
  write(*,*)model%n_features_in_,model%n_iter_,merge(1,0,model%converged_)
  write(*,'(*(es25.16e3,1x))')model%weights_
  write(*,'(*(es25.16e3,1x))')model%means_
  write(*,'(*(es25.16e3,1x))')model%covariances_
  write(*,'(*(es25.16e3,1x))')model%precisions_cholesky_
  write(*,'(*(es25.16e3,1x))')model%precisions_
  write(*,'(*(es25.16e3,1x))')model%lower_bound_
  write(*,'(*(es25.16e3,1x))')scores
  write(*,'(*(es25.16e3,1x))')prob
  write(*,*)labels
  write(*,'(*(es25.16e3,1x))')model%score(query),model%aic(query),model%bic(query)
  write(*,*)fitlabels
  write(*,'(*(es25.16e3,1x))')samples
  write(*,*)samplelabels
  deallocate(x,query,samples,samplelabels)
 end do
end program
