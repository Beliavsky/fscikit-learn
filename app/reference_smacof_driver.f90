program reference_smacof_driver
 use sklearn_common,only:dp
 use sklearn_smacof,only:smacof,smacof_result
 use sklearn_mds,only:mds
 implicit none
 type(smacof_result)::fit
 type(mds)::model
 real(dp),allocatable::d(:,:),init(:,:)
 real(dp)::eps
 integer::n,k,initial,metric,seed,runs,limit,i,status,case_id
 character(8)::normalized
 character(16)::argument
 call get_command_argument(1,argument)
 case_id=0
 do
  read(*,*,iostat=status)n,k,initial,metric,seed,runs,limit,eps,normalized
  if(status<0)exit
  if(status/=0)error stop 'invalid SMACOF request'
  allocate(d(n,n))
  do i=1,n
   read(*,*)d(i,:)
  end do
  if(initial>0)then
   allocate(init(n,initial))
   do i=1,n
    read(*,*)init(i,:)
   end do
  end if
  if(argument=='estimator')then
   model%n_components=k;model%metric=metric/=0;model%n_init=runs;model%max_iter=limit
   model%eps=eps;model%random_state=seed;model%normalized_stress=normalized;model%dissimilarity='precomputed'
   call model%fit(d,init)
   fit%embedding=model%embedding_;fit%stress=model%stress_;fit%n_iter=model%n_iter_
   fit%n_init_effective=runs;if(allocated(init))fit%n_init_effective=1
   fit%normalized_stress=normalized=='true'.or.(normalized=='auto'.and.metric==0)
  else
   fit=smacof(d,n_components=k,metric=metric/=0,init=init,n_init=runs,max_iter=limit, &
    eps=eps,random_state=seed,normalized_stress=trim(normalized))
  end if
  case_id=case_id+1;write(*,'(a,i0)')'CASE ',case_id
  write(*,*)shape(fit%embedding),fit%n_iter,fit%n_init_effective,merge(1,0,fit%normalized_stress)
  write(*,'(*(es26.17e3,1x))')fit%embedding
  write(*,'(es26.17e3)')fit%stress
  deallocate(d);if(allocated(init))deallocate(init)
 end do
end program
