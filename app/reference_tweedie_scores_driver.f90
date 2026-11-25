program reference_tweedie_scores_driver
 use sklearn,only:dp,mean_tweedie_deviance,d2_tweedie_score
 use,intrinsic::ieee_exceptions
 implicit none
 integer::cases,case_index
 read(*,*) cases
 do case_index=1,cases
 block
 integer::metric,ny,qy,np,qp,ky,kp,nw,i,ident
 real(dp)::power,value
 real(dp),allocatable::y(:,:),p(:,:),w(:),yc(:,:),pc(:,:)
 logical::halting(size(ieee_all)),raised(size(ieee_all)),now,state
 read(*,*) ident,metric,power,ny,qy,np,qp,ky,kp,nw
 allocate(y(ny,qy),p(np,qp))
 read(*,*) y
 read(*,*) p
 if(nw>=0) then
  allocate(w(nw));read(*,*) w
 end if
 yc=y;pc=p
 call ieee_set_flag(ieee_inexact,mod(ident,2)==0)
 do i=1,size(ieee_all)
  call ieee_get_flag(ieee_all(i),raised(i))
  call ieee_get_halting_mode(ieee_all(i),halting(i))
 end do
 if(ky==1.and.kp==1) then
  if(metric==0) then
   value=mean_tweedie_deviance(y(:,1),p(:,1),sample_weight=w,power=power)
  else
   value=d2_tweedie_score(y(:,1),p(:,1),sample_weight=w,power=power)
  end if
 else if(ky==1.and.kp==2) then
  if(metric==0) then
   value=mean_tweedie_deviance(y(:,1),p,sample_weight=w,power=power)
  else
   value=d2_tweedie_score(y(:,1),p,sample_weight=w,power=power)
  end if
 else if(ky==2.and.kp==1) then
  if(metric==0) then
   value=mean_tweedie_deviance(y,p(:,1),sample_weight=w,power=power)
  else
   value=d2_tweedie_score(y,p(:,1),sample_weight=w,power=power)
  end if
 else if(ky==2.and.kp==2) then
  if(metric==0) then
   value=mean_tweedie_deviance(y,p,sample_weight=w,power=power)
  else
   value=d2_tweedie_score(y,p,sample_weight=w,power=power)
  end if
 end if
 state=.true.
 do i=1,size(ieee_all)
  call ieee_get_flag(ieee_all(i),now);state=state.and.(now.eqv.raised(i))
  call ieee_get_halting_mode(ieee_all(i),now);state=state.and.(now.eqv.halting(i))
 end do
 write(*,*) ident,merge(1,0,state),merge(1,0,all(y==yc).and.all(p==pc)),value
 end block
 end do
end program
