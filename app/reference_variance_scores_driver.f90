program reference_variance_scores_driver
 use sklearn,only:dp,r2_score,explained_variance_score,r2_score_raw,explained_variance_score_raw
 use,intrinsic::ieee_exceptions
 implicit none
 integer::metric,mode,ny,qy,np,qp,ky,kp,nw,no,i
 real(dp),allocatable::y(:,:),p(:,:),w(:),ow(:),v(:),yc(:,:),pc(:,:)
 logical::force,halting(size(ieee_all)),raised(size(ieee_all)),now,state
 character(24)::aggregation
 read(*,*) metric,mode,force,ny,qy,np,qp,ky,kp,nw,no
 allocate(y(ny,qy),p(np,qp),ow(no))
 read(*,*) y
 read(*,*) p
 if(nw>=0) then
  allocate(w(nw));read(*,*) w
 end if
 if(no>0) read(*,*) ow
 yc=y;pc=p
 aggregation='uniform_average'
 if(mode==1) aggregation='variance_weighted'
 ! Exercise preservation of existing exception flags as well as checked traps.
 call ieee_set_flag(ieee_inexact,.true.)
 do i=1,size(ieee_all)
  call ieee_get_flag(ieee_all(i),raised(i))
  call ieee_get_halting_mode(ieee_all(i),halting(i))
 end do
 if(ky==1.and.kp==1) then
  if(metric==0) then
   if(mode==3) then
    v=r2_score_raw(y(:,1),p(:,1),sample_weight=w,force_finite=force)
   else if(mode==2) then
    v=[r2_score(y(:,1),p(:,1),sample_weight=w,multioutput=ow,force_finite=force)]
   else
    v=[r2_score(y(:,1),p(:,1),sample_weight=w,multioutput=trim(aggregation),force_finite=force)]
   end if
  else if(metric==1) then
   if(mode==3) then
    v=explained_variance_score_raw(y(:,1),p(:,1),sample_weight=w,force_finite=force)
   else if(mode==2) then
    v=[explained_variance_score(y(:,1),p(:,1),sample_weight=w,multioutput=ow,force_finite=force)]
   else
    v=[explained_variance_score(y(:,1),p(:,1),sample_weight=w,multioutput=trim(aggregation),force_finite=force)]
   end if
  end if
 else if(ky==1.and.kp==2) then
  if(metric==0) then
   if(mode==3) then
    v=r2_score_raw(y(:,1),p,sample_weight=w,force_finite=force)
   else if(mode==2) then
    v=[r2_score(y(:,1),p,sample_weight=w,multioutput=ow,force_finite=force)]
   else
    v=[r2_score(y(:,1),p,sample_weight=w,multioutput=trim(aggregation),force_finite=force)]
   end if
  else if(metric==1) then
   if(mode==3) then
    v=explained_variance_score_raw(y(:,1),p,sample_weight=w,force_finite=force)
   else if(mode==2) then
    v=[explained_variance_score(y(:,1),p,sample_weight=w,multioutput=ow,force_finite=force)]
   else
    v=[explained_variance_score(y(:,1),p,sample_weight=w,multioutput=trim(aggregation),force_finite=force)]
   end if
  end if
 else if(ky==2.and.kp==1) then
  if(metric==0) then
   if(mode==3) then
    v=r2_score_raw(y,p(:,1),sample_weight=w,force_finite=force)
   else if(mode==2) then
    v=[r2_score(y,p(:,1),sample_weight=w,multioutput=ow,force_finite=force)]
   else
    v=[r2_score(y,p(:,1),sample_weight=w,multioutput=trim(aggregation),force_finite=force)]
   end if
  else if(metric==1) then
   if(mode==3) then
    v=explained_variance_score_raw(y,p(:,1),sample_weight=w,force_finite=force)
   else if(mode==2) then
    v=[explained_variance_score(y,p(:,1),sample_weight=w,multioutput=ow,force_finite=force)]
   else
    v=[explained_variance_score(y,p(:,1),sample_weight=w,multioutput=trim(aggregation),force_finite=force)]
   end if
  end if
 else if(ky==2.and.kp==2) then
  if(metric==0) then
   if(mode==3) then
    v=r2_score_raw(y,p,sample_weight=w,force_finite=force)
   else if(mode==2) then
    v=[r2_score(y,p,sample_weight=w,multioutput=ow,force_finite=force)]
   else
    v=[r2_score(y,p,sample_weight=w,multioutput=trim(aggregation),force_finite=force)]
   end if
  else if(metric==1) then
   if(mode==3) then
    v=explained_variance_score_raw(y,p,sample_weight=w,force_finite=force)
   else if(mode==2) then
    v=[explained_variance_score(y,p,sample_weight=w,multioutput=ow,force_finite=force)]
   else
    v=[explained_variance_score(y,p,sample_weight=w,multioutput=trim(aggregation),force_finite=force)]
   end if
  end if
 end if
 state=.true.
 do i=1,size(ieee_all)
  call ieee_get_flag(ieee_all(i),now);state=state.and.(now.eqv.raised(i))
  call ieee_get_halting_mode(ieee_all(i),now);state=state.and.(now.eqv.halting(i))
 end do
 write(*,*) 'state',merge(1,0,state)
 write(*,*) 'identity',merge(1,0,all(y==yc).and.all(p==pc))
 write(*,*) 'values',v
end program
