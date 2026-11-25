program reference_distribution_errors_driver
 use sklearn,only:dp,mean_squared_log_error,root_mean_squared_log_error, &
  mean_squared_log_error_raw,root_mean_squared_log_error_raw,mean_poisson_deviance,mean_gamma_deviance
 use,intrinsic::ieee_exceptions
 implicit none
 integer::cases,case_index
 read(*,*) cases
 do case_index=1,cases
 block
 integer::metric,mode,ny,qy,np,qp,ky,kp,nw,no,i,ident
 real(dp),allocatable::y(:,:),p(:,:),w(:),ow(:),v(:),yc(:,:),pc(:,:)
 logical::halting(size(ieee_all)),raised(size(ieee_all)),now,state
 character(24)::aggregation
 read(*,*) ident,metric,mode,ny,qy,np,qp,ky,kp,nw,no
 allocate(y(ny,qy),p(np,qp),ow(no))
 read(*,*) y
 read(*,*) p
 if(nw>=0) then
  allocate(w(nw));read(*,*) w
 end if
 if(no>0) read(*,*) ow
 yc=y;pc=p
 aggregation='uniform_average'
 if(mode==3) aggregation='raw_values'
 if(mode==4) aggregation='variance_weighted'
 if(mode==5) aggregation='bad'
 call ieee_set_flag(ieee_inexact,.true.)
 do i=1,size(ieee_all)
  call ieee_get_flag(ieee_all(i),raised(i))
  call ieee_get_halting_mode(ieee_all(i),halting(i))
 end do
 if(ky==1.and.kp==1) then
  if(metric==0) then
   if(mode==1) then
    v=mean_squared_log_error_raw(y(:,1),p(:,1),sample_weight=w)
   else if(mode==2) then
    v=[mean_squared_log_error(y(:,1),p(:,1),sample_weight=w,multioutput=ow)]
   else
    v=[mean_squared_log_error(y(:,1),p(:,1),sample_weight=w,multioutput=trim(aggregation))]
   end if
  else if(metric==1) then
   if(mode==1) then
    v=root_mean_squared_log_error_raw(y(:,1),p(:,1),sample_weight=w)
   else if(mode==2) then
    v=[root_mean_squared_log_error(y(:,1),p(:,1),sample_weight=w,multioutput=ow)]
   else if(mode==6) then
    v=[root_mean_squared_log_error(y=y(:,1),predicted=p(:,1),sample_weight=w,output_weights=ow)]
   else if(mode==7) then
    v=[root_mean_squared_log_error(y(:,1),p(:,1),sample_weight=w, &
     multioutput='uniform_average',output_weights=ow)]
   else
    v=[root_mean_squared_log_error(y(:,1),p(:,1),sample_weight=w,multioutput=trim(aggregation))]
   end if
  else if(metric==2) then
   v=[mean_poisson_deviance(y(:,1),p(:,1),sample_weight=w)]
  else if(metric==3) then
   v=[mean_gamma_deviance(y(:,1),p(:,1),sample_weight=w)]
  end if
 else if(ky==1.and.kp==2) then
  if(metric==0) then
   if(mode==1) then
    v=mean_squared_log_error_raw(y(:,1),p,sample_weight=w)
   else if(mode==2) then
    v=[mean_squared_log_error(y(:,1),p,sample_weight=w,multioutput=ow)]
   else
    v=[mean_squared_log_error(y(:,1),p,sample_weight=w,multioutput=trim(aggregation))]
   end if
  else if(metric==1) then
   if(mode==1) then
    v=root_mean_squared_log_error_raw(y(:,1),p,sample_weight=w)
   else if(mode==2) then
    v=[root_mean_squared_log_error(y(:,1),p,sample_weight=w,multioutput=ow)]
   else if(mode==6) then
    v=[root_mean_squared_log_error(y=y(:,1),predicted=p,sample_weight=w,output_weights=ow)]
   else if(mode==7) then
    v=[root_mean_squared_log_error(y(:,1),p,sample_weight=w, &
     multioutput='uniform_average',output_weights=ow)]
   else
    v=[root_mean_squared_log_error(y(:,1),p,sample_weight=w,multioutput=trim(aggregation))]
   end if
  else if(metric==2) then
   v=[mean_poisson_deviance(y(:,1),p,sample_weight=w)]
  else if(metric==3) then
   v=[mean_gamma_deviance(y(:,1),p,sample_weight=w)]
  end if
 else if(ky==2.and.kp==1) then
  if(metric==0) then
   if(mode==1) then
    v=mean_squared_log_error_raw(y,p(:,1),sample_weight=w)
   else if(mode==2) then
    v=[mean_squared_log_error(y,p(:,1),sample_weight=w,multioutput=ow)]
   else
    v=[mean_squared_log_error(y,p(:,1),sample_weight=w,multioutput=trim(aggregation))]
   end if
  else if(metric==1) then
   if(mode==1) then
    v=root_mean_squared_log_error_raw(y,p(:,1),sample_weight=w)
   else if(mode==2) then
    v=[root_mean_squared_log_error(y,p(:,1),sample_weight=w,multioutput=ow)]
   else if(mode==6) then
    v=[root_mean_squared_log_error(y=y,predicted=p(:,1),sample_weight=w,output_weights=ow)]
   else if(mode==7) then
    v=[root_mean_squared_log_error(y,p(:,1),sample_weight=w, &
     multioutput='uniform_average',output_weights=ow)]
   else
    v=[root_mean_squared_log_error(y,p(:,1),sample_weight=w,multioutput=trim(aggregation))]
   end if
  else if(metric==2) then
   v=[mean_poisson_deviance(y,p(:,1),sample_weight=w)]
  else if(metric==3) then
   v=[mean_gamma_deviance(y,p(:,1),sample_weight=w)]
  end if
 else if(ky==2.and.kp==2) then
  if(metric==0) then
   if(mode==1) then
    v=mean_squared_log_error_raw(y,p,sample_weight=w)
   else if(mode==2) then
    v=[mean_squared_log_error(y,p,sample_weight=w,multioutput=ow)]
   else
    v=[mean_squared_log_error(y,p,sample_weight=w,multioutput=trim(aggregation))]
   end if
  else if(metric==1) then
   if(mode==1) then
    v=root_mean_squared_log_error_raw(y,p,sample_weight=w)
   else if(mode==2) then
    v=[root_mean_squared_log_error(y,p,sample_weight=w,multioutput=ow)]
   else if(mode==6) then
    v=[root_mean_squared_log_error(y=y,predicted=p,sample_weight=w,output_weights=ow)]
   else if(mode==7) then
    v=[root_mean_squared_log_error(y,p,sample_weight=w, &
     multioutput='uniform_average',output_weights=ow)]
   else
    v=[root_mean_squared_log_error(y,p,sample_weight=w,multioutput=trim(aggregation))]
   end if
  else if(metric==2) then
   v=[mean_poisson_deviance(y,p,sample_weight=w)]
  else if(metric==3) then
   v=[mean_gamma_deviance(y,p,sample_weight=w)]
  end if
 end if
 state=.true.
 do i=1,size(ieee_all)
  call ieee_get_flag(ieee_all(i),now);state=state.and.(now.eqv.raised(i))
  call ieee_get_halting_mode(ieee_all(i),now);state=state.and.(now.eqv.halting(i))
 end do
 write(*,*) ident,merge(1,0,state),merge(1,0,all(y==yc).and.all(p==pc)),v
 end block
 end do
end program
