program fastica_sources
 use sklearn,only:dp,fastica,fastica_result
 implicit none
 type(fastica_result)::fit
 real(dp)::x(500,2),s1,s2,t
 integer::i
 do i=1,500
  t=real(i,dp)/500;s1=sin(31*t);s2=merge(1._dp,-1._dp,sin(73*t)>0)
  x(i,:)=[s1+.5_dp*s2+3,.2_dp*s1+s2-1]
 end do
 fit=fastica(x,whiten_solver='eigh',random_state=42,max_iter=1000)
 print *,'Iterations: ',fit%n_iter
 print *,'Source dimensions: ',shape(fit%sources)
 print *,'Unmixing: ',fit%unmixing
end program
