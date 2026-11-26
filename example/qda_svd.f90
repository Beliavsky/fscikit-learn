program qda_svd
 use sklearn,only:dp,quadratic_discriminant_analysis
 implicit none
 type(quadratic_discriminant_analysis)::model
 real(dp)::x(8,5)
 real(dp),allocatable::logs(:,:)
 integer::i,c,y(8)
 do i=1,8
  y(i)=merge(-3,7,i<=4)
  x(i,:)=[sin(real(i,dp)),cos(.7_dp*i),sin(2._dp*i),cos(3._dp*i),.1_dp*i]+.1_dp*y(i)
 end do
 model%reg_param=.2_dp;call model%fit(x,y)
 logs=model%predict_log_proba(x(:2,:))
 do c=1,size(model%classes_)
  print *,'Class:',model%classes_(c),'axes:',size(model%subspaces_(c)%scalings),'rank:',model%subspaces_(c)%rank
 end do
 print *,'Predictions:',model%predict(x(:2,:))
 print *,'Log probabilities:'
 do i=1,size(logs,1)
  print *,logs(i,:)
 end do
end program
