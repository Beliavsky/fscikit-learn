program warm_start_forests
 use sklearn_common,only:dp
 use sklearn_ensemble,only:random_forest_regressor
 implicit none
 type(random_forest_regressor)::forest
 real(dp)::x(60,2),y(60,2)
 integer::i,n
 do i=1,60
  x(i,:)=[real(i,dp)/60,real(mod(i,7),dp)];y(i,:)=[sin(x(i,1)),x(i,1)**2]
 end do
 forest%warm_start=.true.;forest%random_state=7;forest%oob_score=.true.
 do n=20,100,20
  forest%n_estimators=n;call forest%fit(x,y)
  print *,'Trees:',size(forest%estimators_),' OOB R2:',forest%oob_score_
 end do
end program
