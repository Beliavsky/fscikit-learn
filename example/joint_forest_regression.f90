program joint_forest_regression
 use sklearn_common,only:dp
 use sklearn_ensemble,only:extra_trees_regressor
 implicit none
 type(extra_trees_regressor)::model
 real(dp)::x(40,2),y(40,2)
 real(dp),allocatable::prediction(:,:)
 integer::i
 do i=1,40
  x(i,:)=[real(i,dp)/40,real(mod(i,7),dp)];y(i,:)=[sin(x(i,1)),x(i,1)**2]
 end do
 model%n_estimators=100;model%random_state=7
 call model%set_bootstrap(.true.)
 model%oob_score=.true.
 call model%fit(x,y)
 prediction=model%predict_multi(x)
 print *,'Training R2:',model%score(x,y)
 print *,'OOB R2:',model%oob_score_
 print *,'First predicted outputs:',prediction(1,:)
end program
