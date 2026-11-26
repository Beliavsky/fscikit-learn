program joint_forest_classification
 use sklearn_common,only:dp
 use sklearn_ensemble,only:random_forest_classifier
 use sklearn_tree,only:tree_probability
 implicit none
 type(random_forest_classifier)::model
 type(tree_probability),allocatable::prob(:)
 real(dp)::x(80,2)
 integer::y(80,2),i
 do i=1,80
  x(i,:)=[real(i,dp)/80,real(mod(i,7),dp)];y(i,:)=[merge(1,0,i>40),merge(1,0,mod(i,7)>3)]
 end do
 model%n_estimators=100;model%random_state=7;model%oob_score=.true.;model%class_weight='balanced'
 call model%fit(x,y);prob=model%predict_proba_multi(x)
 print *,'Subset accuracy:',model%score(x,y)
 print *,'OOB subset accuracy:',model%oob_score_
 print *,'First output probabilities:',prob(1)%values(1,:)
end program
