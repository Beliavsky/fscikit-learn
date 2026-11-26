program joint_tree_classification
 use sklearn_tree,only:decision_tree_classifier,tree_probability
 use sklearn_common,only:dp
 implicit none
 type(decision_tree_classifier)::model
 type(tree_probability),allocatable::probabilities(:)
 real(dp)::x(6,1)
 integer::y(6,2),i
 integer,allocatable::prediction(:,:)
 x(:,1)=[0._dp,1._dp,2._dp,3._dp,4._dp,5._dp]
 y(:,1)=[-7,-7,-7,11,11,11];y(:,2)=[0,1,2,0,1,2]
 model%random_state=7;call model%fit(x,y)
 prediction=model%predict_multi(x);probabilities=model%predict_proba_multi(x)
 do i=1,6;print *,x(i,1),prediction(i,:);end do
 print *,'Class counts:',model%n_classes_
end program
