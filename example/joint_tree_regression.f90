program joint_tree_regression
 use sklearn_tree,only:decision_tree_regressor
 use sklearn_common,only:dp
 implicit none
 type(decision_tree_regressor)::model
 real(dp)::x(6,1),y(6,2)
 real(dp),allocatable::prediction(:,:)
 integer::i
 x(:,1)=[0._dp,1._dp,2._dp,3._dp,4._dp,5._dp]
 y(:,1)=x(:,1)**2;y(:,2)=5-x(:,1)
 model%max_depth=3;model%random_state=7
 call model%fit(x,y);prediction=model%predict_multi(x)
 do i=1,size(x,1);print *,x(i,1),prediction(i,:);end do
end program
