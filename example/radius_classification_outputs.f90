program radius_classification_outputs
 use sklearn,only:dp,radius_neighbors_classifier,neighbor_probability
 implicit none
 type(radius_neighbors_classifier)::model
 type(neighbor_probability),allocatable::prob(:)
 real(dp)::x(6,1),query(3,1)
 integer::labels(6,2)
 integer,allocatable::prediction(:,:)
 x(:,1)=[0._dp,1._dp,2._dp,3._dp,4._dp,5._dp]
 labels(:,1)=[0,0,0,1,1,1];labels(:,2)=[2,2,4,4,9,9]
 query(:,1)=[.2_dp,4.8_dp,100._dp]
 model%radius=.7_dp;model%outlier_mode='most_frequent'
 call model%fit(x,labels)
 prediction=model%predict_matrix(query);prob=model%predict_proba_multioutput(query)
 print *,'Joint radius labels:',prediction
 print *,'Fitted outlier labels:',model%outlier_label_
 print *,'Second-output classes:',prob(2)%classes
 print *,'Second-output probabilities:',prob(2)%values
end program
