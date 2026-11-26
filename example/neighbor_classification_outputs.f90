program neighbor_classification_outputs
 use sklearn,only:dp,kneighbors_classifier
 use sklearn_neighbors,only:neighbor_probability
 implicit none
 type(kneighbors_classifier)::model
 type(neighbor_probability),allocatable::prob(:)
 real(dp)::x(6,1),query(2,1)
 integer::labels(6,2)
 integer,allocatable::prediction(:,:)
 x(:,1)=[0._dp,1._dp,2._dp,3._dp,4._dp,5._dp]
 labels(:,1)=[0,0,0,1,1,1];labels(:,2)=[2,2,4,4,9,9]
 query(:,1)=[.2_dp,4.8_dp];model%n_neighbors=2
 call model%fit(x,labels)
 prediction=model%predict_matrix(query);prob=model%predict_proba_multioutput(query)
 print *,'Joint labels:',prediction
 print *,'Second-output classes:',prob(2)%classes
 print *,'Second-output probabilities:',prob(2)%values
end program
