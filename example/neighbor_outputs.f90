program neighbor_outputs
 use sklearn,only:dp,kneighbors_regressor,radius_neighbors_regressor
 implicit none
 type(kneighbors_regressor)::knn
 type(radius_neighbors_regressor)::radius
 real(dp)::x(8,1),y(8,2),query(2,1)
 real(dp),allocatable::prediction(:,:)
 integer::i
 do i=1,8
  x(i,1)=i-1;y(i,:)=[sin(x(i,1)),cos(x(i,1))]
 end do
 query(:,1)=[.5_dp,2.5_dp]
 knn%n_neighbors=2;call knn%fit(x,y)
 prediction=knn%predict_matrix(query)
 print *,'Joint K-neighbor predictions:'
 do i=1,size(prediction,1)
  print *,prediction(i,:)
 end do
 prediction=knn%predict_matrix()
 print *,'Training prediction excluding its own row:',prediction(1,:)
 radius%radius=.6_dp;radius%weights='distance';call radius%fit(x,y)
 prediction=radius%predict_matrix(query)
 print *,'Joint radius prediction:',prediction(1,:)
end program
