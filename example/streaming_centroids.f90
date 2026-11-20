program streaming_centroids
  use sklearn, only: dp,mini_batch_kmeans
  implicit none
  type(mini_batch_kmeans) :: model
  real(dp) :: first(4,2),second(3,2),query(2,2)
  first(:,1)=[-2._dp,-1._dp,4._dp,5._dp]; first(:,2)=[0._dp,1._dp,5._dp,4._dp]
  second(:,1)=[-3._dp,6._dp,5._dp]; second(:,2)=[1._dp,5._dp,6._dp]
  query(:,1)=[-2._dp,5._dp]; query(:,2)=[1._dp,5._dp]
  model%n_clusters=2; model%random_state=42; model%reassignment_ratio=0
  call model%partial_fit(first); call model%partial_fit(second)
  print *, 'Centers, first coordinate: ',model%cluster_centers_(:,1)
  print *, 'Accumulated weights: ',model%counts_
  print *, 'Query cluster labels: ',model%predict(query)
end program
