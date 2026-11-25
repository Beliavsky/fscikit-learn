program connected_linkage
  use sklearn,only:dp,agglomerative_clustering,csr_from_dense
  implicit none
  type(agglomerative_clustering)::model
  real(dp)::x(4,1),g(4,4)
  integer::i
  x(:,1)=[0._dp,2._dp,5._dp,9._dp];g=0
  do i=1,3
    g(i,i+1)=1
  end do
  model%linkage='average';model%connectivity_csr=csr_from_dense(g)
  model%n_clusters=0;model%distance_threshold=3._dp
  call model%fit(x)
  print *,'Average-linkage threshold labels:',model%labels_
  print *,'Merge distances:',model%distances_
end program
