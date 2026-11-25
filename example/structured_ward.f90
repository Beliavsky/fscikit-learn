program structured_ward
  use sklearn,only:dp,agglomerative_clustering,csr_from_dense
  implicit none
  type(agglomerative_clustering)::model
  real(dp)::x(4,1),graph(4,4)
  integer::i
  x(:,1)=[0._dp,1._dp,4._dp,5._dp];graph=0
  do i=1,3
    graph(i,i+1)=1
  end do
  model%connectivity_csr=csr_from_dense(graph)
  model%n_clusters=0;model%distance_threshold=2._dp
  call model%fit(x)
  print *,'Threshold clusters:',model%n_clusters_
  print *,'Labels:',model%labels_
  model%n_clusters=2;deallocate(model%distance_threshold)
  model%compute_full_tree='false';model%compute_distances=.false.
  call model%fit(x)
  print *,'Partial-tree merge count:',size(model%children_,1)
end program
