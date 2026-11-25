program single_linkage_trees
  use sklearn,only:dp,agglomerative_clustering,csr_from_dense
  implicit none
  type(agglomerative_clustering)::model
  real(dp)::x(4,1),g(4,4)
  integer::i
  x(:,1)=[0._dp,4._dp,5._dp,10._dp];g=0
  do i=1,3
    g(i,i+1)=1
  end do
  model%linkage='single';model%connectivity_csr=csr_from_dense(g)
  model%compute_full_tree='false';model%n_clusters=3
  call model%fit(x)
  print *,'Single-linkage partition:',model%labels_
  print *,'Full child-tree merges:',size(model%children_,1)
  model%n_clusters=0;model%distance_threshold=4;model%compute_full_tree='auto'
  call model%fit(x)
  print *,'Threshold partition:',model%labels_
end program
