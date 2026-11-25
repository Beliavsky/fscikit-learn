program ward_trees
  use sklearn,only:dp,ward_tree,hierarchy_result
  implicit none
  type(hierarchy_result)::tree
  real(dp)::x(4),connectivity(4,4)
  integer::i
  x=[0._dp,1._dp,4._dp,5._dp];connectivity=1
  tree=ward_tree(x,connectivity,n_clusters=2,return_distance=.true.)
  do i=1,size(tree%children,1)
    print *,'Merge:',tree%children(i,:),'height:',tree%distances(i)
  end do
  print *,'Forest parents:',tree%parents
end program
