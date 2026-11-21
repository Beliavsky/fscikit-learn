program weighted_extra_trees
  use sklearn, only: dp,extra_tree_classifier,extra_tree_regressor
  implicit none
  type(extra_tree_classifier) :: classifier
  type(extra_tree_regressor) :: regressor
  real(dp) :: x(6,2),weights(6),y(6)
  x(:,1)=[0._dp,1._dp,2._dp,3._dp,4._dp,5._dp]; x(:,2)=[0._dp,1._dp,0._dp,1._dp,0._dp,1._dp]
  weights=[3._dp,0._dp,2._dp,1._dp,0._dp,4._dp]; y=[0._dp,0._dp,1._dp,1._dp,2._dp,2._dp]
  classifier%max_depth=3; classifier%random_state=42
  regressor%max_depth=3; regressor%random_state=42
  call classifier%fit(x,int(y),weights); call regressor%fit(x,y,weights)
  print *, 'ExtraTree classes:',classifier%predict(x)
  print *, 'ExtraTree regression:',regressor%predict(x)
  print *, 'Active root rows:',classifier%nodes_(1)%n_samples
  print *, 'Weighted root mass:',classifier%nodes_(1)%weighted_n_samples
end program
