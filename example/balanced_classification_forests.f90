program balanced_classification_forests
  use sklearn, only: dp,random_forest_classifier,extra_trees_classifier
  implicit none
  type(random_forest_classifier) :: forest
  type(extra_trees_classifier) :: extra
  real(dp) :: x(8,1),weights(8)
  real(dp), allocatable :: probability(:,:)
  integer :: y(8),i
  x(:,1)=[(real(i,dp),i=1,8)];y=[-7,-7,-7,-7,9,9,42,42];weights=[1._dp,0._dp,2._dp,1._dp,1._dp,1._dp,1._dp,1._dp]
  forest%n_estimators=4;forest%random_state=7;forest%max_depth=2
  forest%criterion='entropy';forest%class_weight='balanced_subsample'
  call forest%fit(x,y,weights)
  print *, 'Balanced forest labels:',forest%predict(x)
  print *, 'Weighted forest accuracy:',forest%score(x,y,weights)
  probability=forest%predict_proba(x)
  print *, 'Class order:',forest%classes_
  print *, 'First-row probabilities:',probability(1,:)
  extra%n_estimators=4;extra%criterion='log_loss';extra%class_weight='balanced'
  call extra%fit(x,y,weights)
  print *, 'Balanced ExtraTrees labels:',extra%predict(x)
end program
