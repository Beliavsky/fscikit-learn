program group_validation
  use sklearn, only: dp,group_kfold,index_split,linear_regression
  implicit none
  type(group_kfold) :: splitter
  type(index_split), allocatable :: folds(:)
  type(linear_regression) :: model
  real(dp) :: x(12,1),y(12)
  integer :: groups(12),i
  do i=1,12
    x(i,1)=real(i,dp); y(i)=2*x(i,1)+3; groups(i)=(i-1)/3
  end do
  splitter%n_splits=4; folds=splitter%split(groups)
  do i=1,size(folds)
    call model%fit(x(folds(i)%train,:),y(folds(i)%train))
    print '(a,i0,a,f8.5)', 'Fold ',i,' R2: ',model%score(x(folds(i)%test,:),y(folds(i)%test))
  end do
end program
