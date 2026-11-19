program streaming_pca
  use sklearn, only: dp,incremental_pca
  implicit none
  type(incremental_pca) :: model
  real(dp) :: x(6,2)
  real(dp), allocatable :: z(:,:)
  x(:,1)=[1._dp,2._dp,3._dp,4._dp,5._dp,6._dp]; x(:,2)=2*x(:,1)+1
  model%n_components=1
  call model%partial_fit(x(:3,:)); call model%partial_fit(x(4:,:))
  z=model%transform(x)
  print *, 'Samples seen:',model%n_samples_seen_
  print *, 'Explained variance ratio:',model%explained_variance_ratio_
end program
