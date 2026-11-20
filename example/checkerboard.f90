program checkerboard
  use sklearn, only: dp,spectral_biclustering
  implicit none
  type(spectral_biclustering) :: model
  real(dp) :: x(6,4)
  real(dp), allocatable :: block(:,:)
  x(:,1)=[12._dp,11._dp,13._dp,2._dp,3._dp,2._dp]; x(:,2)=x(:,1)+.1_dp
  x(:,3)=[2._dp,3._dp,2._dp,15._dp,14._dp,16._dp]; x(:,4)=x(:,3)+.2_dp
  model%n_clusters=2; model%method='log'; model%n_components=1; model%n_best=1
  call model%fit(x)
  print *, 'Row groups: ',model%row_labels_
  print *, 'Column groups: ',model%column_labels_
  print *, 'Bicluster grid contains ',model%n_biclusters_,' blocks'
  block=model%get_submatrix(0,x)
  print *, 'First block dimensions: ',shape(block)
end program
