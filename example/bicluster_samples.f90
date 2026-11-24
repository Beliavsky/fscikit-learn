program bicluster_samples
  use sklearn, only: dp,make_biclusters,make_checkerboard,spectral_coclustering,adjusted_rand_score
  implicit none
  type(spectral_coclustering) :: model
  real(dp), allocatable :: x(:,:)
  logical, allocatable :: rows(:,:),cols(:,:)
  integer, allocatable :: truth(:)
  integer :: i
  call make_biclusters([60,45],3,x,rows,cols,random_state=42)
  allocate(truth(size(x,1)))
  do i=1,size(x,1)
    truth(i)=findloc(rows(:,i),.true.,dim=1)-1
  end do
  model%n_clusters=3;model%random_state=42;call model%fit(x)
  print *, 'Diagonal data shape:',shape(x)
  print *, 'Recovered row partition ARI:',adjusted_rand_score(truth,model%row_labels_)
  call make_checkerboard([36,28],[2,3],x,rows,cols,random_state=42)
  print *, 'Checkerboard data shape:',shape(x)
  print *, 'Cartesian blocks:',size(rows,1)
  print *, 'Positive checkerboard:',all(x>0)
end program
