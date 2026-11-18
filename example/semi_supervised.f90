program semi_supervised_example
  use sklearn
  implicit none
  type(label_spreading) :: model
  real(dp), allocatable :: x(:,:)
  integer, allocatable :: truth(:),labels(:)
  integer :: i
  call make_moons(100,x,truth,noise=0.05_dp,random_state=42)
  labels=truth
  do i=1,size(labels)
    if(mod(i,5)/=0) labels(i)=-1
  end do
  model%kernel='knn'; model%n_neighbors=7; model%max_iter=1000; model%tol=1.e-8_dp
  call model%fit(x,labels)
  print *, 'Labeled samples: ',count(labels/=-1),' of ',size(labels)
  print *, 'Transductive accuracy: ',accuracy_score(truth,model%transduction_)
  print *, 'Converged: ',model%converged_,' iterations: ',model%n_iter_
end program
