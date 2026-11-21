program neighborhood_components
  use sklearn, only: dp,neighborhood_components_analysis
  implicit none
  type(neighborhood_components_analysis) :: model
  real(dp) :: x(24,3)
  real(dp), allocatable :: embedded(:,:)
  integer :: y(24),i
  do i=1,24
    y(i)=mod(i,3)
    x(i,:)=[real(y(i),dp),sin(real(i,dp)),cos(real(i,dp))]
  end do
  model%n_components=2;model%init='identity';model%max_iter=5
  embedded=model%fit_transform(x,y)
  print *, 'Embedded shape:',shape(embedded)
  print *, 'Expected correctly classified neighbors:',model%objective_
  print *, 'Completed optimizer iterations:',model%n_optimizer_iter_
end program
