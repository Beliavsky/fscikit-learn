program sparse_components
  use sklearn, only: dp,sparse_pca,dictionary_learning
  implicit none
  type(sparse_pca) :: projection
  type(dictionary_learning) :: learner
  real(dp) :: x(24,5)
  real(dp), allocatable :: scores(:,:),restored(:,:),codes(:,:)
  integer :: i
  do i=1,24
    x(i,:)=[sin(real(i,dp)),cos(real(i,dp)),sin(real(2*i,dp)),cos(real(3*i,dp)),sin(real(4*i,dp))]
  end do
  projection%n_components=3; projection%alpha=.2_dp; projection%max_iter=40
  scores=projection%fit_transform(x); restored=projection%inverse_transform(scores)
  learner%n_components=3; learner%alpha=.2_dp; learner%max_iter=40
  codes=learner%fit_transform(x)
  print *, 'SparsePCA iterations and reconstruction RMSE: ',projection%n_iter_,sqrt(sum((x-restored)**2)/size(x))
  print *, 'Dictionary iterations and final objective: ',learner%n_iter_,learner%error_(learner%n_iter_)
  print *, 'Nonzero learned code entries: ',count(abs(codes)>1.e-12_dp)
end program
