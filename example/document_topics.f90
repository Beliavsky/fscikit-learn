program document_topics
  use sklearn, only: dp,latent_dirichlet_allocation
  implicit none
  type(latent_dirichlet_allocation) :: model
  real(dp) :: counts(6,6)
  real(dp), allocatable :: topics(:,:)
  integer :: i
  counts=1
  counts(1:3,1:3)=8; counts(4:6,4:6)=8
  counts(2,1)=12; counts(5,6)=12
  model%n_components=2; model%max_iter=30; model%random_state=42
  topics=model%fit_transform(counts)
  print *, 'Document topic proportions:'
  do i=1,size(counts,1)
    print '(2f10.6)',topics(i,:)
  end do
  print *, 'Variational score/perplexity: ',model%score(counts),model%perplexity(counts)
  model%total_samples=100
  call model%partial_fit(counts(:2,:))
  print *, 'Updates performed: ',model%n_batch_iter_-1
end program
