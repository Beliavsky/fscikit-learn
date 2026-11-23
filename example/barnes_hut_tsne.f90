program barnes_hut_tsne
  use sklearn, only: dp,tsne,csr_matrix,csr_from_dense,pairwise_distances,trustworthiness
  implicit none
  type(tsne) :: model
  type(csr_matrix) :: graph
  real(dp) :: x(20,3)
  real(dp), allocatable :: embedding(:,:),distances(:,:)
  integer :: i
  do i=1,size(x,1)
    x(i,:)=[cos(.3_dp*i),sin(.3_dp*i),.05_dp*i]
  end do
  model%perplexity=4;model%init='random';model%random_state=42;model%max_iter=300
  embedding=model%fit_transform(x)
  print *, 'Barnes-Hut embedding shape:',shape(embedding)
  print *, 'KL divergence:',model%kl_divergence_
  print *, 'Trustworthiness:',trustworthiness(x,embedding,3)
  distances=pairwise_distances(x,x);graph=csr_from_dense(distances,.true.)
  model%metric='precomputed';embedding=model%fit_transform(graph)
  print *, 'CSR distance embedding shape:',shape(embedding)
end program
