program nonlinear_embedding
  use sklearn, only: tsne,trustworthiness,dp
  implicit none
  type(tsne) :: model
  real(dp) :: x(48,4),group
  real(dp), allocatable :: embedding(:,:)
  integer :: i
  do i=1,48
    group=real((i-1)/16,dp)
    x(i,:)=[3*group+sin(real(i,dp)),2*group+cos(.7_dp*i),sin(.3_dp*i),cos(.1_dp*i)]
  end do
  model%method='exact';model%perplexity=8; model%max_iter=400; model%random_state=42
  embedding=model%fit_transform(x)
  print *, 'Dense exact t-SNE embedding shape: ',shape(embedding)
  print *, 'Learning rate, final KL and zero-based iteration: ', &
    model%learning_rate_,model%kl_divergence_,model%n_iter_
  print *, 'Four-neighbor trustworthiness: ',trustworthiness(x,embedding,4)
end program
