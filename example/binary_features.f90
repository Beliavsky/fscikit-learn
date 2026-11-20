program binary_features
  use sklearn, only: dp,bernoulli_rbm
  implicit none
  type(bernoulli_rbm) :: model
  real(dp) :: x(6,4)
  real(dp), allocatable :: features(:,:),scores(:),draws(:,:)
  integer :: i
  x(1,:)=[0._dp,0._dp,1._dp,1._dp]; x(2,:)=x(1,:); x(3,:)=[0._dp,1._dp,1._dp,0._dp]
  x(4,:)=[1._dp,1._dp,0._dp,0._dp]; x(5,:)=x(4,:); x(6,:)=[1._dp,0._dp,0._dp,1._dp]
  model%n_components=2; model%batch_size=3; model%n_iter=100; model%random_state=42
  features=model%fit_transform(x); scores=model%score_samples(x); draws=model%gibbs(x)
  print *, 'Hidden activation probabilities:'
  do i=1,size(x,1)
    print '(2f10.6)',features(i,:)
  end do
  print *, 'Mean pseudo-likelihood: ',sum(scores)/size(scores)
  print *, 'Active visible units in Gibbs draw: ',count(draws==1)
  call model%partial_fit(x)
  print *, 'Updated weight norm: ',sqrt(sum(model%components_**2))
end program
