program variational_mixture
  use sklearn, only: dp,bayesian_gaussian_mixture
  use sklearn_legacy_rng, only: legacy_random_state
  implicit none
  type(bayesian_gaussian_mixture) :: model
  type(legacy_random_state) :: rng
  real(dp) :: x(60,2),query(3,2)
  real(dp), allocatable :: probabilities(:,:)
  integer :: i
  call rng%seed(42)
  do i=1,60
    x(i,:)=[rng%normal(),rng%normal()]*.5_dp
    if(i<=30) then
      x(i,1)=x(i,1)-4
    else
      x(i,:)=x(i,:)+[4._dp,1._dp]
    end if
  end do
  model%n_components=6; model%weight_concentration_prior=.1_dp
  model%max_iter=500; model%tol=1.e-5_dp; model%random_state=42
  call model%fit(x)
  query(:,1)=[-4._dp,4._dp,0._dp]; query(:,2)=[0._dp,1._dp,.5_dp]
  probabilities=model%predict_proba(query)
  print *, 'Posterior component weights: ',model%weights_
  print *, 'Components with weight above 0.05: ',count(model%weights_>.05_dp)
  print *, 'Query labels: ',model%predict(query)
  print *, 'Probability row sums: ',sum(probabilities,dim=2)
  print *, 'Iterations/converged: ',model%n_iter_,model%converged_
end program
