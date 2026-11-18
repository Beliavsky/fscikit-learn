program independent_components
  use sklearn
  implicit none
  type(fast_ica) :: model
  real(dp) :: x(1000,2),t,a,b
  real(dp), allocatable :: sources(:,:),reconstructed(:,:)
  integer :: i
  do i=1,1000
    t=real(i,dp)/1000; a=sin(31*t); b=merge(1._dp,-1._dp,sin(73*t)>0)
    x(i,:)=[a+0.5_dp*b+3,0.2_dp*a+b-1]
  end do
  model%random_state=42; model%tol=1.e-8_dp
  sources=model%fit_transform(x); reconstructed=model%inverse_transform(sources)
  print *, 'Converged: ',model%converged_, ' iterations: ',model%n_iter_
  print *, 'Source variances: ',sum(sources**2,dim=1)/size(x,1)
  print *, 'Maximum reconstruction error: ',maxval(abs(reconstructed-x))
end program
