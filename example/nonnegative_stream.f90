program nonnegative_stream
  use sklearn, only: dp,mini_batch_nmf
  implicit none
  type(mini_batch_nmf) :: model
  real(dp) :: x(12,5)
  real(dp), allocatable :: factors(:,:),reconstruction(:,:)
  integer :: i,j
  do j=1,5
    do i=1,12
      x(i,j)=.2_dp+abs(sin(.3_dp*i)*cos(.2_dp*j))+real(mod(i+j,3),dp)/5
    end do
  end do
  model%n_components=2; model%init='nndsvdar'; model%max_iter=40
  model%batch_size=4; model%random_state=42; model%beta_loss=1
  factors=model%fit_transform(x); reconstruction=model%inverse_transform(factors)
  print *, 'NMF epochs/updates: ',model%n_iter_,model%n_steps_
  print *, 'KL reconstruction error: ',model%reconstruction_err_
  print *, 'Frobenius reconstruction RMSE: ',sqrt(sum((x-reconstruction)**2)/size(x))
  call model%partial_fit(x(:6,:)); factors=model%transform(x)
  print *, 'Updates after streaming: ',model%n_steps_
  print *, 'Minimum component/factor values: ',minval(model%components_),minval(factors)
end program
