program incremental_mlp
  use sklearn_neural_network, only: mlp_classifier
  use sklearn_common, only: dp
  implicit none
  type(mlp_classifier) :: model
  real(dp) :: x(6,2)
  integer :: y(6),pass,first,last,calls
  x(:,1)=[-2._dp,-1._dp,-1._dp,1._dp,1._dp,2._dp]
  x(:,2)=[-1._dp,-2._dp,-1._dp,1._dp,2._dp,1._dp];y=[0,0,0,1,1,1]
  model%hidden_layer_sizes=[integer::];model%solver='sgd';model%learning_rate_init=.1_dp
  model%momentum=.8_dp;model%tol=0;calls=0
  do pass=1,60
    do first=1,6,2
      last=first+1
      if(calls==0) then
        call model%partial_fit(x(first:last,:),y(first:last),classes=[0,1])
      else
        call model%partial_fit(x(first:last,:),y(first:last))
      end if
      calls=calls+1
    end do
  end do
  print *, 'Predicted labels:',model%predict(x)
  print *, 'Partial-fit calls and retained epochs:',calls,size(model%loss_curve_)
  print *, 'Last-call epochs and total samples:',model%n_iter_,model%t_
  print *, 'Training accuracy:',model%score(x,y)
end program
