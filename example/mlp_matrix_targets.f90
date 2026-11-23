program mlp_matrix_targets
  use sklearn_neural_network, only: mlp_classifier,mlp_regressor
  use sklearn_common, only: dp
  implicit none
  type(mlp_regressor) :: regression
  type(mlp_classifier) :: classification
  real(dp) :: x(6,2),responses(6,2)
  real(dp), allocatable :: prediction(:,:),probability(:,:)
  integer :: labels(6,2)
  integer, allocatable :: indicator(:,:)
  x(:,1)=[-1._dp,1._dp,0._dp,0._dp,1._dp,-1._dp]
  x(:,2)=[0._dp,0._dp,1._dp,-1._dp,1._dp,-1._dp]
  responses(:,1)=1+2*x(:,1)-x(:,2);responses(:,2)=-2-x(:,1)+.5_dp*x(:,2)
  regression%hidden_layer_sizes=[integer::];regression%solver='lbfgs';regression%tol=1.e-8_dp
  call regression%fit(x,responses)
  prediction=regression%predict_multioutput(x)
  labels=merge(1,0,x>0)
  classification%hidden_layer_sizes=[integer::];classification%solver='lbfgs';classification%tol=1.e-8_dp
  call classification%fit(x,labels)
  indicator=classification%predict_indicator(x);probability=classification%predict_proba(x)
  print *, 'Joint response shape:',shape(prediction)
  print *, 'Uniform-average R2:',regression%score(x,responses)
  print *, 'Indicator prediction shape:',shape(indicator)
  print *, 'Exact-row accuracy:',classification%score(x,labels)
  print *, 'Overlapping-label probabilities:',probability(5,:)
end program
