program test_mlp_sparse
  use sklearn_neural_network, only: mlp_regressor,mlp_classifier
  use sklearn_sparse, only: csr_matrix,csr_from_dense,csr_from_arrays
  use sklearn_common, only: dp,require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  type(mlp_regressor) :: dense,sparse,wide
  type(mlp_classifier) :: c,d,indicator
  type(csr_matrix) :: a,duplicate,large
  real(dp) :: x(6,3),target(6,2)
  real(dp), allocatable :: pd(:,:),ps(:,:),prob(:,:),prediction(:)
  integer :: i,solver
  x=reshape([0._dp,1._dp,0._dp,-1._dp,0._dp,2._dp, &
             1._dp,0._dp,0._dp,2._dp,-1._dp,0._dp, &
             0._dp,0._dp,0._dp,1._dp,0._dp,0._dp],[6,3])
  target(:,1)=x(:,1)+.3_dp*x(:,2);target(:,2)=sin(x(:,1));a=csr_from_dense(x)
  do solver=1,3
    dense=mlp_regressor();dense%hidden_layer_sizes=[4,3];dense%activation='tanh'
    dense%max_iter=3;dense%batch_size=2;dense%random_state=7;dense%shuffle=.true.
    if(solver==2) dense%solver='sgd'
    if(solver==3) then
      dense%solver='lbfgs';dense%max_iter=1
    end if
    sparse=dense;call dense%fit(x,target);call sparse%fit(a,target)
    pd=dense%predict_multioutput(x);ps=sparse%predict_multioutput(a)
    call require(maxval(abs(pd-ps))<1.e-12_dp,'CSR MLP solvers agree with dense training')
    call require(abs(dense%score(x,target)-sparse%score(a,target))<1.e-12_dp,'CSR joint response scoring')
    dense%warm_start=.true.;sparse%warm_start=.true.;call dense%fit(a,target);call sparse%fit(x,target)
    pd=dense%predict_multioutput(a);ps=sparse%predict_multioutput(x)
    call require(maxval(abs(pd-ps))<1.e-11_dp,'MLP warm fit can change input storage')
  end do
  dense=mlp_regressor();dense%hidden_layer_sizes=[integer::];dense%batch_size=2;sparse=dense
  call dense%partial_fit(x,target(:,1));call sparse%partial_fit(a,target(:,1))
  call dense%partial_fit(a,target(:,1));call sparse%partial_fit(x,target(:,1))
  call require(all(dense%layers_(1)%weights==sparse%layers_(1)%weights), &
    'CSR partial fits share persistent optimizer state with dense calls')
  duplicate=csr_from_arrays(2,3,[1,5,5],[3,1,3,2],[.25_dp,2._dp,.75_dp,0._dp])
  c%hidden_layer_sizes=[3];c%max_iter=2;d=c
  call c%fit(duplicate,[0,1]);call d%fit(duplicate%to_dense(),reshape([0,1],[2,1]))
  prob=c%predict_proba(duplicate);pd=d%predict_proba(duplicate%to_dense())
  call require(maxval(abs(prob-pd))<1.e-12_dp,'duplicate, unsorted, zero and empty CSR entries')
  call require(size(c%predict_log_proba(duplicate),2)==2,'CSR classifier log probabilities')
  call require(c%score(duplicate,[0,1])==d%score(duplicate%to_dense(),[0,1]),'CSR vector classifier scoring')
  call d%fit(duplicate,reshape([0,1],[2,1]));pd=d%predict_proba(duplicate)
  call require(maxval(abs(prob-pd))<1.e-12_dp,'CSR one-column classifier fit adapter')
  call d%partial_fit(duplicate,reshape([0,1],[2,1]),classes=[0,1])
  call require(d%n_iter_==1,'CSR one-column classifier partial adapter')
  indicator%hidden_layer_sizes=[integer::];call indicator%partial_fit(a, &
    reshape([0,1,0,1,0,1,1,0,1,0,1,0],[6,2]),classes=[0,1])
  call require(size(indicator%predict_indicator(a),2)==2,'CSR multilabel predictions')
  call require(ieee_is_finite(indicator%score(a,reshape([0,1,0,1,0,1,1,0,1,0,1,0],[6,2]))), &
    'CSR multilabel scoring')
  ! A dense copy of this feature matrix would require 3.2 GB.
  large%n_rows=4000;large%n_cols=100000
  allocate(large%indptr(4001),large%indices(4000),large%data(4000))
  large%indptr=[(i,i=1,4001)];large%indices=100000;large%data=.1_dp
  wide%hidden_layer_sizes=[integer::];wide%solver='sgd';wide%max_iter=1
  call wide%fit(large,spread(.2_dp,1,4000));prediction=wide%predict(large)
  call require(wide%n_features_in_==100000.and.all(ieee_is_finite(prediction)), &
    'wide CSR MLP avoids allocating a dense input matrix')
  print *, 'CSR MLP kernels, solver transitions and wide sparse inputs: PASS'
end program
