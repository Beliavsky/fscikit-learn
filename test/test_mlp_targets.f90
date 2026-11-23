program test_mlp_targets
  use sklearn_neural_network, only: mlp_regressor,mlp_classifier
  use sklearn_common, only: dp,require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
  implicit none
  type(mlp_regressor) :: r
  type(mlp_classifier) :: c
  real(dp) :: x(2,1),y(2,2)
  real(dp), allocatable :: prediction(:,:),probability(:,:)
  integer :: labels(2,2)
  x(:,1)=[1._dp,2._dp];y(:,1)=[1._dp,2._dp];y(:,2)=[-1._dp,0._dp]
  r%hidden_layer_sizes=[integer::];r%solver='sgd';r%momentum=0;r%nesterovs_momentum=.false.
  r%shuffle=.false.;r%max_iter=1;r%learning_rate_init=.1_dp;r%alpha=0;r%tol=0
  call r%fit(x,y);r%layers_(1)%weights=0;r%layers_(1)%bias=0;r%warm_start=.true.
  call r%fit(x,y)
  call require(abs(r%loss_-.75_dp)<1.e-14_dp,'multioutput MLP loss averages over responses')
  call require(maxval(abs(r%layers_(1)%weights(1,:)-[.25_dp,-.05_dp]))<1.e-14_dp, &
    'multioutput MLP gradients retain sklearn sample-only normalization')
  call require(maxval(abs(r%layers_(1)%bias-[.15_dp,-.05_dp]))<1.e-14_dp,'multioutput MLP intercept gradient')
  prediction=r%predict_multioutput(x)
  call require(all(shape(prediction)==[2,2]).and.r%n_outputs_==2,'joint MLP response shape')
  call require(abs(r%score(x,y)+2.015_dp)<1.e-12_dp,'multioutput MLP uniform-average R2')
  call require(ieee_is_nan(r%score(x(:1,:),y(:1,:))),'single-sample MLP R2 is undefined')
  r%warm_start=.false.;call r%fit(x,y(:,1:1));prediction=r%predict_multioutput(x)
  call require(size(prediction,2)==1.and.all(prediction(:,1)==r%predict(x)),'one-column response compatibility')
  labels=reshape([1,0,0,1],[2,2])
  c%hidden_layer_sizes=[integer::];c%solver='sgd';c%momentum=0;c%nesterovs_momentum=.false.
  c%shuffle=.false.;c%max_iter=1;c%learning_rate_init=.1_dp;c%alpha=0;c%tol=0
  call c%fit(x,labels);c%layers_(1)%weights=0;c%layers_(1)%bias=0;c%warm_start=.true.
  call c%fit(x,labels)
  call require(abs(c%loss_-2*log(2._dp))<1.e-14_dp,'multilabel MLP sums binary cross-entropies')
  call require(maxval(abs(c%layers_(1)%weights(1,:)-[-.025_dp,.025_dp]))<1.e-14_dp,'independent sigmoid gradients')
  call require(c%out_activation_=='logistic'.and.c%multilabel_.and.all(c%classes_==[0,1]),'multilabel metadata')
  c%layers_(1)%weights=0;c%layers_(1)%bias=[1._dp,2._dp];probability=c%predict_proba(x)
  call require(all(sum(probability,dim=2)>1.5_dp),'multilabel probabilities are independent')
  call require(all(c%predict_indicator(x)==1).and.c%score(x,labels)==0,'multilabel exact-row accuracy')
  c%layers_(1)%bias=0
  call require(all(c%predict_indicator(x)==0),'multilabel threshold ties are negative')
  c%warm_start=.false.;call c%fit(x,reshape([0,1],[2,1]))
  call require(.not.c%multilabel_.and.size(c%predict_proba(x),2)==2,'one-column classification delegates to binary')
  print *, 'MLP joint responses, indicators and target scores: PASS'
end program
