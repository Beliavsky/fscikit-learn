program test_rbm
  use sklearn_rbm, only: bernoulli_rbm
  use sklearn_common, only: dp,require,sigmoid
  implicit none
  type(bernoulli_rbm) :: model,copy,stream,single
  real(dp) :: x(4,3),before(4,3),initial,conditional
  real(dp), allocatable :: h(:,:),scores(:),draws(:,:),other(:,:),weights(:,:)
  x=reshape([0._dp,0._dp,1._dp,1._dp,0._dp,1._dp,0._dp,1._dp,0._dp,1._dp,1._dp,1._dp],[4,3])
  before=x; model%n_components=2; model%batch_size=3; model%n_iter=0
  call model%fit(x)
  call require(all(shape(model%components_)==[2,3]).and.all(shape(model%h_samples_)==[3,2]),'RBM parameter shapes')
  call require(all(model%h_samples_==0),'RBM initial persistent chain')
  model%components_=0; model%intercept_hidden_=[2._dp,-2._dp]; model%intercept_visible_=0
  h=model%transform(x)
  call require(maxval(abs(h(:,1)-sigmoid(2._dp)))<1.e-14_dp,'RBM analytic hidden probability')
  call require(maxval(abs(h(:,2)-sigmoid(-2._dp)))<1.e-14_dp,'RBM negative hidden logit')
  scores=model%score_samples(x)
  call require(maxval(abs(scores+3*log(2._dp)))<1.e-13_dp,'RBM uniform visible pseudo-likelihood')
  model%intercept_hidden_=[1000._dp,-1000._dp]; h=model%transform(x)
  scores=model%score_samples(x)
  call require(all(h(:,1)==1).and.all(h(:,2)==0),'RBM stable saturated sigmoid')
  call require(maxval(abs(scores+3*log(2._dp)))<1.e-12_dp,'RBM stable saturated free energy')
  copy=model; draws=model%gibbs(x); other=copy%gibbs(x)
  call require(all(draws==other).and.all(draws==0.or.draws==1),'RBM copied sampling stream')
  model%n_iter=4; call model%fit(x); copy=model; weights=model%components_
  call require(any(model%h_samples_==1),'RBM trained persistent samples')
  h=model%transform(x); other=model%fit_transform(x)
  call require(maxval(abs(h-other))<1.e-14_dp.and.all(weights==model%components_),'RBM fresh fit reproducibility')
  call require(all(before==x),'RBM preserves input')
  stream%n_components=2; stream%batch_size=3
  call stream%partial_fit(x(:2,:)); copy=stream
  call stream%partial_fit(x(3:,:)); call copy%partial_fit(x(3:,:))
  call require(all(stream%components_==copy%components_),'RBM persistent stream copy continuation')
  call require(all(stream%h_samples_==0.or.stream%h_samples_==1),'RBM persistent chain binary')
  call require(any(abs(stream%components_)>0),'RBM incremental weight update')
  model%n_components=3; call model%fit(x(:,:1))
  call require(all(shape(model%components_)==[3,1]).and.model%n_features_in_==1,'RBM fresh refit dimensions')
  single%n_components=1; single%batch_size=1; single%n_iter=0; single%random_state=1
  call single%fit(0*x(:1,:1)); initial=single%components_(1,1); conditional=sigmoid(initial)
  call single%partial_fit(0*x(:1,:1))
  call require(abs(single%components_(1,1)-(initial-.1_dp*conditional))<1.e-14_dp,'RBM analytic PCD weight update')
  call require(abs(single%intercept_hidden_(1)-.1_dp*(.5_dp-conditional))<1.e-14_dp,'RBM analytic PCD hidden update')
  call require(abs(single%intercept_visible_(1)+.1_dp)<1.e-14_dp,'RBM analytic PCD visible update')
  call require(single%h_samples_(1,1)==0,'RBM partial_fit starts its own seeded stream after fit')
  print *, 'BernoulliRBM tests passed'
end program
