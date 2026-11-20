program test_lda
  use sklearn_lda, only: latent_dirichlet_allocation
  use sklearn_common, only: dp,require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan,ieee_is_finite
  implicit none
  type(latent_dirichlet_allocation) :: model,copy,stream,zero_model,seed_model
  real(dp) :: x(4,3),before(4,3),counts(3),expected,one(2,1)
  real(dp), allocatable :: topics(:,:),raw(:,:),weights(:,:)
  x=reshape([2._dp,1._dp,0._dp,3._dp,0._dp,3._dp,2._dp,1._dp,1._dp,0._dp,4._dp,1._dp],[4,3])
  before=x; counts=sum(x,dim=1)
  model%n_components=1; model%doc_topic_prior=.7_dp; model%topic_word_prior=.3_dp; model%max_iter=2
  topics=model%fit_transform(x); raw=model%transform(x,normalize=.false.)
  call require(all(topics==1),'LDA one-topic normalized posterior')
  call require(maxval(abs(raw(:,1)-(sum(x,dim=2)+.7_dp)))<1.e-11_dp,'LDA analytic document posterior mass')
  call require(maxval(abs(model%components_(1,:)-(counts+.3_dp)))<1.e-11_dp,'LDA analytic topic posterior')
  expected=log_gamma(.9_dp)-log_gamma(sum(counts)+.9_dp)+sum(log_gamma(counts+.3_dp)-log_gamma(.3_dp))
  call require(abs(model%score(x)-expected)<1.e-10_dp,'LDA one-topic conjugate evidence bound')
  call require(abs(log(model%perplexity(x))+model%score(x)/sum(x))<1.e-13_dp,'LDA score/perplexity identity')
  call require(model%n_iter_==2.and.model%n_batch_iter_==3,'LDA batch update counters')
  call require(all(x==before),'LDA preserves input')
  model%max_iter=100; model%evaluate_every=1; model%perp_tol=1.e-6_dp
  call model%fit(x)
  call require(model%n_iter_==1.and.model%n_batch_iter_==3,'LDA source early-stop iteration counter')
  model=latent_dirichlet_allocation(); model%n_components=3; model%max_iter=0
  call model%fit(x); weights=model%components_; copy=model
  call require(model%n_iter_==0.and.model%n_batch_iter_==1,'LDA zero-iteration initialization')
  call require(all(weights>0).and.all(model%exp_dirichlet_component_>0),'LDA positive initialization')
  call model%fit(x)
  call require(all(model%components_==weights),'LDA deterministic fresh initialization')
  call stream%partial_fit(x(:1,:))
  call require(ieee_is_nan(stream%bound_),'LDA streaming has no fitted perplexity until fit')
  stream%n_components=10; copy=stream
  call stream%partial_fit(x(2:,:)); call copy%partial_fit(x(2:,:))
  call require(all(stream%components_==copy%components_),'LDA copied RNG stream continuation')
  call require(stream%n_iter_==0.and.stream%n_batch_iter_==3,'LDA partial fit counters')
  stream%learning_method='batch'; stream%batch_size=2
  call stream%partial_fit(x)
  call require(stream%n_batch_iter_==5,'LDA partial fit always uses online chunks')
  model%max_iter=3; model%learning_method='online'; model%batch_size=3; model%evaluate_every=-1
  call model%fit(x)
  call require(model%n_iter_==3.and.model%n_batch_iter_==7,'LDA online fit batch counters')
  topics=model%transform(0*x)
  call require(maxval(abs(topics-1._dp/3))<1.e-14_dp,'LDA empty document prior symmetry')
  model%max_doc_update_iter=0; raw=model%transform(x,normalize=.false.)
  call require(all(raw==1),'LDA zero document iterations returns deterministic initial values')
  model%max_doc_update_iter=100; model%n_components=2; model%max_iter=2; one=1
  call model%fit(one)
  call require(all(shape(model%components_)==[2,1]).and.model%n_features_in_==1,'LDA fresh refit dimensions')
  call require(ieee_is_finite(model%bound_),'LDA one-feature fitted perplexity')
  seed_model%n_components=2; seed_model%max_iter=0
  call seed_model%fit(x)
  call require(maxval(abs(seed_model%components_(1,:)- &
    [1.1833542379999842_dp,1.0371517687346199_dp,1.194978503884998_dp]))<1.e-14_dp,'LDA NumPy Gamma initialization row one')
  call require(maxval(abs(seed_model%components_(2,:)- &
    [.90225084075185669_dp,.9863974719624401_dp,1.0382225651950758_dp]))<1.e-14_dp,'LDA NumPy Gamma initialization row two')
  zero_model%n_components=3; zero_model%max_iter=3; zero_model%evaluate_every=1
  call zero_model%fit(0*x)
  call require(ieee_is_nan(zero_model%bound_).and.zero_model%score(0*x)==0,'LDA undefined zero-word perplexity')
  call require(zero_model%n_iter_==3,'LDA undefined perplexity does not trigger early stopping')
  call zero_model%partial_fit(0*x)
  call require(ieee_is_nan(zero_model%perplexity(0*x)),'LDA zero-evidence streaming roundoff')
  zero_model%doc_topic_prior=1.e-7_dp; zero_model%topic_word_prior=1.e-7_dp; zero_model%max_iter=1
  call zero_model%fit(0*x)
  expected=zero_model%perplexity(x)
  call require(.not.ieee_is_finite(expected).and..not.ieee_is_nan(expected), &
    'LDA overflowing perplexity returns infinity without a trap')
  print *, 'LatentDirichletAllocation tests passed'
end program
