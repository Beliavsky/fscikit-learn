program test_discrete_stream
  use sklearn_common, only: dp,require
  use sklearn_naive_bayes, only: multinomial_nb,bernoulli_nb,complement_nb
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
  implicit none
  type(multinomial_nb) :: stream,batch,copy
  type(bernoulli_nb) :: binary
  type(complement_nb) :: complement
  real(dp) :: x(6,3),weights(6)
  integer :: y(6)
  real(dp), allocatable :: prob(:,:)
  x=reshape([1._dp,2._dp,0._dp,1._dp,3._dp,1._dp,0._dp,1._dp,2._dp,0._dp,2._dp,1._dp, &
    1._dp,0._dp,1._dp,2._dp,3._dp,0._dp],[6,3]);y=[-2,-2,3,3,8,8];weights=[.5_dp,2._dp,1._dp,0._dp,3._dp,.2_dp]
  call stream%partial_fit(x(:2,:),y(:2),[8,-2,3],weights(:2))
  call require(all(stream%class_count_==[2.5_dp,0._dp,0._dp]),'streaming NB initializes missing classes')
  call stream%partial_fit(x(3:,:),y(3:),sample_weight=weights(3:));call batch%fit(x,y,weights)
  call require(maxval(abs(stream%feature_count_-batch%feature_count_))<1.e-14_dp,'weighted streaming NB event counts')
  call require(maxval(abs(stream%predict_proba(x)-batch%predict_proba(x)))<1.e-13_dp,'weighted streaming equals batch NB')
  copy=stream;call copy%partial_fit(x(:1,:),y(:1))
  call require(copy%class_count_(1)==stream%class_count_(1)+1,'copied NB state advances independently')
  call stream%fit(x(:2,:2),y(:2))
  call require(size(stream%classes_)==1.and.stream%n_features_in_==2,'batch NB fit resets streaming state')
  binary%binarize_threshold=.5_dp;call binary%partial_fit(x,y,[-2,3,8],weights)
  call require(all(binary%feature_count_<=spread(binary%class_count_,2,3)),'Bernoulli NB counts weighted threshold events')
  complement%norm=.true.;call complement%fit(x(:,:1),y)
  prob=complement%predict_proba(x(:,:1))
  call require(all(ieee_is_nan(prob)),'one-feature normalized ComplementNB preserves source undefined probabilities')
  call batch%fit(x,y,0*weights);prob=batch%predict_proba(x)
  call require(all(ieee_is_nan(prob)),'zero-weight empirical NB priors remain explicitly undefined')
  batch%fit_prior=.false.;call batch%fit(x,y,0*weights);prob=batch%predict_proba(x)
  call require(maxval(abs(sum(prob,dim=2)-1))<1.e-12_dp,'zero-weight NB with uniform priors is defined')
  x=1;weights=1.e20_dp;call binary%fit(x,y,weights);prob=binary%predict_proba(x)
  call require(all(ieee_is_nan(prob)),'saturated Bernoulli probabilities preserve source undefined likelihoods safely')
  print *, 'Weighted discrete NB streaming: PASS'
end program
