program test_nb_smoothing
  use sklearn_naive_bayes, only: multinomial_nb,bernoulli_nb,complement_nb
  use sklearn_common, only: dp,require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan,ieee_is_finite
  implicit none
  type(multinomial_nb) :: model,stream
  type(bernoulli_nb) :: bern
  type(complement_nb) :: comp
  real(dp) :: x(3,3),q(2,3),binary(8,2),query(2,2)
  real(dp), allocatable :: probability(:,:),joint(:,:),expected(:,:)
  x(1,:)=[2._dp,0._dp,1._dp];x(2,:)=[1._dp,0._dp,2._dp];x(3,:)=[0._dp,3._dp,0._dp]
  model%alpha_per_feature=[0._dp,.2_dp,1._dp]
  call model%fit(x,[-3,-3,9])
  q(1,:)=[1._dp,0._dp,1._dp];q(2,:)=[0._dp,1._dp,1._dp]
  probability=model%predict_proba(q)
  call require(probability(1,1)==1.and.probability(1,2)==0,'zero alpha suppresses absent events')
  call require(all(ieee_is_nan(probability(2,:))),'dense zero times infinite weight follows source NaN')
  call require(all(model%predict(q)==[-3,9]),'source NaN argmax and finite argmax')
  model%force_alpha=.false.;call model%fit(x,[-3,-3,9]);probability=model%predict_proba(q)
  call require(all(ieee_is_finite(probability)),'feature alpha floor restores finite probabilities')
  call require(all(model%alpha_per_feature==[0._dp,.2_dp,1._dp]),'floor preserves input vector')
  stream%alpha_per_feature=model%alpha_per_feature;stream%force_alpha=.false.
  call stream%partial_fit(x(:2,:),[-3,-3],classes=[9,-3]);call stream%partial_fit(x(3:,:),[9])
  call require(maxval(abs(stream%feature_log_prob_-model%feature_log_prob_))<1.e-13_dp,'streaming vector smoothing')
  stream%alpha_per_feature=[1._dp,2._dp,3._dp]
  call stream%partial_fit(x(:1,:),[-3],sample_weight=[0._dp])
  expected=log(stream%feature_count_+spread(stream%alpha_per_feature,1,2))
  expected=expected-spread(log(sum(stream%feature_count_+spread(stream%alpha_per_feature,1,2),dim=2)),2,3)
  call require(maxval(abs(stream%feature_log_prob_-expected))<1.e-13_dp,'updated alpha recomputes from raw counts')
  model=multinomial_nb();model%class_prior=[0._dp,0._dp];call model%fit(x,[-3,-3,9])
  probability=model%predict_proba(q)
  call require(all(ieee_is_nan(probability)).and.all(model%predict(q)==-3),'all-zero fixed priors retained')
  binary(1,:)=[0._dp,0._dp];binary(2,:)=[0._dp,1._dp]
  binary(3,:)=[0._dp,0._dp];binary(4,:)=[0._dp,1._dp]
  binary(5,:)=[0._dp,0._dp];binary(6,:)=[1._dp,1._dp]
  binary(7,:)=[0._dp,1._dp];binary(8,:)=[1._dp,0._dp]
  query(1,:)=[0._dp,1._dp];query(2,:)=[1._dp,0._dp]
  bern%alpha=0;call bern%fit(binary,[-3,-3,-3,-3,9,9,9,9]);probability=bern%predict_proba(query)
  call require(all(ieee_is_nan(probability(1,:))).and.all(ieee_is_finite(probability(2,:))), &
    'Bernoulli zero smoothing distinguishes zero and positive query events')
  call require(all(bern%predict(query)==[-3,9]),'Bernoulli dense source labels')
  bern%force_alpha=.false.;call bern%fit(binary,[-3,-3,-3,-3,9,9,9,9]);probability=bern%predict_proba(query)
  call require(all(ieee_is_finite(probability)).and.bern%alpha==0,'scalar floor preserves input alpha')
  comp%alpha=0;call comp%fit(x,[-3,-3,9],sample_weight=[0._dp,0._dp,1._dp])
  joint=comp%joint_log_likelihood(reshape([1._dp,1._dp,1._dp],[1,3]))
  call require(ieee_is_nan(joint(1,2)),'undefined Complement class remains NaN')
  comp%norm=.true.;call comp%fit(x,[-3,-3,9]);probability=comp%predict_proba(q)
  call require(all(ieee_is_nan(probability)),'normalized infinite complement weights retain source NaNs')
  print *, 'Discrete NB feature smoothing and zero boundaries: PASS'
end program
