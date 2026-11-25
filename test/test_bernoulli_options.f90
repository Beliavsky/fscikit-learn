program test_bernoulli_options
  use sklearn_common,only:dp,require
  use sklearn_naive_bayes,only:bernoulli_nb
  use sklearn_sparse,only:csr_matrix,csr_from_dense,csr_from_arrays
  use,intrinsic::ieee_arithmetic
  implicit none
  type(bernoulli_nb)::model,saved,defaults
  type(csr_matrix)::sparse,query_sparse
  real(dp)::x(3,2),query(3,2),ones(2,1),queries(3,1)
  real(dp),allocatable::joint(:,:),probability(:,:)
  integer::y(3)
  model%binarize=.false.;model%binarize_threshold=ieee_value(0._dp,ieee_quiet_nan)
  x(:,1)=[.25_dp,.75_dp,-.2_dp];x(:,2)=[.5_dp,.1_dp,.4_dp];y=[-7,-7,4]
  query=x
  call model%fit(x,y)
  call require(maxval(abs(model%feature_count_(1,:)-[1._dp,.6_dp]))<1.e-15_dp,'raw feature counts')
  call require(maxval(abs(exp(model%feature_log_prob_(1,:))-[.5_dp,.4_dp]))<1.e-15_dp,'raw conditional probabilities')
  saved=model;sparse=csr_from_dense(x)
  call model%fit(sparse,y)
  call require(maxval(abs(model%feature_log_prob_-saved%feature_log_prob_))<1.e-15_dp,'raw dense/CSR fit')
  joint=model%joint_log_likelihood(query);query_sparse=csr_from_dense(query)
  call require(maxval(abs(joint-model%joint_log_likelihood(query_sparse)))<1.e-14_dp,'raw dense/CSR likelihood')
  call defaults%fit(x,y)
  call require(any(defaults%feature_count_/=saved%feature_count_),'default binarization remains enabled')
  model%alpha=0;ones=1;queries(:,1)=[-1._dp,0._dp,1._dp]
  call model%fit(ones,[-7,-7]);joint=model%joint_log_likelihood(queries)
  call require(joint(1,1)==ieee_value(0._dp,ieee_negative_inf),'negative raw query times infinite log odds')
  call require(all(ieee_is_nan(joint(2:3,1))),'dense zero/unit query IEEE behavior')
  query_sparse=csr_from_dense(queries);joint=model%joint_log_likelihood(query_sparse)
  call require(all(joint(:2,1)==ieee_value(0._dp,ieee_negative_inf)),'structural zero skips infinite product')
  call require(ieee_is_nan(joint(3,1)),'stored unit query IEEE behavior')
  ones=0;call model%fit(ones,[-7,-7]);joint=model%joint_log_likelihood(queries)
  call require(joint(1,1)==ieee_value(0._dp,ieee_positive_inf),'negative raw query times negative infinite log odds')
  model%alpha=1;call model%fit(x,y,[0._dp,1._dp,2._dp])
  call require(maxval(abs(model%feature_count_(2,:)-[-.4_dp,.8_dp]))<1.e-15_dp,'signed weighted feature counts')
  sparse=csr_from_arrays(2,1,[1,3,5],[1,1,1,1],[.2_dp,.3_dp,-.1_dp,.4_dp])
  call model%fit(sparse,[-7,4])
  call require(maxval(abs(model%feature_count_(:,1)-[.5_dp,.3_dp]))<1.e-15_dp,'raw duplicate entries are retained')
  print *,'Bernoulli raw-feature modes and signed IEEE likelihoods passed'
end program
