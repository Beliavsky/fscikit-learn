program test_dummy_extended
  use sklearn_common,only:dp,require
  use sklearn_dummy
  use sklearn_sparse,only:csr_matrix,csr_from_arrays
  use sklearn_base_dispatch,only:base_weighted,train_classifier,train_regressor
  use sklearn_voting,only:voting_classifier,voting_regressor,voting_estimator
  use,intrinsic::ieee_arithmetic
  use,intrinsic::ieee_exceptions
  implicit none
  type(dummy_classifier)::classifier,saved
  type(dummy_regressor)::regressor,copy
  type(dummy_probability),allocatable::proba(:)
  type(csr_matrix)::wide,empty
  type(voting_classifier)::vote
  type(voting_regressor)::vote_reg
  type(voting_estimator)::bases(2)
  real(dp)::x(4,0),query(8,2),y(4),w(4),targets(4,2)
  real(dp)::features(4,1)
  real(dp)::ties_x(12,0)
  real(dp),allocatable::pred(:,:)
  integer::labels(4),binary(4,2),state_size
  integer,allocatable::before_rng(:),after_rng(:)
  logical::flags(size(ieee_all)),modes(size(ieee_all)),after(size(ieee_all)),after_modes(size(ieee_all))
  query=ieee_value(0._dp,ieee_quiet_nan);y=[1._dp,2._dp,3._dp,4._dp]
  w=[0._dp,1._dp,3._dp,2._dp];labels=[-7,4,4,19]
  call classifier%fit(x,labels,w)
  call require(all(classifier%classes_==[-7,4,19]),'zero-weight classes are retained')
  call require(maxval(abs(classifier%class_prior_-[0._dp,2._dp/3,1._dp/3]))<1.e-15_dp,'weighted class priors')
  call require(classifier%n_features_in_==0.and.classifier%n_outputs_==1,'dummy feature metadata')
  saved=classifier;classifier%strategy='uniform'
  call require(all(classifier%predict(query)==4),'classifier freezes strategy at fit')
  classifier%random_state=7;call classifier%fit(x,labels,w)
  call require(all(classifier%predict(query)==[-7,4,19,-7,4,19,-7,4]),'NumPy seeded uniform labels')
  classifier%strategy='stratified';call classifier%fit(x,labels,w)
  call require(all(classifier%predict(query)==[4,19,4,19,19,4,4,4]),'NumPy seeded multinomial labels')
  call random_seed(size=state_size);allocate(before_rng(state_size),after_rng(state_size))
  call random_seed(get=before_rng);proba=classifier%predict_proba_multi(query);call random_seed(get=after_rng)
  call require(all(before_rng==after_rng),'dummy private RNG leaves Fortran global RNG unchanged')
  call require(all(saved%class_prior_==[0._dp,2._dp/3,1._dp/3]),'classifier copies own distributions')
  regressor%strategy='median';call regressor%fit(x,y)
  call require(regressor%constant_==2.5_dp,'ordinary median interpolates')
  call regressor%fit(x,y,[1._dp,1._dp,1._dp,1._dp])
  call require(regressor%constant_==2,'weighted median selects lower observation')
  regressor%strategy='quantile';regressor%quantile=0
  call regressor%fit(x,y,w);call require(regressor%constant_==2,'weighted zero quantile skips zero weights')
  call regressor%fit(x,y,[0._dp,0._dp,0._dp,0._dp])
  call require(regressor%constant_==4,'all-zero weighted percentile source contract')
  regressor%strategy='mean';call regressor%fit(x,y,[-1._dp,2._dp,0._dp,1._dp])
  call require(regressor%constant_==3.5_dp,'signed weighted mean')
  targets(:,1)=y;targets(:,2)=-2*y
  call regressor%fit(x,targets,w);copy=regressor;pred=regressor%predict_multi(query)
  call require(all(shape(pred)==[8,2]).and.maxval(abs(pred(:,2)+2*pred(:,1)))==0,'joint multioutput constants')
  regressor%constant_values=[19._dp,-7._dp];regressor%strategy='constant'
  call regressor%fit(x,targets);regressor%constant_values=0
  pred=regressor%predict_multi(query)
  call require(all(pred(:,1)==19).and.all(pred(:,2)==-7),'regression stores fitted constants')
  binary(:,1)=[0,1,1,0];binary(:,2)=[1,0,1,1]
  classifier%strategy='prior';call classifier%fit(x,binary,w)
  call require(size(classifier%predict_proba_multi(query))==2,'multiple probability heads')
  call require(classifier%score(x,binary,w)==.5_dp,'multioutput binary subset accuracy')
  call ieee_set_flag(ieee_all,.false.);call ieee_set_flag(ieee_underflow,.true.)
  call ieee_get_flag(ieee_all,flags);call ieee_get_halting_mode(ieee_all,modes)
  call classifier%fit(x,labels,[0._dp,0._dp,0._dp,0._dp])
  call ieee_get_flag(ieee_all,after);call ieee_get_halting_mode(ieee_all,after_modes)
  call require(all(flags.eqv.after).and.all(modes.eqv.after_modes),'dummy fit restores IEEE state')
  call require(all(ieee_is_nan(classifier%class_prior_)),'zero-total class priors are NaN')
  call require(all(classifier%predict(query)==-7),'NumPy first-NaN argmax')
  call require(base_weighted(classifier).and.base_weighted(regressor),'dummy weighted ensemble capabilities')
  call train_classifier(classifier,x,labels,w)
  call require(abs(classifier%class_prior_(2)-2._dp/3)<1.e-15_dp,'weighted classifier dispatch')
  regressor%strategy='mean';call train_regressor(regressor,x,y,w)
  call require(abs(regressor%constant_-19._dp/6)<1.e-15_dp,'weighted regressor dispatch')
  wide=csr_from_arrays(4,1000000000,[1,1,1,1,1],[integer::],[real(dp)::])
  empty=csr_from_arrays(0,0,[1],[integer::],[real(dp)::])
  call regressor%fit(wide,y,w)
  call require(regressor%n_features_in_==1000000000,'wide CSR dummy feature count')
  call require(size(regressor%predict(empty))==0,'empty queries ignore feature shape')
  features=0;classifier%strategy='prior'
  bases(1)%name='prior';bases(2)%name='frequent'
  allocate(bases(1)%estimator,source=classifier)
  classifier%strategy='most_frequent';allocate(bases(2)%estimator,source=classifier)
  vote%voting='soft';vote%weights=[1._dp,2._dp]
  call vote%fit(features,labels,bases,sample_weight=w)
  pred=vote%predict_proba(features)
  call require(maxval(abs(pred(1,:)-[0._dp,8._dp/9,1._dp/9]))<1.e-15_dp,'weighted dummy classifier voting')
  deallocate(bases(1)%estimator,bases(2)%estimator)
  regressor%strategy='mean';allocate(bases(1)%estimator,source=regressor)
  regressor%strategy='median';allocate(bases(2)%estimator,source=regressor)
  vote_reg%weights=[1._dp,2._dp];call vote_reg%fit(features,y,bases,sample_weight=w)
  call require(maxval(abs(vote_reg%predict(features)-55._dp/18))<1.e-15_dp,'weighted dummy regressor voting')
  regressor%strategy='quantile';regressor%quantile=.5_dp
  call regressor%fit(ties_x,[-1._dp,0._dp,1._dp,1._dp,0._dp,1._dp,-1._dp,0._dp,0._dp,-1._dp,0._dp,1._dp], &
    [1._dp,1._dp,1._dp,1._dp,0._dp,-1._dp,1._dp,-1._dp,0._dp,-1._dp,-1._dp,0._dp])
  call require(regressor%constant_==1,'NumPy tie ordering for signed weighted percentiles')
  print *,'Extended Dummy baselines passed; ignored one billion CSR features without densifying'
end program
