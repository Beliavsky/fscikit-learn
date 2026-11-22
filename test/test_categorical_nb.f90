program test_categorical_nb
  use sklearn_categorical_nb, only: categorical_nb
  use sklearn_common, only: dp
  use sklearn_common, only: require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan
  implicit none
  type(categorical_nb) :: batch,stream,single
  real(dp) :: x(4,2),w(4),query(1,2)
  real(dp), allocatable :: probability(:,:),logs(:,:)
  integer :: y(4)
  x(1,:)=[0._dp,0._dp]; x(2,:)=[1._dp,0._dp]
  x(3,:)=[1._dp,1._dp]; x(4,:)=[1._dp,1._dp]
  y=[-3,-3,9,9]; w=[1._dp,2._dp,3._dp,0._dp]
  call batch%fit(x,y,sample_weight=w)
  call require(all(batch%class_count_==3),'weighted class counts')
  call require(all(batch%category_count_(1)%values(1,:)==[1._dp,2._dp]),'weighted categorical counts')
  call require(maxval(abs(exp(batch%feature_log_prob_(1)%values(1,:))-[.4_dp,.6_dp]))<1.e-14_dp, &
    'Laplace category smoothing')
  query(1,:)=[0._dp,1._dp]; probability=batch%predict_proba(query)
  call require(maxval(abs(probability(1,:)-[1._dp/3,2._dp/3]))<1.e-14_dp,'analytic categorical posterior')
  call stream%partial_fit(x(:2,:),y(:2),classes=[9,-3],sample_weight=w(:2))
  probability=stream%predict_proba(x(:2,:)); logs=stream%predict_log_proba(x(:2,:))
  call require(all(probability(:,2)==0).and.all(.not.ieee_is_finite(logs(:,2))), &
    'unobserved class has zero empirical prior without invalid logarithms')
  call stream%partial_fit(x(3:,:),y(3:),sample_weight=w(3:))
  call require(all(stream%n_categories_==2),'incremental category expansion')
  call require(maxval(abs(stream%predict_proba(x)-batch%predict_proba(x)))<1.e-14_dp,'partial and batch agreement')
  call stream%partial_fit(x(:1,:),y(:1),sample_weight=[0._dp])
  call require(all(stream%n_categories_==1),'latest batch category metadata matches source')
  call require(size(stream%category_count_(1)%values,2)==2,'small batch retains accumulated category tables')
  batch%min_categories=3; call batch%fit(x,y); query=2
  probability=batch%predict_proba(query)
  call require(abs(sum(probability)-1._dp)<1.e-14_dp,'reserved unseen categories have smoothed probabilities')
  batch%class_prior=[1._dp,0._dp]; call batch%fit(x,y)
  call require(all(batch%predict(x)==-3),'explicit zero prior suppresses class')
  call single%fit(x,[5,5,5,5]); probability=single%predict_proba(x)
  call require(all(probability==1).and.single%score(x,[5,5,5,5])==1,'single-class categorical fit')
  batch%class_prior=[.5_dp,.5_dp]; call batch%fit(x,y,sample_weight=[1._dp,1._dp,0._dp,0._dp])
  call require(all(batch%class_count_==[2._dp,0._dp]),'zero-weight class counts retained')
  x=0; batch%class_prior=[.5_dp,.5_dp]; call batch%fit(x,y)
  call require(all(batch%predict(x)==-3),'exact posterior ties choose first sorted class')
  batch=categorical_nb(); batch%alpha=0; batch%min_categories=3
  x(1,:)=[0._dp,0._dp]; x(2,:)=[1._dp,0._dp]
  x(3,:)=[1._dp,1._dp]; x(4,:)=[1._dp,1._dp]
  batch%class_prior=[.2_dp,.2_dp]; call batch%fit(x,y)
  call require(maxval(abs(batch%class_log_prior_-log(.2_dp)))<1.e-14_dp,'nonunit priors retained')
  query=2; probability=batch%predict_proba(query)
  call require(all(ieee_is_nan(probability)),'unsmoothed impossible observation remains undefined')
  call require(all(batch%predict(query)==-3),'all negative infinite scores choose first class')
  batch%force_alpha=.false.; call batch%fit(x,y)
  probability=batch%predict_proba(query)
  call require(all(ieee_is_finite(probability)),'force_alpha false floors zero smoothing')
  call require(abs(sum(probability)-1)<1.e-13_dp,'floored smoothing normalization')
  batch=categorical_nb(); call batch%fit(x,y,sample_weight=0*w)
  probability=batch%predict_proba(x)
  call require(all(ieee_is_nan(probability)),'zero empirical mass is undefined')
  batch%fit_prior=.false.; call batch%fit(x,y,sample_weight=0*w)
  probability=batch%predict_proba(x)
  call require(maxval(abs(probability-.5_dp))<1.e-14_dp,'zero weights with uniform priors')
  batch%class_prior=[0._dp,0._dp]; call batch%fit(x,y)
  probability=batch%predict_proba(x)
  call require(all(ieee_is_nan(probability)),'all-zero fixed priors remain undefined')
  stream=categorical_nb(); stream%alpha=0
  call stream%partial_fit(x(:2,:),y(:2),classes=[9,-3])
  probability=stream%predict_proba(x(:2,:))
  call require(all(ieee_is_nan(probability)),'unsmoothed missing class is undefined')
  call require(all(stream%predict(x(:2,:))==9),'first NaN score follows source argmax')
  single=stream; call single%partial_fit(x(3:,:),y(3:))
  call require(all(stream%class_count_==[2._dp,0._dp]),'copy update preserves original state')
  call single%fit(x(:,:1),[5,5,5,5]); call require(single%n_features_in_==1,'fit resets feature width')
  print *, 'CategoricalNB tests passed'
end program
