program test_centroid_priors
  use sklearn_neighbors, only: nearest_centroid
  use sklearn_common, only: dp,require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan,ieee_is_finite
  implicit none
  type(nearest_centroid) :: model,copy
  real(dp) :: x(5,1),q(1,1),anisotropic(4,2),point(1,2)
  real(dp), allocatable :: probability(:,:),decision(:,:)
  x(:,1)=[-2._dp,-1._dp,0._dp,2._dp,3._dp];q=.5_dp
  call model%fit(x,[-3,-3,-3,9,9])
  call require(all(model%predict(q)==-3),'uniform raw-distance prediction')
  model%priors=[1._dp,19._dp];call model%fit(x,[-3,-3,-3,9,9])
  call require(maxval(abs(model%class_prior_-[.05_dp,.95_dp]))<1.e-14_dp,'normalized custom priors')
  call require(all(model%priors==[1._dp,19._dp]),'input priors preserved')
  call require(all(model%predict(q)==9),'nonuniform prior changes prediction')
  decision=model%decision_function(q);probability=model%predict_proba(q)
  call require(all(shape(decision)==[1,1]),'binary decision shape')
  call require(abs(log(probability(1,2)/probability(1,1))-decision(1,1))<1.e-12_dp,'binary log odds')
  call require(maxval(abs(exp(model%predict_log_proba(q))-probability))<1.e-14_dp,'log probabilities')
  copy=model;copy%priors=[1._dp,0._dp];call copy%fit(x,[-3,-3,-3,9,9])
  call require(all(copy%predict(q)==-3).and.all(model%predict(q)==9),'copied state independent')
  probability=copy%predict_proba(q)
  call require(probability(1,2)==0,'zero prior suppresses class')
  model=nearest_centroid();model%prior_mode='empirical';call model%fit(x,[-3,-3,-3,9,9])
  call require(maxval(abs(model%class_prior_-[.6_dp,.4_dp]))<1.e-14_dp,'empirical priors')
  model%priors=[.5_dp,.5000001_dp];call model%fit(x,[-3,-3,-3,9,9])
  call require(sum(model%class_prior_)>1,'source isclose retains near-normalized priors')
  model%priors=[0._dp,0._dp];call model%fit(x,[-3,-3,-3,9,9])
  probability=model%predict_proba(q)
  call require(all(ieee_is_nan(probability)).and.all(model%predict(q)==-3),'zero-total priors source NaNs')
  anisotropic(1,:)=[-.1_dp,-10._dp];anisotropic(2,:)=[.1_dp,10._dp]
  anisotropic(3,:)=[1.9_dp,-8._dp];anisotropic(4,:)=[2.1_dp,12._dp];point(1,:)=[0._dp,3._dp]
  model=nearest_centroid();call model%fit(anisotropic,[-3,-3,9,9])
  probability=model%predict_proba(point)
  call require(all(model%predict(point)==9).and.probability(1,1)>probability(1,2), &
    'source uniform raw prediction differs from standardized probability argmax')
  x(:,1)=[-1._dp,-1._dp,2._dp,2._dp,2._dp]
  call model%fit(x,[-3,-3,9,9,9]);probability=model%predict_proba(q)
  call require(all(model%within_class_std_dev_==0).and.all(ieee_is_finite(probability)), &
    'zero pooled variance uses unscaled features in scores')
  call require(all(.not.ieee_is_finite(model%deviations_)),'source infinite deviations retained')
  model%shrink_threshold=1;call model%fit(x,[-3,-3,9,9,9])
  call require(all(ieee_is_nan(model%centroids_)),'zero times infinite shrunken deviation is undefined')
  model=nearest_centroid();call model%fit(x([1,3],:),[-3,9])
  call require(all(ieee_is_nan(model%within_class_std_dev_)),'singleton classes retain undefined pooled variance')
  call require(all(model%predict(q)==-3),'singleton uniform prediction remains defined')
  print *, 'NearestCentroid priors and discriminants: PASS'
end program
