program test_classification_forests
  use sklearn_common, only: dp,require
  use sklearn_ensemble, only: random_forest_classifier,extra_trees_classifier
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan,ieee_is_finite
  implicit none
  type(random_forest_classifier) :: forest
  type(extra_trees_classifier) :: extra
  real(dp) :: x(8,1),w(8)
  real(dp), allocatable :: probability(:,:),logs(:,:)
  integer :: y(8),i
  integer, allocatable :: leaves(:,:)
  x(:,1)=[(real(i,dp),i=1,8)];y=[-7,-7,-7,-7,9,9,42,42];w=1
  forest%n_estimators=3;forest%random_state=7;forest%max_depth=2;forest%criterion='entropy'
  call forest%fit(x,y,w)
  call require(all([(forest%estimators_(i)%random_state,i=1,3)]==[327741615,976413892,1202242073]),'source classifier forest seeds')
  call require(forest%n_classes_==3.and.forest%n_features_in_==1,'classifier forest learned dimensions')
  call require(all(forest%classes_==[-7,9,42]),'forest preserves original sorted labels')
  call require(all(forest%estimators_(1)%classes_==[0,1,2]),'component trees use source encoded class indices')
  probability=forest%predict_proba(x);call require(all(abs(sum(probability,dim=2)-1)<1e-12_dp),'forest class probabilities')
  leaves=forest%apply(x);call require(all(shape(leaves)==[8,3]),'classifier forest leaf matrix')
  forest%bootstrap=.false.;forest%class_weight='balanced';w(2)=0;w(3)=2
  call forest%fit(x,y,w)
  call require(abs(forest%estimators_(1)%nodes_(1)%weighted_n_samples-8)<1e-12_dp,'balanced factors multiply sample weights')
  allocate(forest%class_weight_values);forest%class_weight='none'
  forest%class_weight_values%labels=[-7,42];forest%class_weight_values%weights=[2._dp,0._dp]
  call forest%fit(x,y,w)
  call require(all(forest%classes_==[-7,9,42]),'zero-weight class remains in forest class space')
  extra%n_estimators=3;extra%criterion='log_loss';extra%class_weight='balanced_subsample'
  call extra%fit(x,y)
  call require(all(extra%predict(x)==y),'log-loss ExtraTrees interpolation')
  call require(extra%score(x,y,w)==1,'weighted classification forest accuracy')
  logs=extra%predict_log_proba(x)
  call require(count(.not.ieee_is_finite(logs))==16,'zero-probability classes have negative infinite logs')
  w=0;call extra%fit(x,y,w);probability=extra%predict_proba(x)
  call require(all(ieee_is_nan(probability)),'zero-weight forest preserves NaN probabilities')
  call require(all(extra%predict(x)==-7),'NaN class probabilities select the first source label')
  call require(all(extra%feature_importances_==0),'all-stump classifier importance is zero')
  print *, 'Classification forests, seeded weights, class balancing and probabilities: PASS'
end program
