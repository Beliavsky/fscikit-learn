program test_joint_forests
 use sklearn_common,only:dp,require
 use sklearn_ensemble,only:random_forest_regressor,extra_trees_regressor
 use sklearn_metrics,only:r2_score
 implicit none
 type(random_forest_regressor)::forest
 type(extra_trees_regressor)::extra
 real(dp)::x(12,2),y(12,2),w(12),expected(12,2)
 real(dp),allocatable::prediction(:,:),part(:,:),scalar(:)
 integer,allocatable::bags(:,:),again(:,:)
 integer::i,j,k,counts(12)
 do i=1,12
  x(i,:)=[real(i,dp),real(mod(i,3),dp)];y(i,:)=[real(i*i,dp),real(13-i,dp)]
 end do
 w=1;w(2)=0;w(4)=2
 forest%n_estimators=7;forest%random_state=7;forest%max_samples=8;forest%oob_score=.true.
 call forest%fit(x,y,w)
 prediction=forest%predict_multi(x);bags=forest%estimators_samples();again=bags
 call require(forest%n_outputs_==2.and.all(shape(prediction)==[12,2]),'joint forest output shape')
 call require(all(shape(bags)==[8,7]),'bootstrap sample matrix shape')
 expected=0;counts=0
 do j=1,7
  call require(forest%estimators_(j)%n_outputs_==2,'forest uses joint component trees')
  part=forest%estimators_(j)%predict_multi(x)
  do i=1,12
   if(any(bags(:,j)==i))cycle
   counts(i)=counts(i)+1;expected(i,:)=expected(i,:)+part(i,:)
  end do
 end do
 do i=1,12
  expected(i,:)=expected(i,:)/max(1,counts(i))
 end do
 call require(all(forest%oob_counts_==counts),'OOB coverage derived from bootstrap membership')
 call require(maxval(abs(forest%oob_prediction_multi_-expected))<1.e-12_dp,'OOB averages unsampled trees')
 call require(abs(forest%oob_score_-r2_score(y,expected))<1.e-12_dp,'OOB R2 ignores fit sample weights')
 bags=0;bags=forest%estimators_samples()
 call require(all(bags==again),'bootstrap samples are reproducible owned results')
 forest%oob_score=.false.;forest%max_samples=0;call forest%set_bootstrap(.false.)
 call forest%fit(x,y(:,1),w);scalar=forest%predict(x);prediction=forest%predict_multi(x)
 call require(forest%n_outputs_==1.and.all(scalar==prediction(:,1)),'scalar forest refit and matrix adapter')
 call require(allocated(forest%oob_counts_).and.allocated(forest%oob_prediction_multi_), &
  'non-OOB refit retains cached OOB storage')
 bags=forest%estimators_samples()
 do j=1,7
  call require(all(bags(:,j)==[(k,k=1,12)]),'nonbootstrap samples preserve training order')
 end do
 forest%oob_score=.true.;call forest%set_bootstrap(.true.);call forest%fit(x,y(:,1))
 call require(allocated(forest%oob_prediction_).and..not.allocated(forest%oob_prediction_multi_), &
  'scalar OOB storage')
 extra%n_estimators=3;extra%random_state=42
 call extra%fit(x,y);prediction=extra%predict_multi(x)
 call require(.not.extra%bootstrap.and.maxval(abs(prediction-y))<1.e-12_dp,'default ExtraTrees interpolation')
 call require(abs(extra%score(x,y)-1)<1.e-12_dp,'joint forest score')
 call extra%set_bootstrap(.true.);extra%oob_score=.true.;extra%max_samples=1
 call extra%fit(x,y)
 call require(extra%bootstrap.and.allocated(extra%oob_prediction_multi_),'explicit ExtraTrees bootstrap and OOB')
 call require(all(extra%feature_importances_==0),'stump forests have zero importances')
 do i=1,12
  if(extra%oob_counts_(i)==0)call require(all(extra%oob_prediction_multi_(i,:)==0),'uncovered OOB values are zero')
 end do
 print *,'Joint forest predictions, bootstrap ownership, OOB averages, scores and refits: PASS'
end program
