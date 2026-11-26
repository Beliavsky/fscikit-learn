program test_joint_class_forests
 use sklearn_common,only:dp,require
 use sklearn_ensemble,only:random_forest_classifier,extra_trees_classifier
 use sklearn_tree,only:tree_probability
 use sklearn_metrics,only:accuracy_score
 implicit none
 type(random_forest_classifier)::model
 type(extra_trees_classifier)::extra
 type(tree_probability),allocatable::prob(:),part(:),expected(:)
 real(dp)::x(16,2),w(16)
 integer::y(16,2),counts(16),i,j,k,c
 integer,allocatable::bags(:,:),again(:,:),prediction(:,:),oob_labels(:,:)
 do i=1,16
  x(i,:)=[real(i,dp),real(mod(i,3),dp)];y(i,:)=[mod(i,2),merge(0,1,i<=8)]
 end do
 w=1;w(2)=0;w(4)=2
 model%n_estimators=7;model%random_state=7;model%max_samples=10;model%oob_score=.true.
 model%class_weight='balanced_subsample'
 call model%fit(x,y,w)
 prob=model%predict_proba_multi(x);prediction=model%predict_multi(x)
 call require(model%n_outputs_==2.and.all(model%n_classes_multi_==2),'joint forest class metadata')
 call require(all(shape(prediction)==[16,2]),'joint forest prediction shape')
 bags=model%estimators_samples();again=bags;allocate(expected(2),oob_labels(16,2));counts=0
 do k=1,2;allocate(expected(k)%values(16,2));expected(k)%values=0;end do
 do j=1,7
  call require(model%estimators_(j)%n_outputs_==2,'classifier forest uses shared component trees')
  part=model%estimators_(j)%predict_proba_multi(x)
  do i=1,16
   if(any(bags(:,j)==i))cycle
   counts(i)=counts(i)+1
   do k=1,2;expected(k)%values(i,:)=expected(k)%values(i,:)+part(k)%values(i,:);end do
  end do
 end do
 do k=1,2
  do i=1,16
   expected(k)%values(i,:)=expected(k)%values(i,:)/max(1,counts(i))
   oob_labels(i,k)=maxloc(expected(k)%values(i,:),dim=1)-1
  end do
  call require(maxval(abs(model%oob_decision_function_multi_(k)%values-expected(k)%values))<1.e-12_dp, &
   'classifier OOB probabilities average only unsampled trees')
 end do
 call require(all(counts==model%oob_counts_),'classifier OOB coverage')
 call require(model%oob_score_==accuracy_score(y,oob_labels),'classifier OOB subset accuracy')
 bags=0;bags=model%estimators_samples();call require(all(bags==again),'classifier bootstrap arrays are owned')
 model%oob_score=.false.;model%class_weight='none';model%max_samples=0;call model%set_bootstrap(.false.)
 call model%fit(x,y(:,1));prob=model%predict_proba_multi(x)
 call require(model%n_outputs_==1.and.all(model%predict(x)==y(:,1)),'scalar classifier forest refit')
 call require(allocated(model%oob_counts_).and.allocated(model%oob_decision_function_multi_), &
  'classifier refit retains cached OOB arrays')
 call require(maxval(abs(prob(1)%values-model%predict_proba(x)))==0,'single-output probability adapter')
 extra%n_estimators=3;extra%random_state=42
 call extra%fit(x,y);prediction=extra%predict_multi(x)
 call require(.not.extra%bootstrap.and.all(prediction==y),'default joint ExtraTrees interpolation')
 call extra%set_bootstrap(.true.);extra%max_samples=1;extra%oob_score=.true.;call extra%fit(x,y)
 call require(extra%bootstrap.and.allocated(extra%oob_decision_function_multi_),'ExtraTrees explicit bootstrap/OOB')
 call require(all(extra%feature_importances_==0),'stump classifier forest zero importances')
 do i=1,16
  if(extra%oob_counts_(i)>0)cycle
  do k=1,2
   call require(all(extra%oob_decision_function_multi_(k)%values(i,:)==0),'uncovered classifier OOB probabilities are zero')
  end do
 end do
 print *,'Joint classifier forests, class sets, probabilities, OOB reconstruction and refits: PASS'
end program
