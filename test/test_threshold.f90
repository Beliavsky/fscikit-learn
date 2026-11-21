program test_threshold
  use sklearn_threshold
  use sklearn_linear_model, only: logistic_regression
  use sklearn_naive_bayes, only: gaussian_nb
  use sklearn_common, only: dp,require
  implicit none
  type(logistic_regression) :: base
  type(gaussian_nb) :: nb
  type(fixed_threshold_classifier) :: fixed
  type(tuned_threshold_classifier_cv) :: tuned,copy
  real(dp) :: x(90,2),weights(90)
  integer :: y(90),i
  integer, allocatable :: predicted(:)
  real(dp), allocatable :: prob(:,:),decision(:,:)
  do i=1,90
    x(i,:)=[sin(real(i,dp)),cos(real(i,dp)*.37_dp)]
    y(i)=merge(7,-3,x(i,1)+.3_dp*x(i,2)>.5_dp)
    weights(i)=1+.1_dp*mod(i,4)
  end do
  base%tol=1.e-10_dp; base%max_iter=1000
  call fixed%fit(x,y,base,weights); prob=fixed%predict_proba(x)
  predicted=fixed%predict(x)
  call require(all(predicted==merge(7,-3,prob(:,2)>=.5_dp)),'fixed automatic probability cutoff')
  fixed%pos_label=-3; fixed%threshold=.2_dp; predicted=fixed%predict(x)
  call require(all(predicted==merge(-3,7,prob(:,1)>=.2_dp)),'fixed reversed positive class')
  fixed%response_method='decision_function'; fixed%threshold=0
  decision=fixed%decision_function(x); predicted=fixed%predict(x)
  call require(all(predicted==merge(-3,7,-decision(:,1)>=0)),'fixed reversed decision response')
  tuned%cv=3; tuned%n_thresholds=25; tuned%store_cv_results=.true.
  call tuned%fit(x,y,base,weights)
  call require(all(shape(tuned%fold_scores_)==[3,25]),'threshold fold score layout')
  i=maxloc(tuned%mean_test_score_,dim=1)
  call require(tuned%best_threshold_==tuned%thresholds_(i),'first best threshold selection')
  prob=tuned%predict_proba(x); predicted=tuned%predict(x)
  call require(all(predicted==merge(7,-3,prob(:,2)>=tuned%best_threshold_)),'tuned probability cutoff')
  copy=tuned; tuned%store_cv_results=.false.; tuned%thresholds=[.1_dp,.3_dp,.3_dp,.7_dp]
  call tuned%fit(x,y,nb)
  call require(.not.allocated(tuned%thresholds_),'threshold refit clears optional learned CV arrays')
  call require(allocated(copy%thresholds_),'threshold copy preserves independent CV arrays')
  tuned=copy; tuned%refit=.false.; tuned%prefit=.true.
  call base%fit(x,y,weights); call tuned%fit(x,y,base,weights)
  call require(size(tuned%cv_splits_)==1.and.size(tuned%cv_splits_(1)%train)==0,'prefit threshold validation')
  tuned%prefit=.false.; tuned%validation_fraction=.3_dp
  call tuned%fit(x,y,base,weights,objective=utility)
  call require(size(tuned%cv_splits_)==1,'single holdout threshold tuning without final refit')
  call require(tuned%score(x,y,weights)>=0,'threshold score is weighted accuracy')
  print *, 'Threshold classifiers: PASS'
contains
  real(dp) function utility(truth,prediction,sample_weight) result(value)
    integer, intent(in) :: truth(:),prediction(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp) :: cost(size(truth))
    cost=0
    where(truth==-3.and.prediction==7) cost=1
    where(truth==7.and.prediction==-3) cost=3
    value=-sum(cost)/size(cost)
    if(present(sample_weight)) value=-sum(cost*sample_weight)/sum(sample_weight)
  end function
end program
