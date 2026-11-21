program test_self_training
  use sklearn, only: dp,self_training_classifier,gaussian_nb,kneighbors_classifier,logistic_regression
  use sklearn_common, only: require
  implicit none
  type(self_training_classifier) :: model,copy
  type(gaussian_nb) :: nb
  type(kneighbors_classifier) :: knn
  type(logistic_regression) :: lr
  real(dp) :: x(8,1),z(4,1)
  real(dp), allocatable :: prob(:,:),logp(:,:),decision(:,:)
  integer :: y(8)=[4,4,-1,-1,-1,-1,9,9],saved(8),i
  x(:,1)=[-4._dp,-3._dp,-2._dp,-1._dp,1._dp,2._dp,3._dp,4._dp]; saved=y
  z(:,1)=[-5._dp,-1._dp,1._dp,5._dp]
  knn%n_neighbors=1
  call model%fit(x,y,knn)
  call require(model%n_iter_==1.and.model%termination_condition_=='all_labeled','confident one-step self-training')
  call require(all(model%transduction_==[4,4,4,4,9,9,9,9]),'analytic nearest-neighbor transduction')
  call require(all(model%labeled_iter_==[0,0,1,1,1,1,0,0]),'initial labels have iteration zero')
  call require(all(y==saved).and..not.allocated(knn%classes_),'base estimator and input labels preserved')
  call require(all(model%predict(z)==[4,4,9,9]),'inductive prediction')
  prob=model%predict_proba(z)
  call require(count(prob==0)==4,'nearest-neighbor zero probabilities')
  copy=model; model%criterion='k_best'; model%k_best=1; model%max_iter=-1
  call model%fit(x,y,knn)
  call require(model%n_iter_==4,'unlimited self-training terminates after labeling all rows')
  call require(all(model%labeled_iter_==[0,0,1,2,3,4,0,0]),'stable input-order confidence ties')
  call require(copy%n_iter_==1,'deep-copy self-training state')
  model%max_iter=2; call model%fit(x,y,knn)
  call require(model%termination_condition_=='max_iter'.and.count(model%transduction_==-1)==2,'iteration cap')
  model%max_iter=0; call model%fit(x,y,nb)
  call require(model%n_iter_==0.and.all(model%transduction_==y),'zero-iteration final supervised fit')
  prob=model%predict_proba(z)
  logp=model%predict_log_proba(z)
  call require(maxval(abs(exp(logp)-prob))<1.e-14_dp,'Gaussian log probabilities delegated')
  call nb%fit(x([1,2,7,8],:),y([1,2,7,8]))
  call require(maxval(abs(prob-nb%predict_proba(z)))<1.e-14_dp,'mandatory final refit')
  model%criterion='threshold'; model%threshold=.99999_dp; model%max_iter=1
  knn%n_neighbors=4; call model%fit(x,y,knn)
  call require(model%termination_condition_=='max_iter'.and.model%n_iter_==1,'max_iter overrides no_change')
  model%max_iter=10; call model%fit(x,y,knn)
  call require(model%termination_condition_=='no_change','no confident samples stopping rule')
  model%threshold=.5_dp; call model%fit(x,y,knn)
  call require(model%termination_condition_=='no_change','threshold comparison is strict')
  model%criterion='k_best'; model%k_best=100; model%max_iter=1; call model%fit(x,y,knn)
  call require(model%termination_condition_=='all_labeled','all_labeled overrides iteration cap')
  call require(all(model%transduction_(3:6)==4),'equal class probabilities choose first sorted class')
  model%criterion='threshold'; model%threshold=.7_dp; model%max_iter=10
  call model%fit(x,y,lr); decision=model%decision_function(z); prob=model%predict_proba(z)
  call require(maxval(abs(prob(:,2)-1/(1+exp(-decision(:,1)))))<1.e-13_dp,'delegated logistic decision')
  call require(model%score(z,[4,4,9,9])==1,'self-training accuracy')
  y=[4,4,4,4,9,9,9,9]; call model%fit(x,y,lr)
  call require(model%n_iter_==0.and.model%termination_condition_=='all_labeled','fully labeled fit')
  do i=1,8
    call require(model%labeled_iter_(i)==0,'fully labeled history')
  end do
  y=[4,4,-1,-1,-1,-1,-1,-1]
  call model%fit(x,y,nb)
  call require(size(model%classes_)==1.and.all(model%transduction_==4),'one-class base estimator support')
  call require(size(nb%classes_)==2,'previously fitted input estimator remains unchanged')
  print *, 'self-training classifier: PASS'
end program
