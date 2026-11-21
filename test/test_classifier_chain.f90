program test_classifier_chain
  use sklearn, only: dp,classifier_chain,logistic_regression,gaussian_nb,kneighbors_classifier,index_split
  use sklearn_common, only: require
  implicit none
  type(classifier_chain) :: model,saved
  type(logistic_regression) :: base
  type(gaussian_nb) :: nb
  type(kneighbors_classifier) :: knn
  type(index_split) :: folds(2)
  real(dp) :: x(12,1),w(12)=1
  real(dp), allocatable :: probability(:,:),decision(:,:)
  integer :: y(12,2),i
  integer, allocatable :: prediction(:,:),template(:)
  x(:,1)=[(real(i-6,dp),i=1,12)]
  y(:,1)=[0,0,0,0,0,0,1,1,1,1,1,1]; y(:,2)=[0,0,0,1,1,1,1,1,1,0,0,0]
  call base%fit(x,y(:,1)); template=base%predict(x)
  model%order=[2,1]; call model%fit(x,y,base)
  prediction=model%predict(x); probability=model%predict_proba(x); decision=model%decision_function(x)
  call require(all(shape(prediction)==[12,2]),'classifier chain prediction dimensions')
  call require(all(probability>=0.and.probability<=1),'classifier chain positive class probabilities')
  call require(all(base%predict(x)==template),'classifier chain preserves pretrained template')
  call require(abs(model%score(x,prediction,w)-1)<1.e-14_dp,'classifier chain exact subset weighted score')
  do i=1,2
    select type(fitted=>model%estimators_(i)%estimator_)
    type is(logistic_regression)
      call require(size(fitted%coef_,2)==i,'classifier chain augments feature counts')
    end select
  end do
  saved=model; model%chain_method='predict_proba'; model%cv=3
  call model%fit(x,y,base)
  call require(model%chain_method_=='predict_proba','classifier chain selected response metadata')
  call require(all(saved%predict(x)==prediction),'classifier chain deep copies fitted state')
  folds(1)%test=[1,3,5,7,9,11]; folds(1)%train=[2,4,6,8,10,12]
  folds(2)%test=folds(1)%train; folds(2)%train=folds(1)%test
  model%cv=0; call model%fit(x,y,base,splits=folds)
  model%chain_methods=[character(20) :: 'decision_function','predict_proba','predict']
  knn%n_neighbors=2; call model%fit(x,y,knn)
  call require(model%chain_method_=='predict_proba','classifier chain fallback method chooses supported capability')
  deallocate(model%chain_methods,model%order); model%random_order=.true.; model%random_state=0
  model%chain_method='predict'; call model%fit(x,y,nb,w)
  call require(all(model%order_==[2,1]),'classifier chain legacy random order')
  y=merge(-7,11,y==0); call model%fit(x,y,nb)
  prediction=model%predict(x)
  call require(abs(model%score(x,prediction)-1)<1.e-14_dp,'source nonzero shared binary indicator labels')
  model%cv=99; call model%fit(x,y(:,:1),nb)
  call require(model%n_outputs_==1.and.size(model%classes_)==1,'classifier chain refit replaces metadata')
  print *, 'classifier chains: PASS'
end program
