program test_cross_val_predict
  use sklearn_validation
  use sklearn_common, only: dp,require
  use sklearn_linear_model, only: ridge,logistic_regression
  use sklearn_naive_bayes, only: gaussian_nb
  implicit none
  type(ridge) :: base
  type(logistic_regression) :: classifier
  type(gaussian_nb) :: bayes
  type(cross_validation_options) :: options
  type(cross_validation_result) :: results
  real(dp) :: x(30,3),y(30)
  real(dp), allocatable :: prediction(:,:),expected(:),prob(:,:),margin(:,:),logp(:,:)
  integer :: i,j,labels(30)
  integer, allocatable :: classes(:)
  do i=1,30
    x(i,:)=[sin(real(i,dp)),cos(real(2*i,dp)),real(i,dp)/20]
  end do
  y=2*x(:,1)-x(:,2);options%cv=3;options%return_indices=.true.;options%return_estimator=.true.
  results=cross_validate(base,x,y,options);prediction=cross_val_predict(base,x,y,options)
  do i=1,3
    select type(model=>results%estimators(i)%model)
    type is(ridge)
      expected=model%predict(x(results%indices(i)%test,:))
      call require(maxval(abs(prediction(results%indices(i)%test,1)-expected))<1.e-12_dp,'OOF preserves sample order')
    end select
  end do
  labels=merge(9,-4,x(:,1)>0);classes=cross_val_predict_class(classifier,x,labels,options)
  options%method='predict_proba';prob=cross_val_predict(classifier,x,labels,options)
  call require(all(classes==merge(9,-4,prob(:,2)>prob(:,1))),'typed OOF labels agree with probabilities')
  call require(all(abs(sum(prob,dim=2)-1)<1.e-12_dp),'OOF probability normalization')
  options%method='decision_function';margin=cross_val_predict(classifier,x,labels,options)
  call require(size(margin,2)==1,'binary OOF decisions are matrix columns')
  options%method='predict_log_proba';logp=cross_val_predict(classifier,x,labels,options)
  call require(maxval(abs(exp(logp)-prob))<1.e-12_dp,'OOF log probabilities')
  options=cross_validation_options();allocate(options%cv_splits(3))
  do i=1,3
    ! Nonmonotone fold/test order exercises inverse concatenation ordering.
    options%cv_splits(i)%test=[(j,j=31-i,1,-3)]
    options%cv_splits(i)%train=pack([(j,j=1,30)],.not.[(any(options%cv_splits(i)%test==j),j=1,30)])
  end do
  prediction=cross_val_predict(base,x,y,options)
  call require(size(prediction,1)==30,'arbitrary partition order')
  labels=-4;labels(1)=9;options=cross_validation_options();options%cv=3
  options%method='predict_proba';prob=cross_val_predict(bayes,x,labels,options)
  call require(all(prob(1:10,2)==0),'missing fold class gets zero probability column')
  options%method='predict_log_proba';logp=cross_val_predict(bayes,x,labels,options)
  call require(all(logp(1:10,2)==-huge(1._dp)),'missing log-probability column gets minimum finite float')
  print *, 'Out-of-fold prediction methods: PASS'
end program
