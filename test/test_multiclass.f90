program test_multiclass
  use sklearn, only: dp,one_vs_rest_classifier,one_vs_one_classifier,output_code_classifier, &
    logistic_regression,gaussian_nb,sgd_classifier,ridge,linear_svc,perceptron
  use sklearn_common, only: require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
  implicit none
  type(one_vs_rest_classifier) :: ovr,saved
  type(one_vs_one_classifier) :: ovo
  type(output_code_classifier) :: code
  type(logistic_regression) :: base
  type(gaussian_nb) :: nb
  type(sgd_classifier) :: sgd
  type(ridge) :: reg
  type(perceptron) :: percep
  type(linear_svc) :: margin
  real(dp) :: x(12,2),query(3,2),weights(12)=1
  real(dp), allocatable :: proba(:,:),decision(:,:)
  integer :: y(12),indicator(12,3),i
  integer, allocatable :: labels(:),matrix(:,:)
  x(:,1)=[-5._dp,-4._dp,-3._dp,-2._dp,-1._dp,0._dp,1._dp,2._dp,3._dp,4._dp,5._dp,6._dp]
  x(:,2)=[(real(mod(i,3),dp),i=1,12)]
  y=[-7,-7,-7,-7,4,4,4,4,19,19,19,19]; query=x([2,6,11],:)
  call percep%fit(x,merge(1,0,y==19)); decision=percep%decision_function(query)
  call require(size(percep%coef_,1)==1.and.size(decision,2)==1,'binary Perceptron uses a positive-class head')
  call base%fit(x,merge(1,0,y==19)); call ovr%fit(x,y,base)
  proba=ovr%predict_proba(query); decision=ovr%decision_function(query)
  call require(all(shape(proba)==[3,3]).and.all(shape(decision)==[3,3]),'OVR multiclass response dimensions')
  call require(maxval(abs(sum(proba,dim=2)-1))<1.e-14_dp,'OVR normalizes disjoint class probabilities')
  saved=ovr; call ovr%fit(x,7+0*y,base)
  labels=ovr%predict(query); proba=ovr%predict_proba(query)
  call require(all(labels==7).and.all(proba(:,1)==1).and.all(proba(:,2)==0),'OVR single-class constant predictor')
  call require(ovr%estimators_(1)%constant_.and..not.allocated(ovr%estimators_(1)%estimator_), &
    'OVR constant heads do not fit the supplied template')
  call require(size(saved%classes_)==3,'OVR fitted deep copy')
  indicator(:,1)=0; indicator(:,2)=merge(1,0,y==4); indicator(:,3)=merge(1,0,y==19)
  call ovr%fit(x,indicator,nb); matrix=ovr%predict_indicator(query); proba=ovr%predict_proba(query)
  call require(ovr%multilabel_.and.all(ovr%classes_==[0,1,2]),'OVR indicator class indices')
  call require(all(matrix==merge(1,0,proba>.5_dp)),'source constant-first head uses probability threshold')
  call require(all(proba(:,1)==0),'OVR keeps marginal indicator probabilities')
  call ovr%fit(x,y,reg); labels=ovr%predict(query)
  call require(all(shape(labels)==[3]),'OVR supports native regression bases')
  call ovo%fit(x,y,base); decision=ovo%decision_function(query); labels=ovo%predict(query)
  call require(size(ovo%estimators_)==3.and.all(shape(decision)==[3,3]),'OVO fits all ordered class pairs')
  call require(abs(ovo%score(query,labels)-1)<1.e-14_dp,'OVO weighted classification score')
  code%code_size=2; code%random_state=42; call code%fit(x,y,base)
  call require(all(shape(code%code_book_)==[3,6]),'output-code book dimensions')
  call require(all(abs(code%code_book_)==1),'decision output-code words use signs')
  call code%fit(x,y,nb); call require(all(code%code_book_>=0),'probability code words use zero and one')
  ovr=one_vs_rest_classifier()
  call ovr%partial_fit(x(:4,:),y(:4),nb,[19,-7,4,4])
  call require(size(ovr%estimators_)==3,'OVR initializes every declared binary head')
  call ovr%partial_fit(x(5:,:),y(5:)); call require(all(ovr%classes_==[-7,4,19]),'OVR normalized streaming classes')
  ovo=one_vs_one_classifier(); call ovo%partial_fit(x(:4,:),y(:4),nb,[-7,4,19])
  call require(ovo%estimators_(3)%n_features_in_==0,'OVO skips unobserved class pairs')
  call ovo%partial_fit(x(5:,:),y(5:)); labels=ovo%predict(query)
  sgd%loss='log_loss'; sgd%shuffle=.false.; sgd%max_iter=3
  indicator(:,1)=merge(1,0,y==-7)
  call ovr%fit(x,indicator,sgd); call ovr%partial_fit(x,indicator)
  call require(ovr%multilabel_,'OVR multilabel partial_fit continues a fitted indicator model')
  weights=0; call ovr%fit(x,y,nb,weights); proba=ovr%predict_proba(query); labels=ovr%predict(query)
  call require(all(ieee_is_nan(proba)).and.all(labels==-7),'OVR retains source undefined-posterior argmax behavior')
  call ovo%fit(x,y,nb,weights); decision=ovo%decision_function(query); labels=ovo%predict(query)
  call require(all(ieee_is_nan(decision)).and.all(labels==-7),'OVO retains quiet NaN confidence behavior')
  print *, 'multiclass reductions: PASS'
end program
