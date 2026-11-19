program test_one_class_svm
  use sklearn
  use sklearn_common, only: require
  implicit none
  type(one_class_svm) :: model,weighted,removed
  real(dp) :: x(40,2),query(2,2),t,beta(40),weights(40),line(4,1)
  real(dp), allocatable :: raw(:),decision(:),gram(:,:),again(:)
  integer, allocatable :: labels(:)
  integer :: i
  do i=1,40
    t=2*acos(-1._dp)*i/40; x(i,:)=[0.5_dp*cos(t),0.5_dp*sin(t)]
  end do
  model%nu=0.25_dp; model%gamma=1; model%tol=1.e-7_dp
  labels=model%fit_predict(x)
  if(.not.model%converged_) print *, 'OneClassSVM iterations/gap: ',model%n_iter_,model%kkt_violation_
  call require(model%converged_,'OneClassSVM KKT convergence')
  beta=0; beta(model%support_)=model%dual_coef_
  call require(abs(sum(beta)-10)<1.e-10_dp.and.all(beta>=0).and.all(beta<=1+1.e-12_dp), &
    'OneClassSVM dual mass and bounds')
  call require(model%n_support_>=10,'OneClassSVM nu bounds support fraction')
  raw=model%score_samples(x); decision=model%decision_function(x)
  call require(maxval(abs(decision-(raw-model%offset_)))<1.e-12_dp,'one-class offset convention')
  do i=1,40
    if(beta(i)>1.e-9_dp.and.beta(i)<1-1.e-9_dp) then
      call require(abs(decision(i))<2.e-6_dp,'one-class free supports on margin')
    else if(beta(i)<=1.e-9_dp) then
      call require(decision(i)>=-2.e-6_dp,'one-class nonsupports within boundary')
    else
      call require(decision(i)<=2.e-6_dp,'one-class upper-bound support inequality')
    end if
  end do
  query(1,:)=0; query(2,:)=5; labels=model%predict(query)
  call require(all(labels==[1,-1]),'one-class distinguishes center from distant outlier')
  gram=rbf_kernel(x,x,1._dp); weighted=model; weighted%kernel='precomputed'
  call weighted%fit(gram); again=weighted%decision_function(gram)
  call require(maxval(abs(again-decision))<1.e-9_dp,'one-class precomputed kernel equivalence')
  weights=1; weights(40)=0; weighted=model; removed=model
  call weighted%fit(x,weights); call removed%fit(x(:39,:))
  call require(maxval(abs(weighted%decision_function(query)-removed%decision_function(query)))<1.e-8_dp, &
    'one-class zero sample weight equivalence')
  model%gamma=0; call model%fit(x); raw=model%score_samples(query)
  call require(maxval(abs(raw-10))<1.e-10_dp,'constant kernel preserves one-class dual mass')
  model%kernel='linear'; call model%fit(x)
  call require(model%converged_.and.maxval(abs(model%coef_))<2.e-6_dp.and.abs(model%offset_)<2.e-6_dp, &
    'linear one-class solution vanishes when the origin is in the training hull')
  line(:,1)=[3._dp,4._dp,1._dp,2._dp]; model%nu=0.5_dp; call model%fit(line)
  call require(abs(model%coef_(1)-3)<1.e-12_dp.and.abs(model%offset_-7.5_dp)<1.e-12_dp, &
    'one-class fully bounded dual uses the midpoint threshold interval')
  print *, 'one-class support vector machine: PASS'
end program
