program test_sgd_one_class
  use sklearn, only: dp,sgd_one_class_svm
  use sklearn_common, only: require
  implicit none
  type(sgd_one_class_svm) :: model,stream
  real(dp) :: one(1,1),x(12,2),weights(12)
  real(dp), allocatable :: scores(:),before(:)
  integer, allocatable :: labels(:)
  integer :: i
  one=1; model%learning_rate='constant'; model%eta0=.1_dp; model%nu=.5_dp
  model%max_iter=1; model%tol=-1; model%shuffle=.false.; model%average=1
  call model%fit(one)
  call require(abs(model%coef_(1)-.1_dp)<1.e-14_dp,'one-class hinge boundary update')
  call require(abs(model%offset_+.05_dp)<1.e-14_dp,'one-class offset update')
  call model%partial_fit(one)
  call require(abs(model%coef_(1)-.09875_dp)<1.e-14_dp,'one-class streaming average coefficients')
  call require(abs(model%offset_+.025_dp)<1.e-14_dp,'one-class streaming average offset')
  call require(model%t_==3.and.model%n_iter_==1,'one-class partial-fit clock')
  do i=1,12
    x(i,:)=[1+.2_dp*sin(real(i,dp)),1+.3_dp*cos(real(i,dp))]
  end do
  model=sgd_one_class_svm(); model%random_state=42; model%max_iter=15; model%tol=-1
  weights=1; weights(1)=0; labels=model%fit_predict(x,weights)
  scores=model%decision_function(x)
  call require(all(labels==merge(1,-1,scores>=0)),'one-class labels and zero convention')
  call require(maxval(abs(model%score_samples(x)-scores-model%offset_))<1.e-14_dp,'one-class score offset')
  before=model%coef_; call model%fit(x,weights)
  call require(maxval(abs(model%coef_-before))<1.e-14_dp,'one-class repeatable shuffle')
  call require(model%t_==1+15*12,'one-class fit resets clock')
  stream%max_iter=3; stream%tol=-1; stream%fit_intercept=.false.; stream%average=1
  call stream%partial_fit(x(:6,:)); call stream%partial_fit(x(7:,:))
  call require(stream%t_==13.and.abs(stream%offset_)<1.e-14_dp,'one-class streaming no-intercept average')
  print *, 'SGDOneClassSVM tests passed'
end program
