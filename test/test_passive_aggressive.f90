program test_passive_aggressive
  use sklearn
  use sklearn_common, only: require
  implicit none
  type(passive_aggressive_regressor) :: reg,repeat
  type(passive_aggressive_classifier) :: classifier,again
  real(dp) :: one(1,2),target(1),sx(2,1),sy(2),x(60,2),y(60),zero(2,2),t
  real(dp), allocatable :: decision(:,:)
  integer :: labels(60),i
  one(1,:)=[2._dp,1._dp]; target=3
  reg%shuffle=.false.; reg%tol=-1; reg%max_iter=1; reg%c=0.2_dp; reg%epsilon=0.5_dp
  call reg%fit(one,target)
  call require(maxval(abs(reg%coef_-[0.4_dp,0.2_dp]))<1.e-14_dp,'PA-I closed-form step')
  call require(abs(reg%intercept_-0.2_dp)<1.e-14_dp,'PA intercept excluded from feature norm')
  call require(reg%n_iter_==1.and.reg%t_==2.and..not.reg%converged_,'fixed epoch metadata')
  reg%loss='squared_epsilon_insensitive'; call reg%fit(one,target)
  call require(maxval(abs(reg%coef_-[2._dp/3,1._dp/3]))<1.e-14_dp,'PA-II closed-form step')
  call require(abs(reg%intercept_-1._dp/3)<1.e-14_dp,'PA-II intercept step')
  sx(:,1)=[1._dp,2._dp]; sy=[1._dp,0._dp]
  reg%loss='epsilon_insensitive'; reg%c=1; reg%epsilon=0; reg%fit_intercept=.false.; reg%average=1
  call reg%fit(sx,sy)
  call require(abs(reg%coef_(1)-0.5_dp)<1.e-14_dp,'post-update coefficient averaging')
  reg%average=2; call reg%fit(sx,sy)
  call require(reg%coef_(1)==0,'delayed averaging')
  reg%average=100; call reg%fit(sx,sy)
  call require(reg%coef_(1)==0,'unreached averaging threshold uses standard coefficients')
  zero=0; reg%average=0; reg%fit_intercept=.true.; reg%epsilon=0.1_dp
  call reg%fit(zero,[2._dp,2._dp])
  call require(all(reg%coef_==0).and.reg%intercept_==0,'PA-I skips zero-feature intercept updates')
  reg%loss='squared_epsilon_insensitive'; call reg%fit(zero(:1,:),[2._dp])
  call require(abs(reg%intercept_-3.8_dp)<1.e-14_dp,'PA-II updates intercept for zero features')
  reg%loss='epsilon_insensitive'; reg%tol=1.e-3_dp; reg%max_iter=20; reg%n_iter_no_change=2
  call reg%fit(zero,[2._dp,2._dp])
  call require(reg%converged_.and.reg%n_iter_==3,'patience counts non-improving online loss')
  do i=1,60
    t=real(i,dp)/60
    x(i,:)=[sin(17*t),cos(23*t)]; y(i)=2*x(i,1)-0.7_dp*x(i,2)+0.3_dp
  end do
  reg%shuffle=.true.; reg%random_state=41; reg%c=0.1_dp; reg%epsilon=0.01_dp
  reg%max_iter=1000; reg%tol=1.e-7_dp; reg%n_iter_no_change=10; reg%average=0
  call reg%fit(x,y); repeat=reg; call repeat%fit(x,y)
  call require(all(reg%coef_==repeat%coef_).and.reg%intercept_==repeat%intercept_,'regression shuffle reproducibility')
  call require(reg%score(x,y)>0.999_dp,'passive-aggressive regression recovery')
  do i=1,60
    t=real(i,dp)/60
    select case(mod(i-1,3))
    case(0); labels(i)=-4; x(i,:)=[-2._dp,-1._dp]
    case(1); labels(i)=7; x(i,:)=[2._dp,-1._dp]
    case(2); labels(i)=21; x(i,:)=[0._dp,2._dp]
    end select
    x(i,:)=x(i,:)+[0.1_dp*sin(17*t),0.1_dp*cos(23*t)]
  end do
  classifier%shuffle=.false.; classifier%max_iter=100; classifier%tol=1.e-5_dp
  call classifier%fit(x,labels)
  call require(classifier%score(x,labels)==1.and.classifier%converged_,'multiclass PA recovery')
  call require(size(classifier%coef_,1)==3.and.classifier%n_iter_==maxval(classifier%n_iter_per_class_), &
    'one-versus-rest planes and epoch counts')
  classifier%shuffle=.true.; classifier%random_state=31; classifier%average=5
  call classifier%fit(x,labels); again=classifier; call again%fit(x,labels)
  call require(all(classifier%coef_==again%coef_),'classification shuffle reproducibility')
  classifier%shuffle=.false.; classifier%average=0; classifier%max_iter=1; classifier%tol=-1
  call classifier%fit(zero,[-3,8]); decision=classifier%decision_function(zero)
  call require(size(decision,2)==1.and.all(decision==0),'binary PA has a single decision column')
  call require(all(classifier%predict(zero)== -3),'zero-margin tie selects lower binary class')
  print *, 'Passive-aggressive models tests passed'
end program
