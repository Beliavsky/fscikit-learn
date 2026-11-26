program test_classic_boosting_losses
 use sklearn_common,only:dp,require
 use sklearn_ensemble,only:gradient_boosting_regressor,gradient_boosting_classifier
 implicit none
 type(gradient_boosting_regressor)::reg
 type(gradient_boosting_classifier)::cls
 real(dp)::x(4,2),y(4),w(4),initial
 real(dp),allocatable::stages(:,:),pred(:),changed(:),joint(:,:)
 integer::i,k
 character(16),parameter::losses(4)=[character(16)::'squared_error','absolute_error','quantile','huber']
 integer::labels(4)
 x(:,1)=[1._dp,2._dp,3._dp,4._dp];x(:,2)=[2._dp,1._dp,4._dp,3._dp]
 y=[0._dp,1._dp,4._dp,9._dp];w=1
 reg%n_estimators=4;reg%max_depth=2;reg%alpha=.5_dp;reg%learning_rate=.2_dp
 do k=1,4
  reg%loss=losses(k);call reg%fit(x,y)
  initial=2.5_dp;if(k==1)initial=3.5_dp
  call require(reg%init_==initial,'default regression initialization matches unweighted DummyRegressor')
  stages=reg%staged_predict(x);pred=reg%predict(x)
  call require(all(stages(:,4)==pred),'last staged regression prediction equals predict')
  if(k==4)call require(reg%huber_delta_(1)==1.5_dp,'Huber first threshold uses weighted lower percentile')
  do i=1,4
   joint=reg%estimators_(i)%predict_multi(x)
   changed=reg%estimators_(i)%predict(x)
   call require(all(joint(:,1)==changed),'updated terminal values stay synchronized in scalar/joint tree storage')
  end do
  call reg%fit(x,y,w)
  initial=1;if(k==1)initial=3.5_dp
  call require(reg%init_==initial,'explicit unit weights use the lower weighted percentile')
  pred=reg%predict(x);reg%learning_rate=2*reg%learning_rate;changed=reg%predict(x)
  call require(maxval(abs(changed-reg%init_-2*(pred-reg%init_)))<1.e-12_dp,'post-fit learning rate follows sklearn prediction contract')
  reg%learning_rate=.2_dp
 end do
 cls%n_estimators=3;labels=[-9,8,-9,8]
 call cls%fit(0*x,labels)
 call require(all(cls%predict(0*x)==8),'zero binary decision chooses positive class, including balanced stumps')
 call require(all(cls%staged_predict(0*x)==8),'staged binary ties choose positive class')
 call cls%set_max_features('log2');call cls%fit(x,labels)
 call require(cls%max_features_==1,'classification boosting accepts named feature budgets')
 call reg%set_max_features(1._dp);call reg%fit(x,y)
 call require(reg%max_features_==2,'regression boosting real one means all features')
  call require(reg%n_estimators_==4.and.reg%n_trees_per_iteration_==1,'regression boosting learned iteration metadata')
 reg%learning_rate=0;call reg%fit(x,y);stages=reg%staged_predict(x)
 call require(all(stages==reg%init_),'zero learning rate keeps every regression stage at initialization')
 cls%learning_rate=0;call cls%fit(x,labels)
 call require(all(cls%predict(x)==8),'zero learning rate is valid for classification and keeps binary tie rule')
 print *,'Four classic regression losses, weighted percentiles, staged outputs, stored leaves and binary ties: PASS'
end program
