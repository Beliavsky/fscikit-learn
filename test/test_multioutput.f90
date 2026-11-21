program test_multioutput
  use sklearn, only: dp,multi_output_regressor,multi_output_classifier,multioutput_probability, &
    linear_regression,ridge,gaussian_nb,logistic_regression,sgd_regressor,sgd_classifier,label_set
  use sklearn_common, only: require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
  implicit none
  type(multi_output_regressor) :: reg,reg_copy
  type(multi_output_classifier) :: clf,clf_copy
  type(linear_regression) :: ols
  type(ridge) :: ridge_base
  type(gaussian_nb) :: nb
  type(logistic_regression) :: lr
  type(sgd_regressor) :: sgdr
  type(sgd_classifier) :: sgdc
  type(label_set) :: classes(2)
  type(multioutput_probability), allocatable :: prob(:)
  real(dp) :: x(8,1),y(8,2),query(3,1),w(8)=1
  real(dp), allocatable :: prediction(:,:),original(:)
  integer :: cy(8,2),i
  integer, allocatable :: labels(:,:),wrong(:,:)
  x(:,1)=[-4._dp,-3._dp,-2._dp,-1._dp,1._dp,2._dp,3._dp,4._dp]
  y(:,1)=2*x(:,1)+1; y(:,2)=-3*x(:,1)+4; query(:,1)=[-2._dp,0._dp,2._dp]
  call ols%fit(x,0*y(:,1)); original=ols%coef_
  call reg%fit(x,y,ols); prediction=reg%predict(query)
  call require(maxval(abs(prediction(:,1)-(2*query(:,1)+1)))<1.e-12_dp,'independent first regression target')
  call require(maxval(abs(prediction(:,2)-(-3*query(:,1)+4)))<1.e-12_dp,'independent second regression target')
  call require(all(ols%coef_==original),'input regressor remains unchanged')
  call require(abs(reg%score(x,y)-1)<1.e-14_dp,'uniform-average regression R2')
  call require(ieee_is_nan(reg%score(x(:1,:),y(:1,:))),'singleton multioutput R2 is quiet NaN')
  call require(reg%score(x,0*y+4,w)==0,'weighted constant targets retain force_finite R2')
  reg_copy=reg; ridge_base%alpha=2; call reg%fit(x,y(:,:1),ridge_base)
  call require(reg%n_outputs_==1.and.reg_copy%n_outputs_==2,'refit and deep-copy target state')
  select type(base=>reg%estimators_(1)%estimator_)
  type is(ridge)
    call require(base%alpha==2,'dynamic regressor subtype retained')
  class default
    call require(.false.,'wrong fitted regressor subtype')
  end select
  cy(:,1)=[0,0,0,0,1,1,1,1]; cy(:,2)=[-7,-7,4,4,11,11,11,11]
  call nb%fit(x,cy(:,1)); call clf%fit(x,cy,nb,w)
  prob=clf%predict_proba(query)
  call require(size(prob)==2.and.size(prob(1)%classes)==2.and.size(prob(2)%classes)==3,'ragged class probabilities')
  call require(all(prob(2)%classes==[-7,4,11]),'per-output class labels')
  labels=clf%predict(x); wrong=labels; wrong(:,2)=999
  call require(clf%score(x,wrong)==0,'classification score requires every target to match')
  clf_copy=clf; call clf%fit(x,cy(:,:1),lr)
  call require(clf%n_outputs_==1.and.clf_copy%n_outputs_==2,'classifier refit and deep copy')
  classes(1)%values=[1,0]; classes(2)%values=[11,-7,4]
  clf=multi_output_classifier()
  call clf%partial_fit(x(:3,:),cy(:3,:),nb,classes,sample_weight=w(:3))
  call require(.not.allocated(clf%classes_),'source partial-only classes metadata remains absent')
  select type(base=>clf%estimators_(1)%estimator_)
  type is(gaussian_nb)
    call require(sum(base%class_count_)==3,'first partial_fit clones configuration without fitted statistics')
  end select
  call clf%partial_fit(x(4:,:),cy(4:,:),sample_weight=w(4:))
  call clf%partial_fit(x,cy(:,:1))
  call require(clf%n_outputs_==1,'source partial_fit can shrink the output list')
  sgdr%shuffle=.false.; sgdr%learning_rate='constant'; sgdr%eta0=.02_dp; sgdr%average=3
  call sgdr%fit(x,y(:,1)); reg=multi_output_regressor()
  call reg%partial_fit(x,y,sgdr,sample_weight=w)
  select type(base=>reg%estimators_(1)%estimator_)
  type is(sgd_regressor)
    call require(base%t_==9,'first partial_fit clears pretrained SGD counters')
  end select
  call reg%partial_fit(x,y(:,:1),sample_weight=w)
  call require(reg%n_outputs_==1,'regression streaming shrink')
  sgdc%loss='log_loss'; sgdc%learning_rate='constant'; sgdc%average=3; sgdc%fit_intercept=.false.
  call sgdc%fit(x,cy(:,1)); clf=multi_output_classifier()
  call clf%partial_fit(x,cy,sgdc,classes,sample_weight=w)
  prob=clf%predict_proba(query)
  do i=1,2
    call require(maxval(abs(sum(prob(i)%values,dim=2)-1))<1.e-14_dp,'SGD output probabilities normalize')
  end do
  print *, 'multioutput regression and classification: PASS'
end program
