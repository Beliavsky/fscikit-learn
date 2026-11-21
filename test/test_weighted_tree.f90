program test_weighted_tree
  use sklearn_tree, only: decision_tree_classifier,decision_tree_regressor,extra_tree_classifier,extra_tree_regressor
  use sklearn_voting, only: voting_classifier,voting_regressor,voting_estimator
  use sklearn_multioutput, only: multi_output_classifier,multi_output_regressor,classifier_chain,regressor_chain
  use sklearn_multiclass, only: one_vs_rest_classifier,one_vs_one_classifier,output_code_classifier
  use sklearn_common, only: dp,require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan,ieee_is_finite,ieee_value,ieee_quiet_nan
  implicit none
  type(voting_classifier) :: voter
  type(voting_regressor) :: voter_reg
  type(voting_estimator) :: bases(1)
  type(multi_output_classifier) :: multiple
  type(multi_output_regressor) :: multiple_reg
  type(classifier_chain) :: chain
  type(regressor_chain) :: chain_reg
  type(one_vs_rest_classifier) :: ovr
  type(one_vs_one_classifier) :: ovo
  type(output_code_classifier) :: code
  type(decision_tree_classifier) :: clf,copy
  type(decision_tree_regressor) :: reg
  type(extra_tree_classifier) :: extra
  type(extra_tree_regressor) :: extra_reg
  real(dp) :: x(4,2),w(4),query(1,2),target(4)
  real(dp), allocatable :: p(:,:),prediction(:)
  integer :: y(4)
  integer, allocatable :: labels(:)
  x(:,1)=[0._dp,0._dp,1._dp,1._dp]; x(:,2)=[0._dp,1._dp,0._dp,1._dp]; y=[0,1,1,0]
  clf%max_depth=2; call clf%fit(x,y); labels=clf%predict(x)
  call require(all(labels==y),'unweighted CART can cross a zero-gain XOR split')
  w=1; call clf%fit(x,y,w); labels=clf%predict(x)
  call require(all(labels==y),'weighted CART can cross a zero-gain XOR split')
  call require(clf%nodes_(1)%weighted_n_samples==4,'weighted root mass metadata')
  copy=clf; w=0; call clf%fit(x,y,w); p=clf%predict_proba(x)
  call require(all(ieee_is_nan(p)),'zero-weight classifier retains source undefined probabilities')
  call require(clf%nodes_(1)%n_samples==0.and.all(clf%predict(x)==0),'zero-weight classifier uses first class')
  call require(all(ieee_is_nan(clf%feature_importances_)),'zero-weight feature importances retain source NaNs')
  call require(all(copy%predict(x)==y),'tree assignment copies fitted nodes')
  x(:,1)=[0._dp,1._dp,2._dp,3._dp]; target=[0._dp,1._dp,0._dp,1._dp]
  call reg%fit(x,target,w); prediction=reg%predict(x)
  call require(all(ieee_is_nan(prediction)),'zero-weight regression retains source undefined leaves')
  w=[-1._dp,1._dp,-1._dp,1._dp]; reg%max_depth=2; call reg%fit(x,target,w)
  prediction=reg%predict(x)
  call require(reg%node_count_==3.and.all(.not.ieee_is_finite(prediction)),'signed zero-mass splits use source proxies')
  w=[5._dp,0._dp,0._dp,5._dp]; clf%min_samples_leaf=2; call clf%fit(x,int(target),w)
  call require(clf%node_count_==1.and.clf%nodes_(1)%n_samples==2,'minimum leaves count nonzero rows, not multiplicities')
  call require(clf%nodes_(1)%weighted_n_samples==10,'bootstrap multiplicities set weighted mass')
  clf%min_samples_leaf=1; y=[0,0,0,1]; w=1; call clf%fit(x,y,w)
  query=x(:1,:); query(1,1)=ieee_value(0._dp,ieee_quiet_nan); labels=clf%predict(query)
  call require(all(labels==0),'weighted finite-trained trees route missing queries to the larger child')
  call extra%fit(x,y); call extra_reg%fit(x,target)
  call require(extra%max_features_==1.and.extra_reg%max_features_==2,'ExtraTree defaults resolve by estimator family')
  call require(abs(extra%score(x,extra%predict(x))-1)<1.e-14_dp,'ExtraTree classifier score')
  call require(ieee_is_nan(extra_reg%score(x(:1,:),target(:1))),'tree singleton R2 returns quiet NaN')
  x(:,1)=[0._dp,0._dp,1._dp,1._dp]; x(:,2)=[0._dp,1._dp,0._dp,1._dp]
  y=[0,1,1,0]; target=real(y,dp); w=[2._dp,1._dp,1._dp,2._dp]
  bases(1)%name='extra'; allocate(bases(1)%estimator,source=extra)
  voter%voting='soft'; call voter%fit(x,y,bases,w)
  call require(all(voter%predict(x)==y),'voting forwards weights to ExtraTree classifiers')
  call check_classifier_mass(voter%estimators_(1)%estimator)
  call multiple%fit(x,reshape([y,1-y],[4,2]),extra,w)
  call require(all(multiple%predict(x)==reshape([y,1-y],[4,2])),'multioutput accepts weighted ExtraTree classifiers')
  call check_classifier_mass(multiple%estimators_(1)%estimator_)
  call chain%fit(x,reshape([y,1-y],[4,2]),extra,w)
  call require(all(chain%predict(x)==reshape([y,1-y],[4,2])),'classifier chains accept weighted ExtraTrees')
  call check_classifier_mass(chain%estimators_(1)%estimator_)
  call ovr%fit(x,y,extra,w); call ovo%fit(x,y,extra,w); call code%fit(x,y,extra,w)
  call check_classifier_mass(ovr%estimators_(1)%estimator_)
  call check_classifier_mass(ovo%estimators_(1)%estimator_)
  call require(all(ovr%predict(x)==y).and.all(ovo%predict(x)==y).and.all(code%predict(x)==y), &
    'multiclass reductions accept weighted ExtraTrees')
  deallocate(bases(1)%estimator); allocate(bases(1)%estimator,source=extra_reg)
  call voter_reg%fit(x,target,bases,w)
  call multiple_reg%fit(x,reshape([target,2*target+1],[4,2]),extra_reg,w)
  call chain_reg%fit(x,reshape([target,2*target+1],[4,2]),extra_reg,w)
  call require(maxval(abs(voter_reg%predict(x)-target))<1.e-12_dp,'voting forwards weights to ExtraTree regression')
  call require(maxval(abs(multiple_reg%predict(x)-reshape([target,2*target+1],[4,2])))<1.e-12_dp, &
    'multioutput forwards ExtraTree regression weights')
  call require(maxval(abs(chain_reg%predict(x)-reshape([target,2*target+1],[4,2])))<1.e-12_dp, &
    'regressor chains forward ExtraTree weights')
  call require(abs(copy%score(x,[0,0,0,0],w)-2._dp/3)<1.e-14_dp,'tree weighted accuracy')
  call extra_reg%fit(x,target,w)
  call require(abs(extra_reg%score(x,[0._dp,0._dp,1._dp,1._dp],w)+1)<1.e-14_dp,'tree weighted R2')
  w=0; call voter_reg%fit(x,target,bases,w); prediction=voter_reg%predict(x)
  call require(all(ieee_is_nan(prediction)),'voting preserves undefined weighted regression predictions')
  print *, 'weighted and standalone extra trees: PASS'
contains
  subroutine check_classifier_mass(model)
    class(*), intent(in) :: model
    select type(model)
    class is(decision_tree_classifier)
      call require(model%nodes_(1)%weighted_n_samples==6,'wrapper forwards training weight mass')
      call require(abs(model%nodes_(1)%probability(1)-2._dp/3)<1.e-14_dp,'wrapper forwards nonuniform class weights')
    class default
      call require(.false.,'wrapper retains the ExtraTree dynamic type')
    end select
  end subroutine
end program
