program test_tree_ensemble
  use sklearn_common
  use sklearn_tree
  use sklearn_ensemble
  implicit none
  type(decision_tree_classifier) :: tc
  type(decision_tree_regressor) :: tr
  type(random_forest_classifier) :: fc
  type(random_forest_regressor) :: fr
  type(extra_trees_classifier) :: ec
  type(extra_trees_regressor) :: er
  type(gradient_boosting_regressor) :: gb
  real(dp) :: x(30,2),y(30)
  real(dp), allocatable :: prob(:,:)
  integer :: labels(30),i
  do i=1,30
    x(i,:)=[real(i,dp)/10,cos(real(i,dp))]; y(i)=x(i,1)**2
    labels(i)=merge(7,-3,i>15)
  end do
  call tc%fit(x,labels); call require(tc%score(x,labels)==1,'tree classification')
  call require(tc%get_depth()==1.and.tc%get_n_leaves()==2,'tree structure')
  call require(abs(tc%feature_importances_(1)-1)<1.e-12_dp,'tree feature importance')
  call tr%fit(x,y); call require(tr%score(x,y)>0.999999_dp,'tree regression')
  tr%max_depth=2; call tr%fit(x,y); call require(tr%get_depth()==2,'tree depth control')
  fc%n_estimators=15; call fc%fit(x,labels); prob=fc%predict_proba(x)
  call require(fc%score(x,labels)>0.95_dp,'forest classification')
  call require(maxval(abs(sum(prob,dim=2)-1))<1.e-12_dp,'forest probabilities')
  fr%n_estimators=15; call fr%fit(x,y); call require(fr%score(x,y)>0.98_dp,'forest regression')
  ec%n_estimators=8; call ec%fit(x,labels); call require(ec%score(x,labels)==1,'extra trees classification')
  er%n_estimators=8; call er%fit(x,y); call require(er%score(x,y)>0.999999_dp,'extra trees regression')
  gb%n_estimators=40; call gb%fit(x,y); call require(gb%score(x,y)>0.999_dp,'gradient boosting')
  call require(gb%train_score_(40)<gb%train_score_(1),'boosting loss decreases')
  call fc%fit(x(:5,:),labels(:5)); call require(fc%score(x(:5,:),labels(:5))==1,'single class forest')
  print *, 'trees and ensembles: PASS'
end program
