program test_model_selection_features
  use sklearn_common, only: dp,require
  use sklearn_model_selection_features, only: select_from_model
  use sklearn_linear_model, only: linear_regression,ridge,lasso,multi_task_elastic_net,sgd_regressor
  use sklearn_feature_extraction, only: feature_string
  use sklearn_pipeline, only: pipeline
  use sklearn_preprocessing, only: standard_scaler
  use sklearn_ensemble, only: random_forest_regressor,extra_trees_classifier,gradient_boosting_regressor
  use sklearn_tree, only: decision_tree_regressor
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_positive_inf,ieee_negative_inf
  implicit none
  type(select_from_model) :: selector,copied,stream
  type(linear_regression) :: base
  type(ridge) :: regularized
  type(lasso) :: sparse
  type(multi_task_elastic_net) :: multiple
  type(sgd_regressor) :: trained,fresh
  type(pipeline) :: chain,prefix
  type(standard_scaler) :: scale
  type(random_forest_regressor) :: forest
  type(extra_trees_classifier) :: extra
  type(gradient_boosting_regressor) :: boosting
  type(decision_tree_regressor) :: tree
  integer :: labels(40)
  real(dp) :: x(40,5),y(40),w(40),expected(5)
  real(dp), allocatable :: z(:,:),restored(:,:),scores(:),a(:),b(:)
  type(feature_string) :: names(5)
  type(feature_string), allocatable :: selected_names(:)
  logical, allocatable :: mask(:)
  integer, allocatable :: indices(:)
  integer :: i
  do i=1,40
    x(i,:)=[sin(real(i,dp)),cos(real(2*i,dp)),sin(real(3*i,dp)),cos(real(4*i,dp)),sin(real(5*i,dp))]
  end do
  do i=1,5
    names(i)%text=achar(96+i)
  end do
  y=3*x(:,2)-2*x(:,4);w=1;w(:3)=0
  z=selector%fit_transform(x,y,base,sample_weight=w,feature_names=names)
  mask=selector%get_support()
  call require(all(mask.eqv.[.false.,.true.,.false.,.true.,.false.]),'model selector informative coefficients')
  call require(.not.allocated(base%coef_),'selector leaves caller unfitted')
  call require(abs(selector%threshold_value()-1)<1.e-10_dp,'model selector mean threshold')
  restored=selector%inverse_transform(z)
  call require(all(restored(:,2)==x(:,2)).and.all(restored(:,1)==0),'selector inverse zero fills')
  selected_names=selector%get_feature_names_out()
  call require(selected_names(1)%text=='b'.and.selected_names(2)%text=='d','selector fitted feature names')
  selector%threshold='2 * median';scores=selector%feature_importances()
  call require(selector%threshold_value()<1.e-10_dp,'scaled median includes negligible coefficients')
  selector%threshold='value';selector%numeric_threshold=ieee_value(0._dp,ieee_positive_inf)
  z=selector%transform(x)
  call require(size(z,2)==0,'selector empty selection')
  selector=select_from_model();selector%prefit=.true.;base%coef_=[-2._dp,2._dp,0._dp,2._dp,-2._dp]
  allocate(selector%estimator,source=base)
  selector%threshold='value';selector%numeric_threshold=ieee_value(0._dp,ieee_negative_inf);selector%max_features=2
  indices=selector%get_support_indices()
  call require(all(indices==[1,2]),'descending importance stable tie order')
  call selector%fit(x);base%coef_=0;copied=selector
  select type(model=>selector%estimator_)
  type is(linear_regression)
    model%coef_=0
  end select
  scores=copied%feature_importances()
  call require(all(scores==[2._dp,2._dp,0._dp,2._dp,2._dp]),'prefit and selector copies independent')
  copied%max_features=0;copied%max_features_=0;z=copied%transform(x)
  call require(size(z,2)==0,'zero cap allowed')
  selector=select_from_model();sparse%alpha=.05_dp
  call selector%fit(x,y,sparse)
  call require(selector%threshold_value()==1.e-5_dp,'Lasso automatic small threshold')
  call require(count(selector%get_support())==2,'Lasso removes noise columns')
  selector=select_from_model();selector%prefit=.true.
  multiple%coef_=reshape([1._dp,-2._dp,0._dp,0._dp,3._dp,4._dp,0._dp,2._dp,-1._dp,1._dp],[2,5])
  allocate(selector%estimator,source=multiple);selector%norm_order=2
  scores=selector%feature_importances();expected=[sqrt(5._dp),0._dp,5._dp,2._dp,sqrt(2._dp)]
  call require(maxval(abs(scores-expected))<1.e-12_dp,'matrix coefficient Euclidean norms')
  selector%norm_order=-1;scores=selector%feature_importances()
  call require(maxval(abs(scores-[2._dp/3,0._dp,12._dp/7,0._dp,.5_dp]))<1.e-12_dp,'negative coefficient norms')
  selector%norm_order=ieee_value(0._dp,ieee_positive_inf);scores=selector%feature_importances()
  call require(all(scores==[2._dp,0._dp,4._dp,2._dp,1._dp]),'infinite coefficient norm')
  selector%norm_order=ieee_value(0._dp,ieee_negative_inf);scores=selector%feature_importances()
  call require(all(scores==[1._dp,0._dp,3._dp,0._dp,1._dp]),'negative infinite coefficient norm')
  trained%learning_rate='constant';trained%eta0=.01_dp;trained%max_iter=3;trained%tol=-1
  call trained%fit(x,y);fresh=trained
  ! Compare a selector cloned from an already-trained estimator with a fresh epoch.
  fresh%max_iter=1;call fresh%fit(x,y)
  allocate(stream%estimator,source=trained);call stream%partial_fit(x,y)
  select type(model=>stream%estimator_)
  type is(sgd_regressor)
    call require(maxval(abs(model%coef_-fresh%coef_))<1.e-13_dp,'first partial_fit clones parameters only')
  end select
  call fresh%partial_fit(x,y);call stream%partial_fit(x,y)
  select type(model=>stream%estimator_)
  type is(sgd_regressor)
    call require(maxval(abs(model%coef_-fresh%coef_))<1.e-13_dp,'subsequent partial_fit preserves state')
  end select
  selector=select_from_model();selector%max_features_func=>half_features
  call selector%fit(x,y,regularized)
  call require(selector%max_features_==2.and.count(selector%get_support())==2,'callable feature cap')
  selector%importance_getter='callback';selector%importance_func=>reversed_importance
  scores=selector%feature_importances()
  call require(all(scores==[5._dp,4._dp,3._dp,2._dp,1._dp]),'importance callback')
  selector=select_from_model();forest%n_estimators=3;forest%max_depth=3
  call forest%fit(x,y);call selector%fit(x,y,forest)
  scores=selector%feature_importances()
  call require(maxval(abs(scores-forest%feature_importances_))<1.e-12_dp,'forest importance adapter')
  boosting%n_estimators=3;call boosting%fit(x,y);call selector%fit(x,y,boosting)
  scores=selector%feature_importances()
  call require(maxval(abs(scores-boosting%feature_importances_))<1.e-12_dp,'boosting importance adapter')
  labels=merge(9,-4,y>0);extra%n_estimators=3;extra%max_depth=3
  call extra%fit(x,labels);call selector%fit(x,real(labels,dp),extra)
  scores=selector%feature_importances()
  call require(maxval(abs(scores-extra%feature_importances_))<1.e-12_dp,'classifier role from real labels')
  tree%max_depth=2;call tree%fit(x,real(labels,dp));call selector%fit(x,labels,tree)
  scores=selector%feature_importances()
  call require(maxval(abs(scores-tree%feature_importances_))<1.e-12_dp,'regressor role from integer targets')
  selector=select_from_model();allocate(selector%estimator,source=regularized)
  allocate(chain%steps(3));chain%steps(1)%name='scale';chain%steps(2)%name='select';chain%steps(3)%name='regress'
  allocate(chain%steps(1)%model,source=scale);allocate(chain%steps(2)%model,source=selector)
  allocate(chain%steps(3)%model,source=regularized)
  call chain%fit(x,labels,feature_names=names);prefix=chain%slice(1,2)
  z=prefix%transform(x);restored=prefix%inverse_transform(z);selected_names=prefix%get_feature_names_out()
  a=chain%predict(x)
  call require(size(z,2)>0.and.size(z,2)==size(selected_names),'selector pipeline names and transform')
  call require(all(shape(restored)==shape(x)).and.all(abs(a)<100),'selector pipeline inverse and prediction')
  print *, 'SelectFromModel tests passed'
contains
  integer function half_features(x) result(n)
    real(dp), intent(in) :: x(:,:)
    n=size(x,2)/2
  end function
  subroutine reversed_importance(model,values,matrix)
    class(*), intent(in) :: model
    real(dp), allocatable, intent(out) :: values(:,:)
    logical, intent(out) :: matrix
    values=reshape([5._dp,4._dp,3._dp,2._dp,1._dp],[1,5]);matrix=.false.
  end subroutine
end program
