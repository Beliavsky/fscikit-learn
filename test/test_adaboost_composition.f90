program test_adaboost_composition
  use sklearn_common, only: dp,require
  use sklearn_adaboost, only: adaboost_classifier
  use sklearn_pipeline, only: pipeline
  use sklearn_preprocessing, only: standard_scaler
  use sklearn_base_dispatch, only: seed_base,base_supported,base_weighted,base_proba,base_decision,base_fitted
  implicit none
  type(adaboost_classifier) :: base,direct
  type(standard_scaler) :: scale
  type(pipeline) :: model,copy
  real(dp) :: x(8,2),w(8)
  real(dp), allocatable :: z(:,:),a(:,:),b(:,:)
  integer :: y(8)
  integer, allocatable :: prediction(:)
  x(:,1)=[-1._dp,-1._dp,1._dp,1._dp,-2._dp,-2._dp,2._dp,2._dp]
  x(:,2)=[-1._dp,1._dp,-1._dp,1._dp,-2._dp,2._dp,-2._dp,2._dp]
  y=[-4,7,7,-4,-4,7,7,-4];w=[0._dp,1._dp,2._dp,3._dp,1._dp,2._dp,3._dp,4._dp]
  base%max_depth=2
  call seed_base(base,42)
  call require(base%tree_random_state==1608637542.and.base%random_state==1273642419, &
    'explicit CART nested seeds follow sklearn sorted parameter order')
  call require(base_supported(base,.true.).and.base_weighted(base).and.base_proba(base).and.base_decision(base), &
    'AdaBoost shared classification capabilities')
  call require(.not.base_fitted(base),'unfitted AdaBoost dispatch')
  allocate(model%steps(2));model%steps(1)%name='scale';allocate(model%steps(1)%model,source=scale)
  model%steps(2)%name='boost';allocate(model%steps(2)%model,source=base)
  call model%fit(x,y,w);prediction=model%predict_class(x)
  call require(all(prediction==y),'weighted AdaBoost pipeline labels')
  call scale%fit(x);z=scale%transform(x);direct=base;call direct%fit(z,y,w)
  a=model%predict_proba(x);b=direct%predict_proba(z)
  call require(all(a==b),'pipeline probability dispatch')
  a=model%decision_function(x);b=direct%decision_function(z)
  call require(all(a==b).and.model%score(x,y,w)==1,'pipeline decision and weighted score dispatch')
  call require(.not.base_fitted(base),'pipeline preserves original unfitted base')
  copy=model;call copy%fit(x,spread(21,1,8));prediction=model%predict_class(x)
  call require(all(prediction==y),'copied pipeline refit is independent')
  print *, 'AdaBoost composition: PASS'
end program
