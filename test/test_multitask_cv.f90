program test_multitask_cv
  use sklearn_multitask, only: multi_task_elastic_net,multi_task_elastic_net_cv,multi_task_lasso_cv
  use sklearn_common, only: dp,require
  implicit none
  type(multi_task_lasso_cv) :: sparse
  type(multi_task_elastic_net_cv) :: net
  type(multi_task_elastic_net) :: base,copy
  real(dp) :: x(4,1),y(4,2),small_x(2,2),small_y(2,2),before(4,1),yb(4,2)
  real(dp), allocatable :: prediction(:,:)
  x(:,1)=[-2._dp,-1._dp,1._dp,2._dp]; y(:,1)=3*x(:,1); y(:,2)=4*x(:,1); before=x; yb=y
  sparse%fit_intercept=.false.; sparse%cv=2; sparse%tol=1.e-12_dp; sparse%alphas=[.1_dp,1._dp,.5_dp]
  call sparse%fit(x,y)
  call require(maxval(abs(sparse%alphas_(:,1)-[1._dp,.5_dp,.1_dp]))<1.e-14_dp,'MultiTask CV descending alpha grid')
  call require(maxval(abs(sparse%mse_path_(:,1,1)-[.2_dp,.05_dp,.002_dp]))<1.e-12_dp,'MultiTask CV analytic fold errors')
  call require(abs(sparse%alpha_-.1_dp)<1.e-14_dp.and.abs(sparse%best_mse_-.002_dp)<1.e-13_dp, &
    'MultiTask CV analytic selected penalty')
  call require(maxval(abs(sparse%coef_(:,1)-[2.976_dp,3.968_dp]))<1.e-12_dp,'MultiTask CV cold full-data refit')
  call require(all(sparse%fold_test_sizes_==2).and.sparse%all_cv_converged_,'MultiTask CV fold diagnostics')
  call require(abs(sparse%score(x,y)-.999936_dp)<1.e-12_dp,'MultiTask CV uniform multi-output R2')
  call require(all(x==before).and.all(y==yb),'MultiTask CV preserves inputs')
  net%fit_intercept=.false.; net%alphas=[0._dp]; net%l1_ratios=[0._dp,1._dp]; net%tol=1.e-12_dp
  net%test_fold=[-1,9,-1,-1]
  call net%fit(x,y)
  call require(net%n_splits_==1.and.all(net%fold_labels_==9),'MultiTask CV predefined single holdout')
  call require(net%alpha_==0.and.net%l1_ratio_==0,'MultiTask CV first-ratio exact tie')
  call require(maxval(abs(net%coef_(:,1)-[3._dp,4._dp]))<1.e-13_dp,'MultiTask CV zero-penalty least squares')
  deallocate(net%test_fold,net%alphas,net%l1_ratios)
  net%cv=2; net%n_alphas=5; net%max_iter=1000000000
  call net%fit(x,0*y)
  call require(all(net%alphas_==1.e-15_dp).and.net%n_iter_==1,'MultiTask CV zero-target grid floor/huge budget')
  call require(all(net%coef_==0),'MultiTask CV zero-target coefficients')
  small_x=0; small_x(1,1)=1; small_y=0; small_y(1,:)=[3._dp,4._dp]
  base%fit_intercept=.false.; base%tol=1.e-12_dp; base%alpha=0
  call base%fit(small_x,small_y)
  call require(maxval(abs(base%coef_(:,1)-[3._dp,4._dp]))<1.e-13_dp,'MultiTask base zero alpha')
  base%alpha=.5_dp; base%l1_ratio=0; call base%fit(small_x,small_y)
  call require(maxval(abs(base%coef_(:,1)-[1.5_dp,2._dp]))<1.e-13_dp,'MultiTask base pure L2 closed form')
  base%selection='random'; base%random_state=42; base%l1_ratio=.5_dp; copy=base
  call base%fit(small_x,small_y); call copy%fit(small_x,small_y)
  call require(all(base%coef_==copy%coef_).and.base%n_iter_==copy%n_iter_,'MultiTask seeded random coordinate repeat')
  prediction=sparse%predict(x)
  call require(all(shape(prediction)==[4,2]),'MultiTask CV inherited prediction layout')
  print *, 'Multi-task cross-validation tests passed'
end program
