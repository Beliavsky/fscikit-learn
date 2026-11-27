program test_huber_controls
  use sklearn_linear_model,only:huber_regressor
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  use sklearn_parameter_space,only:parameter_set,param
  use sklearn_common,only:dp,require
  use sklearn_search_adapters,only:set_search_params,reset_search_clone
  use sklearn_base_dispatch,only:base_weighted,train_regressor
  use sklearn_sgd_clone,only:reset_sgd_fit
  implicit none
  type(huber_regressor)::model,copy,dense
  type(csr_matrix)::sparse
  type(parameter_set)::parameters
  real(dp)::x(41,2),y(41),weight(41),before(41,2),coef(2),score
  real(dp),allocatable::prediction(:)
  integer::i
  do i=1,41
    x(i,:)=[sin(real(i,dp)),cos(real(2*i,dp))]
    y(i)=1.5_dp*x(i,1)-.6_dp*x(i,2)+.3_dp+.03_dp*sin(real(3*i,dp))
    if(mod(i,7)==0)y(i)=y(i)+5
    weight(i)=.5_dp+real(mod(i,5),dp)*.2_dp
  end do
  before=x;sparse=csr_from_dense(x)
  allocate(parameters%items(3))
  parameters%items(1)%name='alpha';parameters%items(1)%value=param(.01_dp)
  parameters%items(2)%name='epsilon';parameters%items(2)%value=param(1.8_dp)
  parameters%items(3)%name='max_iter';parameters%items(3)%value=param(200)
  call set_search_params(model,parameters)
  model%warm_start=.true.
  call model%fit(x,y,weight)
  call require(model%n_features_in_==2.and.size(model%outliers_)==41,'Huber learned shape and outlier state')
  call require(model%scale_>0.and.model%n_iter_>0,'Huber bounded scale and optimizer iterations')
  call require(all(x==before),'Huber dense input preservation')
  call require(base_weighted(model),'Huber weighted registry support')
  call require(maxval(abs(model%coef_-[1.5_dp,-.6_dp]))<.04_dp,'Huber weighted robust recovery')
  copy=model;call reset_search_clone(copy)
  call require(.not.allocated(copy%coef_).and..not.allocated(copy%outliers_),'Huber search clone discards fit state')
  call require(copy%warm_start.and.copy%alpha==model%alpha.and.copy%epsilon==model%epsilon,'Huber clone retains controls')
  call copy%fit(sparse,reshape(y,[41,1]),weight)
  call require(maxval(abs(copy%predict(sparse)-model%predict(x)))<1.e-7_dp,'Huber dense and CSR weighted fit')
  score=copy%score(sparse,y,weight)
  call require(abs(score-model%score(x,y,weight))<1.e-8_dp,'Huber dense and CSR weighted score')
  dense=model;call reset_sgd_fit(dense)
  call train_regressor(dense,x,y,weight)
  call require(maxval(abs(dense%coef_-model%coef_))<1.e-10_dp,'Huber common dispatch uses weighted Huber fit')
  coef=model%coef_;call model%fit(x,y,weight)
  call require(maxval(abs(model%coef_-coef))<1.e-4_dp,'Huber warm repeated fit remains at optimum')
  copy%fit_intercept=.false.;call reset_search_clone(copy)
  call copy%fit(sparse,y,2._dp);prediction=copy%predict(sparse)
  call copy%fit(sparse,reshape(y,[41,1]),2._dp)
  call require(copy%intercept_==0.and.all(abs(copy%coef_)<10),'Huber Source warm no-intercept convention is bounded')
  call require(all(sparse%to_dense()==before),'Huber CSR input preservation')
  copy%max_iter=0;copy%warm_start=.false.;call copy%fit(x,y)
  call require(copy%n_iter_==0.and..not.copy%converged_,'Huber zero iteration control follows Source counter')
  print *, 'Huber weighted dense/CSR fits, scalar weights, warm starts, clone resets and registry: PASS'
end program
