program test_bayesian_ridge_controls
  use sklearn
  use sklearn_common,only:require
  use sklearn_search_adapters,only:reset_search_clone,set_search_params
  use sklearn_parameter_space,only:parameter_set,param
  implicit none
  type(bayesian_ridge)::model,copy,plain
  type(parameter_set)::parameters
  type(pipeline)::pipe
  real(dp)::x(23,3),y(23),weight(23),before(23,3),precision(3,3),identity(3,3)
  integer::i
  do i=1,23
    x(i,:)=[real(i-12,dp)/23,sin(real(i,dp))*.3_dp,real(mod(i,4),dp)*.1_dp]
  end do
  y=.7_dp+.4_dp*x(:,1)-.6_dp*x(:,2)+.01_dp*cos(real([(i,i=1,23)],dp))
  before=x;weight=1;weight(:3)=0;model%compute_score=.true.;model%tol=1.e-9_dp
  call model%fit(x,y,weight)
  call require(model%n_features_in_==3.and.size(model%scores_)==model%n_iter_+1,'Bayesian score history and metadata')
  copy=model;call reset_search_clone(copy)
  call require(.not.allocated(copy%coef_).and.copy%compute_score,'Bayesian parameter-only clone')
  call copy%fit(x,reshape(y,[23,1]),weight)
  call require(maxval(abs(copy%coef_-model%coef_))<1.e-10_dp,'Bayesian column fit')
  call require(abs(copy%score(x,y,weight)-model%score(x,y,weight))<1.e-12_dp,'Bayesian weighted score')
  call plain%fit(x(4:,:),y(4:))
  ! Removing zero-weight rows changes Source n_samples in precision updates;
  ! weighted fitting intentionally retains all rows rather than dropping them.
  call require(all(x==before),'Bayesian fitting preserves design')
  precision=model%alpha_*matmul(transpose(x-spread(model%x_offset_,1,23)), &
    (x-spread(model%x_offset_,1,23))*spread(weight,2,3))
  identity=0
  do i=1,3
    precision(i,i)=precision(i,i)+model%lambda_;identity(i,i)=1
  end do
  call require(maxval(abs(matmul(precision,model%sigma_)-identity))<1.e-8_dp,'Bayesian full-rank posterior covariance')
  allocate(pipe%steps(1));pipe%steps(1)%name='bayes';allocate(pipe%steps(1)%model,source=copy)
  call pipe%fit(x,y,weight)
  call require(maxval(abs(pipe%predict(x)-copy%predict(x)))<1.e-9_dp,'weighted Bayesian Pipeline')
  allocate(parameters%items(2));parameters%items(1)%name='alpha_init';parameters%items(1)%value=param('none')
  parameters%items(2)%name='compute_score';parameters%items(2)%value=param(.false.)
  call set_search_params(copy,parameters);call copy%fit(x,y,2._dp)
  call require(size(copy%scores_)==0.and.copy%alpha_init==-1,'Bayesian search controls and score reset')
  print *,'BayesianRidge weighted fits, covariance, scores, clones and Pipeline: PASS'
end program
