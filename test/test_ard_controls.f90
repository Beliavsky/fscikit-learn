program test_ard_controls
  use sklearn
  use sklearn_common,only:require
  use sklearn_search_adapters,only:reset_search_clone,set_search_params
  use sklearn_parameter_space,only:parameter_set,param
  implicit none
  type(ard_regression)::model,copy
  type(parameter_set)::parameters
  real(dp)::x(25,4),y(25),before(25,4),gram(4,4),eye(4,4)
  integer::i
  do i=1,25
    x(i,:)=[sin(real(i,dp)),cos(real(i,dp)),real(i,dp)/25,real(mod(i,3),dp)]
  end do
  y=.5_dp+x(:,1)-.3_dp*x(:,3)+.01_dp*x(:,2);before=x
  model%compute_score=.true.;model%threshold_lambda=1.e20_dp;call model%fit(x,y)
  call require(size(model%scores_)==model%n_iter_.and.model%n_features_in_==4,'ARD history and metadata')
  gram=model%alpha_*matmul(transpose(x-spread(model%x_offset_,1,25)),x-spread(model%x_offset_,1,25));eye=0
  do i=1,4
    gram(i,i)=gram(i,i)+model%lambda_(i);eye(i,i)=1
  end do
  call require(maxval(abs(matmul(gram,model%sigma_)-eye))<1.e-7_dp,'ARD posterior inverse')
  copy=model;call reset_search_clone(copy)
  call require(.not.allocated(copy%coef_).and.copy%compute_score,'ARD constructor clone')
  call copy%fit(x,reshape(y,[25,1]));call require(maxval(abs(copy%coef_-model%coef_))<1.e-12_dp,'ARD column target')
  allocate(parameters%items(2));parameters%items(1)%name='threshold_lambda';parameters%items(1)%value=param(0._dp)
  parameters%items(2)%name='compute_score';parameters%items(2)%value=param(.false.)
  call set_search_params(copy,parameters);call copy%fit(x,y)
  call require(all(copy%coef_==0).and.size(copy%sigma_)==0.and.size(copy%scores_)==0,'ARD threshold zero and reset')
  call require(all(abs(copy%predict_std(x)-sqrt(1/copy%alpha_))<1.e-12_dp),'ARD empty posterior uncertainty')
  call require(all(x==before),'ARD preserves input')
  print *,'ARD covariance, scores, column targets, clones and controls: PASS'
end program
