program test_pipeline_supervised
  use sklearn_pipeline
  use sklearn_preprocessing, only: standard_scaler,target_encoder
  use sklearn_linear_model, only: logistic_regression,linear_regression
  use sklearn_feature_selection, only: select_k_best
  use sklearn_common, only: dp,require
  implicit none
  type(pipeline) :: model,prefix,regression
  type(standard_scaler) :: scaler
  type(logistic_regression) :: classifier
  type(linear_regression) :: linear
  type(select_k_best) :: selection
  type(target_encoder) :: encoder
  integer :: target(60),i
  real(dp) :: x(60,3),categorical(60,2),response(60)
  real(dp), allocatable :: proba(:,:),z(:,:),full(:,:)
  integer, allocatable :: predicted(:)
  do i=1,60
    x(i,:)=[sin(real(i,dp)),cos(real(i,dp)),real(i,dp)/60]
    target(i)=merge(-2,7,x(i,1)>.1_dp)
    categorical(i,:)=[real(mod(i,5),dp),real(mod(i,7),dp)]
    response(i)=real(mod(i,5),dp)+.3_dp*sin(real(i,dp))
  end do
  selection%k=2;classifier%max_iter=200
  allocate(model%steps(3));model%steps(1)%name='select';allocate(model%steps(1)%model,source=selection)
  model%steps(2)%name='scale';allocate(model%steps(2)%model,source=scaler)
  model%steps(3)%name='classify';allocate(model%steps(3)%model,source=classifier)
  call model%fit(x,target);predicted=model%predict_class(x);proba=model%predict_proba(x)
  call require(model%score(x,target)>.95_dp,'Pipeline supervised selection and classification')
  call require(all(model%classes_==[-2,7]).and.all(shape(proba)==[60,2]),'Pipeline class metadata and probability shape')
  call require(maxval(abs(sum(proba,dim=2)-1))<1.e-14_dp,'Pipeline class probabilities normalized')
  call model%fit(x,real(target,dp))
  call require(model%score(x,real(target,dp))>.95_dp,'Pipeline classifier accepts integral real targets')
  encoder%target_type='continuous' ;encoder%cv=3;encoder%auto_smooth=.false.;encoder%smooth=1
  allocate(regression%steps(2));regression%steps(1)%name='encode';allocate(regression%steps(1)%model,source=encoder)
  regression%steps(2)%name='fit';allocate(regression%steps(2)%model,source=linear)
  call regression%fit(categorical,response)
  prefix=regression%slice(1,1);full=prefix%transform(categorical)
  z=prefix%fit_transform(categorical,response)
  call require(maxval(abs(z-full))>1.e-3_dp,'Pipeline TargetEncoder uses cross-fitted training features')
  call require(regression%score(categorical,response)>.9_dp,'Pipeline target encoding regression')
  print *, 'Supervised pipeline: PASS'
end program
