program sequential_pipeline
  use sklearn, only: dp,pipeline,standard_scaler,ridge,column_transformer,simple_imputer,feature_string
  implicit none
  type(pipeline) :: model,numeric
  type(column_transformer) :: columns
  type(standard_scaler) :: scaler
  type(simple_imputer) :: imputer
  type(ridge) :: regression
  type(feature_string), allocatable :: names(:)
  real(dp) :: x(20,3),y(20)
  real(dp), allocatable :: prediction(:)
  integer :: i
  do i=1,20
    x(i,:)=[real(i,dp)/20,sin(real(i,dp)),cos(real(i,dp))]
    y(i)=1+2*x(i,1)-x(i,2)
  end do
  allocate(numeric%steps(2));numeric%steps(1)%name='impute';allocate(numeric%steps(1)%model,source=imputer)
  numeric%steps(2)%name='scale';allocate(numeric%steps(2)%model,source=scaler)
  allocate(columns%transformers(1));columns%transformers(1)%name='numeric';columns%transformers(1)%columns=[1,2]
  allocate(columns%transformers(1)%model,source=numeric)
  allocate(model%steps(2));model%steps(1)%name='prepare';allocate(model%steps(1)%model,source=columns)
  regression%alpha=.1_dp;model%steps(2)%name='regress';allocate(model%steps(2)%model,source=regression)
  call model%fit(x,y);prediction=model%predict(x)
  block
    type(pipeline) :: preparation
    preparation=model%slice(1,1);names=preparation%get_feature_names_out()
  end block
  print '(a,f10.7)', 'Training R2: ',model%score(x,y)
  print '(a,a)', 'First prepared feature: ',names(1)%text
  print '(a,i0)', 'Predictions: ',size(prediction)
end program
