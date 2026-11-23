program test_polynomial_composition
  use sklearn_common, only: dp,require
  use sklearn_polynomial, only: polynomial_features
  use sklearn_pipeline, only: pipeline
  use sklearn_feature_composition, only: feature_union,column_transformer
  use sklearn_feature_extraction, only: feature_string
  use sklearn_linear_model, only: ridge
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  type(polynomial_features) :: poly
  type(pipeline) :: model
  type(feature_union) :: joined
  type(column_transformer) :: columns
  type(ridge) :: regression,direct
  type(feature_string) :: names(2)
  type(feature_string), allocatable :: output_names(:)
  real(dp) :: x(6,2),y(6)
  real(dp), allocatable :: z(:,:),expected(:,:),prediction(:)
  integer :: i
  x(1,:)=[1._dp,2._dp];x(2,:)=[2._dp,-1._dp];x(3,:)=[0._dp,1._dp]
  x(4,:)=[-1._dp,2._dp];x(5,:)=[-2._dp,0._dp];x(6,:)=[1._dp,-2._dp]
  y=1+2*x(:,1)**2-3*x(:,2)
  names(1)%text='east';names(2)%text='north';poly%include_bias=.false.;regression%alpha=.5_dp
  allocate(expected(6,5));expected(:,1:2)=x
  expected(:,3)=x(:,1)**2;expected(:,4)=x(:,1)*x(:,2);expected(:,5)=x(:,2)**2
  allocate(model%steps(2));model%steps(1)%name='poly';model%steps(2)%name='ridge'
  allocate(model%steps(1)%model,source=poly);allocate(model%steps(2)%model,source=regression)
  direct=regression
  call model%fit(x,y,feature_names=names);call direct%fit(expected,y)
  prediction=model%predict(x)
  call require(maxval(abs(prediction-direct%predict(expected)))<1.e-12_dp,'polynomial pipeline dispatch')
  allocate(joined%transformer_list(2));joined%transformer_list(1)%name='poly'
  allocate(joined%transformer_list(1)%model,source=poly)
  joined%transformer_list(2)%name='raw';joined%transformer_list(2)%action='passthrough'
  z=joined%fit_transform(x,names)
  call require(all(z(:,:5)==expected).and.all(z(:,6:)==x),'feature union polynomial and raw blocks')
  output_names=joined%get_feature_names_out()
  call require(output_names(3)%text=='poly__east^2'.and.output_names(7)%text=='raw__north','composed polynomial names')
  allocate(columns%transformers(1));columns%transformers(1)%name='poly';columns%transformers(1)%columns=[2]
  allocate(columns%transformers(1)%model,source=poly);columns%remainder%action='passthrough'
  z=columns%fit_transform(x,names)
  call require(all(z(:,1)==x(:,2)).and.all(z(:,2)==x(:,2)**2).and.all(z(:,3)==x(:,1)), &
    'column selection and passthrough with polynomial features')
  call require(poly%n_features_in_==0,'polynomial composition prototype preserved')
  print *, 'PolynomialFeatures composition: PASS'
end program
