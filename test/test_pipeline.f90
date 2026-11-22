program test_pipeline
  use sklearn_pipeline
  use sklearn_compose, only: transformed_target_regressor,column_transformer
  use sklearn_preprocessing, only: standard_scaler,minmax_scaler
  use sklearn_feature_selection, only: select_k_best
  use sklearn_linear_model, only: ridge
  use sklearn_feature_extraction, only: feature_string
  use sklearn_common, only: dp,require
  implicit none
  type(pipeline) :: model,copy,prefix,inner,outer
  type(standard_scaler) :: standard
  type(minmax_scaler) :: minmax
  type(ridge) :: base
  type(select_k_best) :: selection
  type(transformed_target_regressor) :: direct,matrix_target
  type(column_transformer) :: column
  type(feature_string), allocatable :: names(:)
  type(feature_string) :: supplied(3)
  real(dp) :: x(30,3),y(30),multi(30,2),w(30)
  real(dp), allocatable :: p(:),reference(:),z(:,:),before(:,:),reconstructed(:,:)
  integer :: i
  do i=1,30
    x(i,:)=[real(i,dp)/30,sin(real(i,dp)),cos(real(i,dp))];y(i)=1+2*x(i,1)-x(i,2)+.3_dp*x(i,3)
  end do
  w=1;w(:3)=0;base%alpha=.2_dp
  allocate(model%steps(2));model%steps(1)%name='scale';allocate(model%steps(1)%model,source=standard)
  model%steps(2)%name='ridge';allocate(model%steps(2)%model,source=base)
  call model%fit(x,y,w);p=model%predict(x)
  call standard%fit(x);z=standard%transform(x);allocate(direct%regressor,source=base)
  call direct%fit(z,y,w);reference=direct%predict(z)
  call require(maxval(abs(p-reference))<1.e-12_dp,'Pipeline weighted ridge delegates transformed data')
  call require(model%score(x,y,w)>.98_dp,'Pipeline weighted original-scale score')
  prefix=model%slice(1,1);names=prefix%get_feature_names_out()
  call require(size(names)==3.and.names(1)%text=='x0','Pipeline prefix names')
  before=prefix%transform(x);reconstructed=prefix%inverse_transform(before)
  call require(maxval(abs(reconstructed-x))<1.e-14_dp,'Pipeline inverse transform')
  copy=model;call copy%fit(x*2,y)
  call require(all(model%predict(x)==p),'Pipeline copy refit independence')
  select type(regressor=>copy%steps(2)%model)
  type is(ridge)
    regressor%alpha=0
  end select
  call copy%fit(x,[(i,i=1,30)])
  call require(maxval(abs(copy%predict(x)-[(real(i,dp),i=1,30)]))<1.e-11_dp, &
    'Pipeline final estimator determines regression role for integer targets')
  deallocate(model%steps(2)%model);allocate(model%steps(2)%model,source=matrix_target)
  multi(:,1)=y;multi(:,2)=2*y+1;call model%fit(x,multi,w);z=model%predict_matrix(x)
  call require(all(shape(z)==[30,2]).and.model%score(x,multi,w)>.999_dp,'Pipeline matrix-target tail')
  allocate(inner%steps(2));inner%steps(1)%name='scale';allocate(inner%steps(1)%model,source=standard)
  inner%steps(2)%name='range';allocate(inner%steps(2)%model,source=minmax)
  allocate(column%transformers(1));column%transformers(1)%name='sequence';column%transformers(1)%columns=[3,1]
  allocate(column%transformers(1)%model,source=inner);z=column%fit_transform(x)
  call require(all(shape(z)==[30,2]).and.minval(z)>=0.and.maxval(z)<=1,'Pipeline inside ColumnTransformer')
  names=column%get_feature_names_out()
  call require(names(1)%text=='sequence__x2'.and.names(2)%text=='sequence__x0','nested pipeline names')
  do i=1,3
    supplied(i)%text='field'//achar(48+i)
  end do
  allocate(outer%steps(2));outer%steps(1)%name='off';outer%steps(1)%action='passthrough'
  outer%steps(2)%name='nested';allocate(outer%steps(2)%model,source=model)
  call outer%fit(x,multi,feature_names=supplied)
  select type(fitted=>outer%steps(2)%model)
  type is(pipeline)
    call require(allocated(fitted%feature_names_in_),'named input reaches final nested pipeline')
    call require(fitted%feature_names_in_(1)%text=='field1','nested final input names preserved')
  end select
  deallocate(inner%steps);allocate(inner%steps(2))
  inner%steps(1)%name='select';selection%k=2;allocate(inner%steps(1)%model,source=selection)
  inner%steps(2)%name='fit';allocate(inner%steps(2)%model,source=base)
  deallocate(outer%steps(2)%model);allocate(outer%steps(2)%model,source=inner)
  call outer%fit(x,[(mod(i,3),i=1,30)],feature_names=supplied)
  select type(fitted=>outer%steps(2)%model)
  type is(pipeline)
    call require(fitted%steps(1)%integer_target_,'integer regression targets reach nested supervised steps unchanged')
  end select
  print *, 'Pipeline: PASS'
end program
