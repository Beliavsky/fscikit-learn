program test_column_transformer
  use sklearn_feature_composition
  use sklearn_preprocessing, only: standard_scaler
  use sklearn_feature_extraction, only: feature_string
  use sklearn_function_transformer, only: function_transformer
  use sklearn_common, only: dp,require
  implicit none
  type(column_transformer) :: model,copy
  type(standard_scaler) :: scaler
  type(feature_string) :: input(3),query_names(4)
  type(feature_string), allocatable :: names(:)
  type(function_transformer) :: trimming
  type(column_transformer) :: changed_rows
  real(dp) :: x(5,3),q(5,4)
  real(dp), allocatable :: z(:,:),v(:,:),before(:,:)
  integer :: i
  x=reshape([(real(i,dp),i=1,15)],[5,3])
  input(1)%text='a';input(2)%text='b';input(3)%text='c'
  allocate(model%transformers(3))
  model%transformers(1)%name='scaled';model%transformers(1)%columns=[3,1]
  allocate(model%transformers(1)%model,source=scaler);model%transformers(1)%weight=2
  model%transformers(2)%name='skipped';model%transformers(2)%columns=[2];model%transformers(2)%action='drop'
  model%transformers(3)%name='empty';model%transformers(3)%columns=[integer::]
  allocate(model%transformers(3)%model,source=scaler)
  model%remainder%action='passthrough'
  z=model%fit_transform(x,input)
  call require(all(shape(z)==[5,2]),'dropped selected columns do not enter remainder')
  call require(abs(sum(z))<1.e-12_dp,'parallel selected scaler blocks centered')
  select type(prototype=>model%transformers(1)%model)
  type is(standard_scaler)
    call require(.not.allocated(prototype%mean_),'ColumnTransformer clones branch prototypes')
  end select
  select type(empty=>model%transformers_(3)%model)
  type is(standard_scaler)
    call require(.not.allocated(empty%mean_),'empty selections are not fitted')
  end select
  names=model%get_feature_names_out()
  call require(names(1)%text=='scaled__c'.and.names(2)%text=='scaled__a','selected names preserve order')
  q(:,1)=99;q(:,2)=x(:,3);q(:,3)=x(:,1);q(:,4)=x(:,2)
  query_names(1)%text='extra';query_names(2)%text='c';query_names(3)%text='a';query_names(4)%text='b'
  v=model%transform(q,query_names);call require(all(v==z),'named queries reorder columns and permit extras')
  v=model%transform(q(:,2:3),query_names(2:3))
  call require(all(v==z),'dropped columns may be absent from named queries')
  copy=model;before=model%transform(x);call copy%fit(x*2,input)
  call require(all(model%transform(x)==before),'fitted composition copies are independent')
  model%transformers(1)%weight=3;v=model%transform(x)
  call require(maxval(abs(v-1.5_dp*z))<1.e-12_dp,'branch weights are read at transform time')
  trimming%func=>trim_row;trimming%feature_names_out='one-to-one'
  allocate(changed_rows%transformers(1));changed_rows%transformers(1)%name='trim'
  allocate(changed_rows%transformers(1)%model,source=trimming)
  changed_rows%transformers(1)%columns=[1]
  v=changed_rows%fit_transform(x)
  call require(size(v,1)==size(x,1)-1,'single-branch composition preserves changed row count')
  model%name_formatter=>format_name;names=model%get_feature_names_out()
  call require(names(1)%text=='scaled::c','custom column feature-name formatter')
  print *, 'ColumnTransformer: PASS'
contains
  function format_name(branch,feature) result(name)
    character(*), intent(in) :: branch,feature
    character(:), allocatable :: name
    name=branch//'::'//feature
  end function
  function trim_row(a) result(b)
    real(dp), intent(in) :: a(:,:)
    real(dp), allocatable :: b(:,:)
    b=a(:size(a,1)-1,:)
  end function
end program
