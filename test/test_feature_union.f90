program test_feature_union
  use sklearn_feature_composition
  use sklearn_preprocessing, only: standard_scaler
  use sklearn_function_transformer, only: function_transformer
  use sklearn_common, only: dp,require
  implicit none
  type(feature_union) :: model,copy
  type(column_transformer) :: nested
  type(standard_scaler) :: scaler
  type(function_transformer) :: trimming
  type(feature_union) :: changed_rows
  real(dp) :: x(4,2)
  real(dp), allocatable :: z(:,:),v(:,:)
  integer :: i
  x=reshape([(real(i,dp),i=1,8)],[4,2])
  allocate(nested%transformers(1))
  nested%transformers(1)%name='one';nested%transformers(1)%action='passthrough'
  nested%transformers(1)%columns=[2]
  allocate(model%transformer_list(3))
  model%transformer_list(1)%name='raw';model%transformer_list(1)%action='passthrough'
  model%transformer_list(1)%weight=-2
  model%transformer_list(2)%name='center';allocate(model%transformer_list(2)%model,source=scaler)
  model%transformer_list(3)%name='nested';allocate(model%transformer_list(3)%model,source=nested)
  z=model%fit_transform(x)
  call require(all(shape(z)==[4,5]),'FeatureUnion concatenated widths')
  call require(all(z(:,:2)==-2*x).and.all(z(:,5)==x(:,2)),'weighted passthrough and recursive graph')
  select type(fitted=>model%transformer_list(2)%model)
  type is(standard_scaler)
    call require(allocated(fitted%mean_),'FeatureUnion transformer_list contains fitted models')
  end select
  copy=model;call copy%fit(x*2);v=model%transform(x)
  call require(all(v==z),'FeatureUnion copy refit independence')
  deallocate(model%transformer_list);allocate(model%transformer_list(0));z=model%fit_transform(x)
  call require(all(shape(z)==[4,0]),'empty union returns zero columns')
  trimming%func=>trim_row;trimming%feature_names_out='one-to-one'
  allocate(changed_rows%transformer_list(1));changed_rows%transformer_list(1)%name='trim'
  allocate(changed_rows%transformer_list(1)%model,source=trimming)
  v=changed_rows%fit_transform(x)
  call require(size(v,1)==size(x,1)-1,'single-branch composition preserves changed row count')
  print *, 'FeatureUnion: PASS'
contains
  function trim_row(a) result(b)
    real(dp), intent(in) :: a(:,:)
    real(dp), allocatable :: b(:,:)
    b=a(:size(a,1)-1,:)
  end function
end program
