program feature_composition
  use sklearn, only: dp,column_transformer,feature_union,standard_scaler,one_hot_encoder,feature_string
  implicit none
  type(column_transformer) :: columns
  type(feature_union) :: features
  type(standard_scaler) :: scaler
  type(one_hot_encoder) :: categories
  type(feature_string), allocatable :: names(:)
  real(dp) :: x(6,3)
  real(dp), allocatable :: z(:,:),combined(:,:)
  integer :: i
  do i=1,6
    x(i,:)=[real(i,dp),real(mod(i,2),dp),real(i*i,dp)]
  end do
  allocate(columns%transformers(2))
  columns%transformers(1)%name='numeric';columns%transformers(1)%columns=[1,3]
  allocate(columns%transformers(1)%model,source=scaler)
  columns%transformers(2)%name='category';columns%transformers(2)%columns=[2]
  categories%handle_unknown_ignore=.true.;allocate(columns%transformers(2)%model,source=categories)
  z=columns%fit_transform(x);names=columns%get_feature_names_out()
  print '(a,2i4)', 'Column transform shape:',shape(z)
  print '(a,a)', 'First output feature: ',names(1)%text
  allocate(features%transformer_list(2))
  features%transformer_list(1)%name='processed';allocate(features%transformer_list(1)%model,source=columns)
  features%transformer_list(2)%name='raw';features%transformer_list(2)%action='passthrough'
  features%transformer_list(2)%weight=.5_dp
  combined=features%fit_transform(x)
  print '(a,2i4)', 'Feature union shape:',shape(combined)
end program
