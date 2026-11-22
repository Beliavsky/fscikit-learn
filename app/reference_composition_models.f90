module reference_composition_models
  use sklearn_common, only: dp
  use sklearn_feature_composition
  use sklearn_feature_extraction, only: feature_string
  use sklearn_preprocessing
  use sklearn_decomposition, only: pca,truncated_svd
  use sklearn_feature_selection, only: variance_threshold
  implicit none
  integer :: selector_calls=0
contains
  function format_name(branch,feature) result(name)
    character(*), intent(in) :: branch,feature
    character(:), allocatable :: name
    name=branch//'::'//feature
  end function
  subroutine selector(x,names,columns)
    real(dp), intent(in) :: x(:,:)
    type(feature_string), intent(in) :: names(:)
    integer, allocatable, intent(out) :: columns(:)
    selector_calls=selector_calls+1;columns=[3,1]
  end subroutine
  function affine(x) result(z)
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    z=2*x+3
  end function
  function expand(x) result(z)
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    z=reshape([x,x*x],[size(x,1),2*size(x,2)])
  end function
  function expanded_names(model,input) result(names)
    class(*), intent(in) :: model
    type(feature_string), intent(in) :: input(:)
    type(feature_string), allocatable :: names(:)
    integer :: i,p
    p=size(input);allocate(names(2*p))
    do i=1,p
      names(i)%text=input(i)%text//'_original';names(p+i)%text=input(i)%text//'_square'
    end do
  end function
  subroutine make_model(kind,config,model)
    integer, intent(in) :: kind,config
    class(*), allocatable, intent(out) :: model
    type(function_transformer) :: f
    type(standard_scaler) :: standard
    type(minmax_scaler) :: minmax
    type(maxabs_scaler) :: maxabs
    type(robust_scaler) :: robust
    type(simple_imputer) :: imputer
    type(one_hot_encoder) :: onehot
    type(ordinal_encoder) :: ordinal
    type(knn_imputer) :: knn
    type(missing_indicator) :: indicator
    type(power_transformer) :: power
    type(quantile_transformer) :: quantile
    type(kbins_discretizer) :: bins
    type(spline_transformer) :: spline
    type(pca) :: principal
    type(truncated_svd) :: svd
    type(variance_threshold) :: variance
    type(feature_union) :: union
    type(column_transformer) :: column
    select case(kind)
    case(0)
      f%func=>affine;f%feature_names_out='one-to-one';allocate(model,source=f)
    case(1);standard%with_mean=mod(config,2)==0;allocate(model,source=standard)
    case(2);minmax%feature_min=-2;minmax%feature_max=3;allocate(model,source=minmax)
    case(3);allocate(model,source=maxabs)
    case(4);robust%with_centering=mod(config,2)==0;allocate(model,source=robust)
    case(5);imputer%strategy='median';allocate(model,source=imputer)
    case(6);onehot%handle_unknown_ignore=.true.;allocate(model,source=onehot)
    case(7)
      ordinal%handle_unknown='use_encoded_value';ordinal%unknown_value=-1;allocate(model,source=ordinal)
    case(8);knn%n_neighbors=3;knn%keep_empty_features=.true.;knn%add_indicator=.true.;allocate(model,source=knn)
    case(9);indicator%error_on_new=.false.;allocate(model,source=indicator)
    case(10);power%standardize=mod(config,2)==0;allocate(model,source=power)
    case(11);quantile%n_quantiles=7;quantile%subsample=0;allocate(model,source=quantile)
    case(12)
      bins%n_bins=3;bins%strategy='uniform'
      if(mod(config,2)==0) bins%encode='ordinal'
      allocate(model,source=bins)
    case(13);spline%n_knots=3;spline%degree=2;spline%include_bias=mod(config,2)==0;allocate(model,source=spline)
    case(14);principal%n_components=1;principal%whiten=mod(config,2)==0;allocate(model,source=principal)
    case(15);svd%n_components=1;allocate(model,source=svd)
    case(16);allocate(model,source=variance)
    case(17)
      allocate(union%transformer_list(2))
      union%transformer_list(1)%name='center';allocate(union%transformer_list(1)%model,source=standard)
      union%transformer_list(2)%name='raw';union%transformer_list(2)%action='passthrough'
      allocate(model,source=union)
    case(18)
      allocate(column%transformers(1));column%transformers(1)%name='first';column%transformers(1)%columns=[1]
      allocate(column%transformers(1)%model,source=standard);column%remainder%action='passthrough'
      allocate(model,source=column)
    case(19)
      f%func=>expand;f%names_func=>expanded_names;allocate(model,source=f)
    end select
  end subroutine
end module
