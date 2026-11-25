module sklearn_preprocessing
  use sklearn_label_binarize,only:label_binarize,label_binarize_csr
  use sklearn_one_hot_encoding,only:one_hot_encoder,onehot_category,onehot_inverse_result,onehot_name_combiner,onehot_name_value
  use sklearn_simple_impute, only: simple_imputer,imputer_statistic
  use sklearn_standard_scaling, only: standard_scaler,scale
  use sklearn_robust_scaling, only: robust_scaler,robust_scale
  use sklearn_minmax_scaling, only: minmax_scaler,minmax_scale
  use sklearn_maxabs_scaling, only: maxabs_scaler,maxabs_scale
  use sklearn_polynomial, only: polynomial_features
  use sklearn_stateless_preprocessing, only: normalizer,binarizer,normalize,binarize,add_dummy_feature
  use sklearn_function_transformer, only: function_transformer
  use sklearn_iterative_impute, only: iterative_imputer,imputer_triplet
  use sklearn_kernel_centering, only: kernel_centerer
  use sklearn_ordinal_encoding, only: ordinal_encoder,ordinal_category
  use sklearn_target_encoding, only: target_encoder,target_category
  use sklearn_label_binarization, only: label_binarizer,multi_label_binarizer,label_set
  use sklearn_spline, only: spline_transformer
  use sklearn_knn_impute, only: knn_imputer,missing_indicator
  use sklearn_common
  use sklearn_quantile_transform, only: quantile_transformer,quantile_transform
  use sklearn_power_transform, only: power_transformer,power_transform
  use sklearn_discretization, only: kbins_discretizer
  implicit none
  private
  public::label_binarize,label_binarize_csr
  public :: onehot_category,onehot_inverse_result,onehot_name_combiner,onehot_name_value
  public :: imputer_statistic
  public :: function_transformer
  public :: iterative_imputer,imputer_triplet
  public :: kernel_centerer
  public :: ordinal_encoder,ordinal_category
  public :: target_encoder,target_category
  public :: label_binarizer,multi_label_binarizer,label_set
  public :: spline_transformer
  public :: knn_imputer,missing_indicator
  public :: power_transform
  public :: quantile_transform
  public :: power_transformer,kbins_discretizer,quantile_transformer
  public :: robust_scale
  public :: minmax_scale
  public :: maxabs_scale
  public :: scale
  public :: standard_scaler, minmax_scaler, maxabs_scaler, robust_scaler, simple_imputer
  public :: normalize, binarize, polynomial_features, label_encoder, one_hot_encoder
  public :: add_dummy_feature
  public :: normalizer,binarizer
  type :: label_encoder
    integer, allocatable :: classes_(:)
  contains
    procedure :: fit=>label_fit
    procedure :: transform=>label_transform
    procedure :: inverse_transform=>label_inverse
  end type
contains
  subroutine label_fit(self,y)
    class(label_encoder), intent(inout) :: self
    integer, intent(in) :: y(:)
    self%classes_=unique_labels(y)
  end subroutine
  function label_transform(self,y) result(z)
    class(label_encoder), intent(in) :: self
    integer, intent(in) :: y(:)
    integer :: z(size(y)),i
    call require(allocated(self%classes_),'label_encoder is not fitted')
    do i=1,size(y)
      z(i)=label_index(self%classes_,y(i))-1
      call require(z(i)>=0,'unknown label')
    end do
  end function
  function label_inverse(self,y) result(z)
    class(label_encoder), intent(in) :: self
    integer, intent(in) :: y(:)
    integer :: z(size(y))
    call require(allocated(self%classes_),'label_encoder is not fitted')
    call require(all(y>=0).and.all(y<size(self%classes_)),'invalid encoded label')
    z=self%classes_(y+1)
  end function
end module
