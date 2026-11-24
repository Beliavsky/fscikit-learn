module sklearn_preprocessing
  use sklearn_simple_impute, only: simple_imputer,imputer_statistic
  use sklearn_standard_scaling, only: standard_scaler,scale
  use sklearn_robust_scaling, only: robust_scaler,robust_scale
  use sklearn_minmax_scaling, only: minmax_scaler,minmax_scale
  use sklearn_maxabs_scaling, only: maxabs_scaler,maxabs_scale
  use sklearn_polynomial, only: polynomial_features
  use sklearn_stateless_preprocessing, only: normalizer,binarizer,normalize,binarize
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
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
  implicit none
  private
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
  public :: normalizer,binarizer
  type :: label_encoder
    integer, allocatable :: classes_(:)
  contains
    procedure :: fit=>label_fit
    procedure :: transform=>label_transform
    procedure :: inverse_transform=>label_inverse
  end type
  type :: category_column
    real(dp), allocatable :: values(:)
  end type
  type :: one_hot_encoder
    logical :: handle_unknown_ignore=.false.
    type(category_column), allocatable :: categories_(:)
    integer :: n_features_out_=0
  contains
    procedure :: fit=>onehot_fit
    procedure :: transform=>onehot_transform
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
  subroutine onehot_fit(self,x)
    class(one_hot_encoder), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: j,i,k,order(size(x,1))
    real(dp) :: tmp(size(x,1))
    call check_x(x)
    if(allocated(self%categories_)) deallocate(self%categories_)
    allocate(self%categories_(size(x,2))); self%n_features_out_=0
    do j=1,size(x,2)
      order=argsort(x(:,j)); tmp=x(order,j); k=1
      do i=2,size(x,1)
        if(tmp(i)/=tmp(k)) then
          k=k+1; tmp(k)=tmp(i)
        end if
      end do
      self%categories_(j)%values=tmp(:k); self%n_features_out_=self%n_features_out_+k
    end do
  end subroutine
  function onehot_transform(self,x) result(z)
    class(one_hot_encoder), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    integer :: i,j,k,offset
    logical :: found
    call require(allocated(self%categories_),'one_hot_encoder is not fitted')
    call check_xy(transpose(x),size(self%categories_))
    allocate(z(size(x,1),self%n_features_out_)); z=0; offset=0
    do j=1,size(x,2)
      do i=1,size(x,1)
        found=.false.
        do k=1,size(self%categories_(j)%values)
          if(x(i,j)==self%categories_(j)%values(k)) then
            z(i,offset+k)=1; found=.true.; exit
          end if
        end do
        call require(found.or.self%handle_unknown_ignore,'unknown category')
      end do
      offset=offset+size(self%categories_(j)%values)
    end do
  end function
end module
