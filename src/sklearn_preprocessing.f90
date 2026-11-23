module sklearn_preprocessing
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
  use sklearn_quantile_transform, only: quantile_transformer
  use sklearn_power_transform, only: power_transformer
  use sklearn_discretization, only: kbins_discretizer
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
  implicit none
  private
  public :: function_transformer
  public :: iterative_imputer,imputer_triplet
  public :: kernel_centerer
  public :: ordinal_encoder,ordinal_category
  public :: target_encoder,target_category
  public :: label_binarizer,multi_label_binarizer,label_set
  public :: spline_transformer
  public :: knn_imputer,missing_indicator
  public :: power_transformer,kbins_discretizer,quantile_transformer
  public :: standard_scaler, minmax_scaler, maxabs_scaler, robust_scaler, simple_imputer
  public :: normalize, binarize, polynomial_features, label_encoder, one_hot_encoder
  public :: normalizer,binarizer
  type :: standard_scaler
    logical :: with_mean=.true., with_std=.true.
    real(dp), allocatable :: mean_(:), var_(:), scale_(:)
  contains
    procedure :: fit=>standard_fit
    procedure :: transform=>standard_transform
    procedure :: inverse_transform=>standard_inverse
  end type
  type :: minmax_scaler
    real(dp) :: feature_min=0, feature_max=1
    logical :: clip=.false.
    real(dp), allocatable :: data_min_(:), data_max_(:), scale_(:), min_(:)
  contains
    procedure :: fit=>minmax_fit
    procedure :: transform=>minmax_transform
    procedure :: inverse_transform=>minmax_inverse
  end type
  type :: maxabs_scaler
    real(dp), allocatable :: scale_(:)
  contains
    procedure :: fit=>maxabs_fit
    procedure :: transform=>maxabs_transform
    procedure :: inverse_transform=>maxabs_inverse
  end type
  type :: robust_scaler
    logical :: with_centering=.true., with_scaling=.true.
    real(dp) :: quantile_low=25, quantile_high=75
    real(dp), allocatable :: center_(:), scale_(:)
  contains
    procedure :: fit=>robust_fit
    procedure :: transform=>robust_transform
    procedure :: inverse_transform=>robust_inverse
  end type
  type :: simple_imputer
    character(16) :: strategy='mean'
    real(dp) :: fill_value=0
    real(dp), allocatable :: statistics_(:)
  contains
    procedure :: fit=>imputer_fit
    procedure :: transform=>imputer_transform
  end type
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
  subroutine standard_fit(self,x)
    class(standard_scaler), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: j
    call check_x(x)
    self%mean_=sum(x,dim=1)/size(x,1)
    self%var_=self%mean_*0
    do j=1,size(x,2)
      self%var_(j)=sum((x(:,j)-self%mean_(j))**2)/size(x,1)
    end do
    self%scale_=sqrt(self%var_)
    where(self%scale_<10*epsilon(1._dp)) self%scale_=1
  end subroutine
  function standard_transform(self,x) result(z)
    class(standard_scaler), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: z(size(x,1),size(x,2))
    call require(allocated(self%mean_),'standard_scaler is not fitted')
    call check_xy(transpose(x),size(self%mean_)); z=x
    if(self%with_mean) z=z-spread(self%mean_,1,size(x,1))
    if(self%with_std) z=z/spread(self%scale_,1,size(x,1))
  end function
  function standard_inverse(self,x) result(z)
    class(standard_scaler), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: z(size(x,1),size(x,2))
    call require(allocated(self%mean_),'standard_scaler is not fitted')
    call check_xy(transpose(x),size(self%mean_)); z=x
    if(self%with_std) z=z*spread(self%scale_,1,size(x,1))
    if(self%with_mean) z=z+spread(self%mean_,1,size(x,1))
  end function
  subroutine minmax_fit(self,x)
    class(minmax_scaler), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: range(size(x,2))
    call check_x(x)
    call require(self%feature_max>self%feature_min,'invalid feature range')
    self%data_min_=minval(x,dim=1); self%data_max_=maxval(x,dim=1)
    range=self%data_max_-self%data_min_
    where(range<10*epsilon(1._dp)) range=1
    self%scale_=(self%feature_max-self%feature_min)/range
    self%min_=self%feature_min-self%data_min_*self%scale_
  end subroutine
  function minmax_transform(self,x) result(z)
    class(minmax_scaler), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: z(size(x,1),size(x,2))
    call require(allocated(self%scale_),'minmax_scaler is not fitted')
    call check_xy(transpose(x),size(self%scale_))
    z=x*spread(self%scale_,1,size(x,1))+spread(self%min_,1,size(x,1))
    if(self%clip) z=max(self%feature_min,min(self%feature_max,z))
  end function
  function minmax_inverse(self,x) result(z)
    class(minmax_scaler), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: z(size(x,1),size(x,2))
    call require(allocated(self%scale_),'minmax_scaler is not fitted')
    call check_xy(transpose(x),size(self%scale_))
    z=(x-spread(self%min_,1,size(x,1)))/spread(self%scale_,1,size(x,1))
  end function
  subroutine maxabs_fit(self,x)
    class(maxabs_scaler), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    call check_x(x); self%scale_=maxval(abs(x),dim=1)
    where(self%scale_<10*epsilon(1._dp)) self%scale_=1
  end subroutine
  function maxabs_transform(self,x) result(z)
    class(maxabs_scaler), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: z(size(x,1),size(x,2))
    call require(allocated(self%scale_),'maxabs_scaler is not fitted')
    call check_xy(transpose(x),size(self%scale_))
    z=x/spread(self%scale_,1,size(x,1))
  end function
  function maxabs_inverse(self,x) result(z)
    class(maxabs_scaler), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: z(size(x,1),size(x,2))
    call require(allocated(self%scale_),'maxabs_scaler is not fitted')
    call check_xy(transpose(x),size(self%scale_))
    z=x*spread(self%scale_,1,size(x,1))
  end function
  real(dp) function percentile(x,p) result(v)
    real(dp), intent(in) :: x(:),p
    integer :: order(size(x)),lo,hi
    real(dp) :: position
    order=argsort(x); position=1+(size(x)-1)*p/100
    lo=floor(position); hi=ceiling(position)
    v=x(order(lo))+(position-lo)*(x(order(hi))-x(order(lo)))
  end function
  subroutine robust_fit(self,x)
    class(robust_scaler), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: j
    call check_x(x)
    call require(self%quantile_low>=0.and.self%quantile_high<=100.and. &
      self%quantile_low<self%quantile_high,'invalid quantile range')
    self%center_=sum(x,dim=1)*0; self%scale_=self%center_+1
    do j=1,size(x,2)
      if(self%with_centering) self%center_(j)=percentile(x(:,j),50._dp)
      if(self%with_scaling) self%scale_(j)=percentile(x(:,j),self%quantile_high)- &
        percentile(x(:,j),self%quantile_low)
    end do
    where(self%scale_<10*epsilon(1._dp)) self%scale_=1
  end subroutine
  function robust_transform(self,x) result(z)
    class(robust_scaler), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: z(size(x,1),size(x,2))
    call require(allocated(self%center_),'robust_scaler is not fitted')
    call check_xy(transpose(x),size(self%center_))
    z=(x-spread(self%center_,1,size(x,1)))/spread(self%scale_,1,size(x,1))
  end function
  function robust_inverse(self,x) result(z)
    class(robust_scaler), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: z(size(x,1),size(x,2))
    call require(allocated(self%center_),'robust_scaler is not fitted')
    call check_xy(transpose(x),size(self%center_))
    z=x*spread(self%scale_,1,size(x,1))+spread(self%center_,1,size(x,1))
  end function
  subroutine imputer_fit(self,x)
    class(simple_imputer), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: col(:)
    integer :: j,i,k,best,n
    call require(size(x,1)>0.and.size(x,2)>0,'imputer X is empty')
    self%statistics_=spread(0._dp,1,size(x,2))
    do j=1,size(x,2)
      col=pack(x(:,j),.not.ieee_is_nan(x(:,j)))
      if(trim(self%strategy)=='constant') then
        self%statistics_(j)=self%fill_value; cycle
      end if
      call require(size(col)>0,'all-missing feature requires constant imputation')
      select case(trim(self%strategy))
      case('mean')
        self%statistics_(j)=sum(col)/size(col)
      case('median')
        self%statistics_(j)=percentile(col,50._dp)
      case('most_frequent')
        best=0; self%statistics_(j)=minval(col)
        do i=1,size(col)
          n=count(col==col(i))
          if(n>best.or.(n==best.and.col(i)<self%statistics_(j))) then
            best=n; self%statistics_(j)=col(i)
          end if
        end do
      case default
        call require(.false.,'unknown imputer strategy')
      end select
    end do
  end subroutine
  function imputer_transform(self,x) result(z)
    class(simple_imputer), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: z(size(x,1),size(x,2))
    integer :: j
    call require(allocated(self%statistics_),'simple_imputer is not fitted')
    call require(size(x,2)==size(self%statistics_),'imputer feature count mismatch')
    z=x
    do j=1,size(x,2)
      where(ieee_is_nan(z(:,j))) z(:,j)=self%statistics_(j)
    end do
    call check_x(z)
  end function
  function polynomial_features(x,degree,include_bias,interaction_only) result(z)
    real(dp), intent(in) :: x(:,:)
    integer, intent(in), optional :: degree
    logical, intent(in), optional :: include_bias,interaction_only
    real(dp), allocatable :: z(:,:), work(:,:)
    integer, allocatable :: indices(:)
    integer :: d,p,n,k,total,level,j,start
    logical :: bias,interaction
    call check_x(x); d=2; if(present(degree)) d=degree
    call require(d>=0.and.d<=12,'polynomial degree must be between 0 and 12')
    bias=.true.; if(present(include_bias)) bias=include_bias
    interaction=.false.; if(present(interaction_only)) interaction=interaction_only
    n=size(x,1); p=size(x,2); total=merge(1,0,bias)
    ! Count combinations using the same recursion that emits sklearn ordering.
    allocate(indices(max(1,d))); k=0
    do level=1,d
      call count_terms(1,1,level)
    end do
    total=total+k
    allocate(z(n,total)); k=0
    if(bias) then
      k=1; z(:,k)=1
    end if
    do level=1,d
      call emit(1,1,level)
    end do
  contains
    recursive subroutine count_terms(pos,first,depth)
      integer, intent(in) :: pos,first,depth
      integer :: t
      if(pos>depth) then
        k=k+1; return
      end if
      do t=first,p
        call count_terms(pos+1,t+merge(1,0,interaction),depth)
      end do
    end subroutine
    recursive subroutine emit(pos,first,depth)
      integer, intent(in) :: pos,first,depth
      integer :: t,h
      if(pos>depth) then
        k=k+1; z(:,k)=1
        do h=1,depth
          z(:,k)=z(:,k)*x(:,indices(h))
        end do
        return
      end if
      do t=first,p
        indices(pos)=t
        call emit(pos+1,t+merge(1,0,interaction),depth)
      end do
    end subroutine
  end function
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
