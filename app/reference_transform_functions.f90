! Validation-only callbacks shared by reference drivers.
module reference_transform_functions
  use sklearn_common, only: dp
  use sklearn_feature_extraction, only: feature_string
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  implicit none
  integer :: callback_mode=0
contains
  function forward(x) result(y)
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:,:)
    integer :: j
    select case(callback_mode)
    case(1,6);y=2*x+3
    case(2);y=log(x)
    case(3);y=x*x
    case(4)
      allocate(y(size(x,1),2*size(x,2)));y(:,:size(x,2))=x;y(:,size(x,2)+1:)=x
    case(5)
      y=x
      do j=2,size(x,2)
        y(:,j)=x(:,j)+.25_dp*x(:,j-1)
      end do
    case(8);allocate(y(size(x,1),size(x,2)));y=ieee_value(0._dp,ieee_quiet_nan)
    case default;y=x
    end select
  end function
  function backward(x) result(y)
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:,:)
    integer :: i,j
    select case(callback_mode)
    case(1);y=(x-3)/2
    case(2);y=exp(x)
    case(3)
      allocate(y(size(x,1),size(x,2)))
      do j=1,size(x,2)
        do i=1,size(x,1)
          if(x(i,j)<0) then
            y(i,j)=ieee_value(0._dp,ieee_quiet_nan)
          else
            y(i,j)=sqrt(x(i,j))
          end if
        end do
      end do
    case(4);y=x(:,:size(x,2)/2)
    case(5)
      y=x
      do j=2,size(x,2)
        y(:,j)=x(:,j)-.25_dp*y(:,j-1)
      end do
    case(6);y=2*x+3
    case(7);y=x+5.e-9_dp
    case(10)
      y=x
      where(x==7.7_dp) y=x+.01_dp
    case default;y=x
    end select
  end function
  function names_prefix(transformer,input_features) result(names)
    class(*), intent(in) :: transformer
    type(feature_string), intent(in) :: input_features(:)
    type(feature_string), allocatable :: names(:)
    integer :: j
    allocate(names(size(input_features)))
    do j=1,size(names)
      names(j)%text='z_'//input_features(j)%text
    end do
    if(size(names)==0) names=[feature_string('z_unknown')]
  end function
end module
