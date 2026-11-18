module sklearn_kernel_ridge
  use sklearn_common
  use sklearn_metrics, only: linear_kernel,rbf_kernel,polynomial_kernel,sigmoid_kernel,r2_score
  implicit none
  private
  public :: kernel_ridge
  type :: kernel_ridge
    real(dp) :: alpha=1,gamma=-1,coef0=1
    integer :: degree=3
    character(16) :: kernel='linear'
    real(dp), allocatable :: x_fit_(:,:),dual_coef_(:)
  contains
    procedure :: fit=>kernel_ridge_fit
    procedure :: predict=>kernel_ridge_predict
    procedure :: score=>kernel_ridge_score
  end type
contains
  function kernel_matrix(self,x,y) result(k)
    class(kernel_ridge), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp), allocatable :: k(:,:)
    real(dp) :: g
    g=self%gamma; if(g<0) g=1._dp/size(x,2)
    select case(trim(self%kernel))
    case('linear'); k=linear_kernel(x,y)
    case('rbf'); k=rbf_kernel(x,y,g)
    case('polynomial','poly'); k=polynomial_kernel(x,y,self%degree,g,self%coef0)
    case('sigmoid'); k=sigmoid_kernel(x,y,g,self%coef0)
    case default; call require(.false.,'unknown kernel ridge kernel')
    end select
  end function
  subroutine kernel_ridge_fit(self,x,y)
    class(kernel_ridge), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), allocatable :: gram(:,:)
    integer :: i
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    call require(self%alpha>0,'kernel ridge alpha must be positive')
    self%x_fit_=x; gram=kernel_matrix(self,x,x)
    do i=1,size(y)
      gram(i,i)=gram(i,i)+self%alpha
    end do
    self%dual_coef_=solve_system(gram,y)
  end subroutine
  function kernel_ridge_predict(self,x) result(y)
    class(kernel_ridge), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: y(size(x,1))
    call require(allocated(self%dual_coef_),'kernel ridge is not fitted')
    y=matmul(kernel_matrix(self,x,self%x_fit_),self%dual_coef_)
  end function
  real(dp) function kernel_ridge_score(self,x,y) result(v)
    class(kernel_ridge), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    v=r2_score(y,self%predict(x))
  end function
end module
