! Dense kernel PCA. Reference: sklearn/decomposition/_kernel_pca.py (BSD-3-Clause).
module sklearn_kernel_pca
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use sklearn_common
  use sklearn_metrics, only: linear_kernel,rbf_kernel,polynomial_kernel,sigmoid_kernel,cosine_similarity
  implicit none
  private
  public :: kernel_pca
  type :: kernel_pca
    integer :: n_components=0,degree=3
    real(dp) :: gamma=-1,coef0=1,gamma_=0,k_fit_all_=0
    real(dp) :: alpha=1
    logical :: remove_zero_eig=.false.,fit_inverse_transform=.false.
    character(16) :: kernel='linear'
    real(dp), allocatable :: x_transformed_fit_(:,:),dual_coef_(:,:)
    real(dp), allocatable :: x_fit_(:,:),eigenvalues_(:),eigenvectors_(:,:),k_fit_rows_(:)
  contains
    procedure :: fit=>kpca_fit
    procedure :: transform=>kpca_transform
    procedure :: fit_transform=>kpca_fit_transform
    procedure :: inverse_transform=>kpca_inverse_transform
  end type
contains
  function gram(self,x,y) result(k)
    class(kernel_pca), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp), allocatable :: k(:,:)
    select case(trim(self%kernel))
    case('linear'); k=linear_kernel(x,y)
    case('rbf'); k=rbf_kernel(x,y,self%gamma_)
    case('poly','polynomial'); k=polynomial_kernel(x,y,self%degree,self%gamma_,self%coef0)
    case('sigmoid'); k=sigmoid_kernel(x,y,self%gamma_,self%coef0)
    case('cosine'); k=cosine_similarity(x,y)
    case('precomputed'); k=x
    case default; call require(.false.,'unknown kernel PCA kernel')
    end select
  end function
  subroutine kpca_fit(self,x)
    class(kernel_pca), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: k(:,:),values(:),vectors(:,:),l(:,:),upper(:,:)
    integer :: n,c,j,p
    real(dp) :: largest,negative_limit
    call check_x(x); n=size(x,1)
    call require(self%n_components>=0,'kernel PCA n_components must be nonnegative')
    call require(ieee_is_finite(self%alpha),'kernel PCA alpha must be finite and nonnegative')
    call require(self%alpha>=0,'kernel PCA alpha must be finite and nonnegative')
    call require(ieee_is_finite(self%gamma).and.ieee_is_finite(self%coef0),'kernel PCA controls must be finite')
    call require(.not.(self%fit_inverse_transform.and.self%kernel=='precomputed'), &
      'kernel PCA inverse fitting does not support precomputed kernels')
    if(allocated(self%dual_coef_)) deallocate(self%dual_coef_)
    if(allocated(self%x_transformed_fit_)) deallocate(self%x_transformed_fit_)
    self%x_fit_=x; self%gamma_=self%gamma
    if(self%gamma_<0) self%gamma_=1._dp/size(x,2)
    call require(self%gamma_>=0.and.self%degree>=0,'invalid kernel PCA parameters')
    k=gram(self,x,x)
    call require(all(shape(k)==[n,n]),'precomputed kernel must be square')
    call require(maxval(abs(k-transpose(k)))<=1.e-10_dp*max(1._dp,maxval(abs(k))), &
      'kernel PCA requires a symmetric training kernel')
    self%k_fit_rows_=sum(k,dim=1)/n; self%k_fit_all_=sum(self%k_fit_rows_)/n
    k=k-spread(self%k_fit_rows_,1,n)-spread(sum(k,dim=2)/n,2,n)+self%k_fit_all_
    allocate(values(n),vectors(n,n)); call symmetric_eigen(k,values,vectors)
    c=self%n_components; if(c==0) c=n
    c=min(c,n); largest=max(0._dp,values(1))
    negative_limit=max(1.e-10_dp,largest*1.e-5_dp)
    call require(minval(values(:c))>=-negative_limit,'kernel PCA selected significant negative eigenvalues')
    where(values<max(largest*1.e-12_dp,0._dp)) values=0
    if(self%remove_zero_eig.or.self%n_components==0) c=count(values(:c)>0)
    self%eigenvalues_=values(:c); self%eigenvectors_=vectors(:,:c)
    do j=1,c
      p=maxloc(abs(self%eigenvectors_(:,j)),dim=1)
      if(self%eigenvectors_(p,j)<0) self%eigenvectors_(:,j)=-self%eigenvectors_(:,j)
    end do
    if(self%fit_inverse_transform) then
      self%x_transformed_fit_=self%eigenvectors_*spread(sqrt(self%eigenvalues_),1,n)
      call check_x(self%x_transformed_fit_)
      k=gram(self,self%x_transformed_fit_,self%x_transformed_fit_)
      call check_x(k)
      do j=1,n
        k(j,j)=k(j,j)+self%alpha
      end do
      l=cholesky_factor(k);upper=transpose(l)
      allocate(self%dual_coef_(n,size(x,2)))
      do j=1,size(x,2)
        self%dual_coef_(:,j)=solve_upper(upper,solve_lower(l,x(:,j)))
      end do
      call check_x(self%dual_coef_)
    end if
  end subroutine
  function kpca_inverse_transform(self,z) result(x)
    class(kernel_pca), intent(in) :: self
    real(dp), intent(in) :: z(:,:)
    real(dp), allocatable :: x(:,:),k(:,:)
    call require(self%fit_inverse_transform,'kernel PCA inverse fitting was not enabled')
    call require(allocated(self%dual_coef_),'kernel PCA inverse model is not fitted')
    call check_x(z)
    call require(size(z,2)==size(self%x_transformed_fit_,2),'kernel PCA inverse component count mismatch')
    k=gram(self,z,self%x_transformed_fit_)
    call check_x(k)
    x=matmul(k,self%dual_coef_)
  end function
  function kpca_transform(self,x) result(z)
    class(kernel_pca), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:),k(:,:),scaled(:,:)
    integer :: n,j
    call require(allocated(self%eigenvalues_),'kernel PCA is not fitted')
    call check_x(x); n=size(self%x_fit_,1)
    if(self%kernel=='precomputed') then
      call require(size(x,2)==n,'kernel PCA precomputed query width mismatch')
    else
      call require(size(x,2)==size(self%x_fit_,2),'kernel PCA feature count mismatch')
    end if
    k=gram(self,x,self%x_fit_)
    k=k-spread(self%k_fit_rows_,1,size(x,1))-spread(sum(k,dim=2)/n,2,n)+self%k_fit_all_
    scaled=self%eigenvectors_
    do j=1,size(self%eigenvalues_)
      if(self%eigenvalues_(j)>0) then
        scaled(:,j)=scaled(:,j)/sqrt(self%eigenvalues_(j))
      else
        scaled(:,j)=0
      end if
    end do
    z=matmul(k,scaled)
  end function
  function kpca_fit_transform(self,x) result(z)
    class(kernel_pca), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x)
    z=self%eigenvectors_*spread(sqrt(self%eigenvalues_),1,size(x,1))
  end function
end module
