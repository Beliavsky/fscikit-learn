! Dense/CSR kernel PCA. Reference: sklearn/decomposition/_kernel_pca.py (BSD-3-Clause).
module sklearn_kernel_pca
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use sklearn_common
  use sklearn_metrics, only: linear_kernel,rbf_kernel,polynomial_kernel,sigmoid_kernel,cosine_similarity
  use sklearn_randomized_svd_core,only:randomized_svd
  use sklearn_legacy_rng,only:legacy_random_state
  use sklearn_sparse,only:csr_matrix
  use sklearn_kernel_dispatch,only:pairwise_kernels
  implicit none
  private
  public :: kernel_pca
  type :: kernel_pca
    integer :: n_components=0,degree=3
    integer :: iterated_power=-1,random_state=0
    character(16)::eigen_solver='dense'
    integer::n_features_in_=0,n_samples_fit_=0
    logical::sparse_fit_=.false.
    type(csr_matrix),allocatable::x_fit_csr_
    real(dp) :: gamma=-1,coef0=1,gamma_=0,k_fit_all_=0
    real(dp) :: alpha=1
    logical :: remove_zero_eig=.false.,fit_inverse_transform=.false.
    character(16) :: kernel='linear'
    real(dp), allocatable :: x_transformed_fit_(:,:),dual_coef_(:,:)
    real(dp), allocatable :: x_fit_(:,:),eigenvalues_(:),eigenvectors_(:,:),k_fit_rows_(:)
  contains
    procedure :: fit_dense=>kpca_fit,fit_csr
    generic :: fit=>fit_dense,fit_csr
    procedure :: transform_dense=>kpca_transform,transform_csr
    generic :: transform=>transform_dense,transform_csr
    procedure :: fit_transform_dense=>kpca_fit_transform,fit_transform_csr
    generic :: fit_transform=>fit_transform_dense,fit_transform_csr
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
  subroutine check_sparse(x)
    type(csr_matrix),intent(in)::x
    call x%validate()
    call require(x%n_rows>0.and.x%n_cols>0,'kernel PCA sparse input must be nonempty')
    call require(all(ieee_is_finite(x%data)),'kernel PCA sparse input must be finite')
  end subroutine
  subroutine initialize_fit(self,n,p)
    class(kernel_pca),intent(inout)::self
    integer,intent(in)::n,p
    call require(self%n_components>=0,'kernel PCA n_components must be nonnegative')
    call require(self%eigen_solver=='dense'.or.self%eigen_solver=='randomized','unsupported kernel PCA eigen_solver')
    call require(self%iterated_power>=-1,'kernel PCA iterated_power must be nonnegative or -1 for auto')
    call require(self%random_state>=0,'kernel PCA random_state must be nonnegative')
    call require(ieee_is_finite(self%alpha),'kernel PCA alpha must be finite and nonnegative')
    call require(self%alpha>=0,'kernel PCA alpha must be finite and nonnegative')
    call require(ieee_is_finite(self%gamma).and.ieee_is_finite(self%coef0),'kernel PCA controls must be finite')
    call require(.not.(self%fit_inverse_transform.and.self%kernel=='precomputed'), &
      'kernel PCA inverse fitting does not support precomputed kernels')
    if(allocated(self%dual_coef_))deallocate(self%dual_coef_)
    if(allocated(self%x_transformed_fit_))deallocate(self%x_transformed_fit_)
    self%n_features_in_=p;self%n_samples_fit_=n;self%gamma_=self%gamma
    if(self%gamma_<0)self%gamma_=1._dp/p
    call require(self%gamma_>=0.and.self%degree>=0,'invalid kernel PCA parameters')
  end subroutine
  subroutine kpca_fit(self,x)
    class(kernel_pca),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    real(dp),allocatable::k(:,:)
    call check_x(x);call initialize_fit(self,size(x,1),size(x,2))
    self%x_fit_=x;self%sparse_fit_=.false.
    if(allocated(self%x_fit_csr_))deallocate(self%x_fit_csr_)
    k=gram(self,x,x);call fit_kernel(self,k)
    if(self%fit_inverse_transform)call learn_inverse(self,x)
  end subroutine
  subroutine fit_csr(self,x)
    class(kernel_pca),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::k(:,:)
    call check_sparse(x);call initialize_fit(self,x%n_rows,x%n_cols)
    call require(self%kernel/='precomputed','kernel PCA precomputed inputs must be dense')
    call require(.not.self%fit_inverse_transform,'kernel PCA inverse fitting does not support sparse designs')
    self%x_fit_csr_=x;self%sparse_fit_=.true.
    if(allocated(self%x_fit_))deallocate(self%x_fit_)
    k=pairwise_kernels(x,x,metric=self%kernel,gamma=self%gamma_,degree=self%degree,coef0=self%coef0,filter_params=.true.)
    call fit_kernel(self,k)
  end subroutine
  subroutine fit_kernel(self,k)
    class(kernel_pca),intent(inout)::self
    real(dp),intent(inout)::k(:,:)
    real(dp),allocatable::values(:),vectors(:,:),right(:,:)
    integer,allocatable::order(:)
    type(legacy_random_state)::rng
    integer::n,c,j,p
    real(dp)::largest,negative_limit,orientation
    n=self%n_samples_fit_;call check_x(k)
    call require(all(shape(k)==[n,n]),'precomputed kernel must be square')
    call require(maxval(abs(k-transpose(k)))<=1.e-10_dp*max(1._dp,maxval(abs(k))), &
      'kernel PCA requires a symmetric training kernel')
    self%k_fit_rows_=sum(k,dim=1)/n; self%k_fit_all_=sum(self%k_fit_rows_)/n
    k=k-spread(self%k_fit_rows_,1,n)-spread(sum(k,dim=2)/n,2,n)+self%k_fit_all_
    c=self%n_components; if(c==0) c=n
    c=min(c,n)
    if(self%eigen_solver=='randomized')then
      call rng%seed(self%random_state)
      if(self%iterated_power==-1)then
        call randomized_svd(k,c,rng,values,vectors,right,flip_sign=.false.)
      else
        call randomized_svd(k,c,rng,values,vectors,right,n_iter=self%iterated_power,flip_sign=.false.)
      end if
      ! sklearn.utils.extmath._randomized_eigsh: sign(diag(Vt @ U)).
      do j=1,c
        orientation=dot_product(right(j,:),vectors(:,j))
        if(orientation==0)then
          values(j)=0
        else if(orientation<0)then
          values(j)=-values(j)
        end if
      end do
    else
      allocate(values(n),vectors(n,n));call symmetric_eigen(k,values,vectors)
    end if
    largest=maxval(values(:c))
    call require(largest>=0,'kernel PCA selected only negative eigenvalues')
    negative_limit=max(1.e-10_dp,largest*1.e-5_dp)
    call require(minval(values(:c))>=-negative_limit,'kernel PCA selected significant negative eigenvalues')
    where(values<max(largest*1.e-12_dp,0._dp)) values=0
    order=argsort(values(:c));order=order(c:1:-1)
    values(:c)=values(order);vectors(:,:c)=vectors(:,order)
    if(self%remove_zero_eig.or.self%n_components==0) c=count(values(:c)>0)
    self%eigenvalues_=values(:c); self%eigenvectors_=vectors(:,:c)
    do j=1,c
      p=maxloc(abs(self%eigenvectors_(:,j)),dim=1)
      if(self%eigenvectors_(p,j)<0) self%eigenvectors_(:,j)=-self%eigenvectors_(:,j)
    end do
  end subroutine
  subroutine learn_inverse(self,x)
    class(kernel_pca),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    real(dp),allocatable::k(:,:),l(:,:),upper(:,:)
    integer::n,j
    n=size(x,1)
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
    call check_x(x); n=self%n_samples_fit_
    if(self%kernel=='precomputed') then
      call require(size(x,2)==n,'kernel PCA precomputed query width mismatch')
    else
      call require(size(x,2)==self%n_features_in_,'kernel PCA feature count mismatch')
    end if
    if(self%sparse_fit_)then
      k=pairwise_kernels(x,self%x_fit_csr_,metric=self%kernel,gamma=self%gamma_,degree=self%degree, &
        coef0=self%coef0,filter_params=.true.)
    else
      k=gram(self,x,self%x_fit_)
    end if
    z=project_kernel(self,k)
  end function
  function transform_csr(self,x)result(z)
    class(kernel_pca),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::z(:,:),k(:,:)
    integer::width
    call require(allocated(self%eigenvalues_),'kernel PCA is not fitted');call check_sparse(x)
    call require(self%kernel/='precomputed','kernel PCA precomputed queries must be dense')
    width=self%n_features_in_;if(self%kernel=='precomputed')width=self%n_samples_fit_
    call require(x%n_cols==width,'kernel PCA sparse query width mismatch')
    if(self%sparse_fit_)then
      k=pairwise_kernels(x,self%x_fit_csr_,metric=self%kernel,gamma=self%gamma_,degree=self%degree, &
        coef0=self%coef0,filter_params=.true.)
    else
      k=pairwise_kernels(x,self%x_fit_,metric=self%kernel,gamma=self%gamma_,degree=self%degree, &
        coef0=self%coef0,filter_params=.true.)
    end if
    z=project_kernel(self,k)
  end function
  function project_kernel(self,raw)result(z)
    class(kernel_pca),intent(in)::self
    real(dp),intent(in)::raw(:,:)
    real(dp),allocatable::z(:,:),k(:,:),scaled(:,:)
    integer::n,j
    n=self%n_samples_fit_;k=raw
    k=k-spread(self%k_fit_rows_,1,size(k,1))-spread(sum(k,dim=2)/n,2,n)+self%k_fit_all_
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
  function fit_transform_csr(self,x)result(z)
    class(kernel_pca),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::z(:,:)
    call self%fit(x)
    z=self%eigenvectors_*spread(sqrt(self%eigenvalues_),1,x%n_rows)
  end function
  function kpca_fit_transform(self,x) result(z)
    class(kernel_pca), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x)
    z=self%eigenvectors_*spread(sqrt(self%eigenvalues_),1,size(x,1))
  end function
end module
