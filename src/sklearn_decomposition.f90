module sklearn_decomposition
  use sklearn_lda, only: latent_dirichlet_allocation
  use sklearn_dictionary, only: sparse_pca,dictionary_learning,sparse_coder,sparse_encode
  use sklearn_incremental_pca, only: incremental_pca
  use sklearn_fast_ica, only: fast_ica
  use sklearn_common
  use sklearn_kernel_pca, only: kernel_pca
  use sklearn_factor_analysis, only: factor_analysis
  implicit none
  private
  public :: latent_dirichlet_allocation
  public :: sparse_pca,dictionary_learning,sparse_coder,sparse_encode
  public :: incremental_pca
  public :: fast_ica,pca, truncated_svd, nmf, kernel_pca, factor_analysis
  type :: pca
    integer :: n_components=0,n_samples_=0
    logical :: whiten=.false.
    real(dp), allocatable :: components_(:,:),mean_(:),explained_variance_(:)
    real(dp), allocatable :: explained_variance_ratio_(:),singular_values_(:)
    real(dp) :: noise_variance_=0
  contains
    procedure :: fit=>pca_fit
    procedure :: transform=>pca_transform
    procedure :: inverse_transform=>pca_inverse
    procedure :: fit_transform=>pca_fit_transform
  end type
  type :: truncated_svd
    integer :: n_components=2
    real(dp), allocatable :: components_(:,:),singular_values_(:),explained_variance_(:)
    real(dp), allocatable :: explained_variance_ratio_(:)
  contains
    procedure :: fit=>svd_fit
    procedure :: transform=>svd_transform
    procedure :: inverse_transform=>svd_inverse
    procedure :: fit_transform=>svd_fit_transform
  end type
  type :: nmf
    integer :: n_components=2,max_iter=1000,n_iter_=0,random_state=0
    real(dp) :: tol=1.e-6_dp,reconstruction_err_=0
    logical :: converged_=.false.
    real(dp), allocatable :: components_(:,:)
  contains
    procedure :: fit=>nmf_fit
    procedure :: fit_transform=>nmf_fit_transform
    procedure :: transform=>nmf_transform
    procedure :: inverse_transform=>nmf_inverse
  end type
contains
  subroutine oriented_eigen(gram,values,vectors)
    real(dp), intent(in) :: gram(:,:)
    real(dp), intent(out) :: values(:),vectors(:,:)
    integer :: j,k
    call symmetric_eigen(gram,values,vectors); values=max(values,0._dp)
    do j=1,size(values)
      k=maxloc(abs(vectors(:,j)),dim=1)
      if(vectors(k,j)<0) vectors(:,j)=-vectors(:,j)
    end do
  end subroutine
  subroutine pca_fit(self,x)
    class(pca), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: centered(size(x,1),size(x,2)),u(size(x,1),size(x,2))
    real(dp) :: values(size(x,2)),vectors(size(x,2),size(x,2)),total
    integer :: k,n,p
    call check_x(x); n=size(x,1); p=size(x,2); self%n_samples_=n
    call require(n>1,'PCA requires at least two samples')
    k=self%n_components; if(k==0) k=min(n,p)
    call require(k>0.and.k<=min(n,p),'invalid PCA n_components')
    self%mean_=sum(x,dim=1)/n; centered=x-spread(self%mean_,1,n)
    call dense_svd(centered,values,u,vectors)
    self%singular_values_=values(:k); values=values**2/(n-1); total=sum(values)
    self%components_=transpose(vectors(:,:k)); self%explained_variance_=values(:k)
    self%explained_variance_ratio_=values(:k)*0
    if(total>0) self%explained_variance_ratio_=values(:k)/total
    self%noise_variance_=0
    if(k<min(n,p)) self%noise_variance_=sum(values(k+1:min(n,p)))/(min(n,p)-k)
  end subroutine
  function pca_transform(self,x) result(z)
    class(pca), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call require(allocated(self%components_),'PCA is not fitted')
    call check_xy(transpose(x),size(self%mean_))
    z=matmul(x-spread(self%mean_,1,size(x,1)),transpose(self%components_))
    if(self%whiten) z=z/spread(max(sqrt(self%explained_variance_),epsilon(1._dp)),1,size(x,1))
  end function
  function pca_inverse(self,z) result(x)
    class(pca), intent(in) :: self
    real(dp), intent(in) :: z(:,:)
    real(dp), allocatable :: x(:,:),work(:,:)
    call require(allocated(self%components_),'PCA is not fitted')
    call check_xy(transpose(z),size(self%components_,1)); work=z
    if(self%whiten) work=work*spread(sqrt(self%explained_variance_),1,size(z,1))
    x=matmul(work,self%components_)+spread(self%mean_,1,size(z,1))
  end function
  function pca_fit_transform(self,x) result(z)
    class(pca), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x); z=self%transform(x)
  end function
  subroutine svd_fit(self,x)
    class(truncated_svd), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: u(size(x,1),size(x,2)),values(size(x,2)),vectors(size(x,2),size(x,2)),total
    real(dp), allocatable :: z(:,:)
    integer :: k,j
    call check_x(x); k=self%n_components
    call require(k>0.and.k<=min(size(x,1),size(x,2)),'invalid SVD n_components')
    call dense_svd(x,values,u,vectors)
    self%components_=transpose(vectors(:,:k)); self%singular_values_=values(:k)
    z=matmul(x,transpose(self%components_)); self%explained_variance_=values(:k)*0
    do j=1,k
      self%explained_variance_(j)=sum((z(:,j)-sum(z(:,j))/size(x,1))**2)/size(x,1)
    end do
    total=0
    do j=1,size(x,2)
      total=total+sum((x(:,j)-sum(x(:,j))/size(x,1))**2)/size(x,1)
    end do
    self%explained_variance_ratio_=self%explained_variance_*0
    if(total>0) self%explained_variance_ratio_=self%explained_variance_/total
  end subroutine
  function svd_transform(self,x) result(z)
    class(truncated_svd), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call require(allocated(self%components_),'truncated_svd is not fitted')
    call check_xy(transpose(x),size(self%components_,2)); z=matmul(x,transpose(self%components_))
  end function
  function svd_inverse(self,z) result(x)
    class(truncated_svd), intent(in) :: self
    real(dp), intent(in) :: z(:,:)
    real(dp), allocatable :: x(:,:)
    call require(allocated(self%components_),'truncated_svd is not fitted')
    call check_xy(transpose(z),size(self%components_,1)); x=matmul(z,self%components_)
  end function
  function svd_fit_transform(self,x) result(z)
    class(truncated_svd), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x); z=self%transform(x)
  end function
  function nmf_fit_transform(self,x) result(w)
    class(nmf), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: w(:,:),h(:,:),numerator(:,:),denominator(:,:)
    real(dp) :: previous,error,initial,scale
    integer :: it,n,p,k
    call check_x(x); call require(all(x>=0),'NMF requires nonnegative X')
    n=size(x,1); p=size(x,2); k=self%n_components
    call require(k>0.and.self%max_iter>0.and.self%tol>0,'invalid NMF controls')
    allocate(w(n,k),h(k,p)); call seed_rng(self%random_state)
    call random_number(w); call random_number(h); scale=sqrt(sum(x)/(n*p*k))
    w=(w+0.1_dp)*scale; h=(h+0.1_dp)*scale
    initial=sqrt(sum((x-matmul(w,h))**2)); previous=initial; self%converged_=.false.
    do it=1,self%max_iter
      numerator=matmul(x,transpose(h)); denominator=matmul(w,matmul(h,transpose(h)))
      w=w*numerator/max(denominator,1.e-15_dp)
      numerator=matmul(transpose(w),x); denominator=matmul(matmul(transpose(w),w),h)
      h=h*numerator/max(denominator,1.e-15_dp)
      error=sqrt(sum((x-matmul(w,h))**2))
      if(abs(previous-error)<=self%tol*max(initial,epsilon(1._dp))) then
        self%converged_=.true.; exit
      end if
      previous=error
    end do
    self%n_iter_=min(it,self%max_iter); self%components_=h; self%reconstruction_err_=error
  end function
  subroutine nmf_fit(self,x)
    class(nmf), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: w(:,:)
    w=self%fit_transform(x)
  end subroutine
  function nmf_transform(self,x) result(w)
    class(nmf), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: w(:,:),gram(:,:),numerator(:,:)
    real(dp) :: previous,error,initial
    integer :: it
    call require(allocated(self%components_),'NMF is not fitted')
    call check_xy(transpose(x),size(self%components_,2)); call require(all(x>=0),'NMF requires nonnegative X')
    allocate(w(size(x,1),size(self%components_,1))); w=1
    gram=matmul(self%components_,transpose(self%components_)); numerator=matmul(x,transpose(self%components_))
    initial=sqrt(sum((x-matmul(w,self%components_))**2)); previous=initial
    do it=1,self%max_iter
      w=w*numerator/max(matmul(w,gram),1.e-15_dp)
      error=sqrt(sum((x-matmul(w,self%components_))**2))
      if(abs(previous-error)<=self%tol*max(initial,epsilon(1._dp))) exit
      previous=error
    end do
  end function
  function nmf_inverse(self,w) result(x)
    class(nmf), intent(in) :: self
    real(dp), intent(in) :: w(:,:)
    real(dp), allocatable :: x(:,:)
    call require(allocated(self%components_),'NMF is not fitted')
    call check_xy(transpose(w),size(self%components_,1)); x=matmul(w,self%components_)
  end function
end module
