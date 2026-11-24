! Dense storage for Gaussian and sparse random projection matrices.
! Adapted from sklearn 1.6.1 random_projection.py; see THIRD_PARTY_NOTICES.md.
module sklearn_random_projection
  use sklearn_common
  use sklearn_ieee_math, only: ieee_quotient
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_random_binomial, only: legacy_binomial
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: gaussian_random_projection,sparse_random_projection,johnson_lindenstrauss_min_dim
  type :: gaussian_random_projection
    integer :: n_components=0,n_components_=0,random_state=0,n_features_in_=0
    real(dp) :: eps=0.1_dp
    logical :: compute_inverse_components=.false.
    real(dp), allocatable :: components_(:,:),inverse_components_(:,:)
  contains
    procedure :: fit=>gaussian_fit
    procedure :: transform=>projection_transform
    procedure :: fit_transform=>projection_fit_transform
    procedure :: inverse_transform=>projection_inverse
  end type
  type, extends(gaussian_random_projection) :: sparse_random_projection
    real(dp) :: density=-1,density_=0
  contains
    procedure :: fit=>sparse_fit
  end type
contains
  integer function johnson_lindenstrauss_min_dim(n_samples,eps) result(n)
    integer, intent(in) :: n_samples
    real(dp), intent(in) :: eps
    real(dp) :: bound
    call require(ieee_is_finite(eps),'Johnson-Lindenstrauss eps must be finite')
    call require(n_samples>0.and.eps>0.and.eps<1,'invalid Johnson-Lindenstrauss parameters')
    bound=4*log(real(n_samples,dp))/(eps**2/2-eps**3/3)
    call require(bound<real(huge(n),dp),'Johnson-Lindenstrauss dimension exceeds integer range')
    n=int(bound)
  end function
  subroutine projection_setup(self,x)
    class(gaussian_random_projection), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    call check_x(x)
    call require(self%n_components>=0,'projection n_components must be nonnegative')
    call require(ieee_is_finite(self%eps),'projection eps must be finite')
    call require(self%eps>0,'projection eps must be positive')
    call require(self%random_state>=0,'projection random_state must be nonnegative')
    self%n_features_in_=size(x,2)
    self%n_components_=self%n_components
    if(self%n_components==0) then
      self%n_components_=johnson_lindenstrauss_min_dim(size(x,1),self%eps)
      call require(self%n_components_>0.and.self%n_components_<=size(x,2), &
        'automatic projection dimension must be between 1 and n_features')
    end if
    if(allocated(self%inverse_components_)) deallocate(self%inverse_components_)
  end subroutine
  function pseudoinverse(a) result(inv)
    real(dp), intent(in) :: a(:,:)
    real(dp) :: inv(size(a,2),size(a,1)),u(size(a,1),size(a,2))
    real(dp) :: s(size(a,2)),v(size(a,2),size(a,2)),weights(size(a,2)),cutoff
    call dense_svd(a,s,u,v); cutoff=1.e-15_dp*maxval(s); weights=0
    where(s>cutoff) weights=1/max(s,tiny(1._dp))
    inv=matmul(v*spread(weights,1,size(a,2)),transpose(u))
  end function
  subroutine gaussian_fit(self,x)
    class(gaussian_random_projection), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    type(legacy_random_state) :: rng
    real(dp) :: scale
    integer :: i,j
    call projection_setup(self,x);call rng%seed(self%random_state)
    if(allocated(self%components_)) deallocate(self%components_)
    allocate(self%components_(self%n_components_,size(x,2)))
    scale=1/sqrt(real(self%n_components_,dp))
    do i=1,self%n_components_
      do j=1,size(x,2)
        self%components_(i,j)=scale*rng%normal()
      end do
    end do
    if(self%compute_inverse_components) self%inverse_components_=pseudoinverse(self%components_)
  end subroutine
  subroutine sparse_fit(self,x)
    class(sparse_random_projection), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    type(legacy_random_state) :: rng
    real(dp) :: magnitude
    integer :: i,j,p,sign
    integer, allocatable :: counts(:),indices(:,:),selected(:)
    call projection_setup(self,x);call rng%seed(self%random_state)
    call require(ieee_is_finite(self%density),'projection density must be finite')
    call require(self%density==-1.or.(self%density>0.and.self%density<=1),'projection density must be in (0,1] or -1')
    self%density_=self%density;p=size(x,2)
    if(self%density==-1) self%density_=1/sqrt(real(p,dp))
    if(allocated(self%components_)) deallocate(self%components_)
    allocate(self%components_(self%n_components_,p));self%components_=0
    if(self%density_==1) then
      magnitude=1/sqrt(real(self%n_components_,dp))
      do i=1,self%n_components_
        do j=1,p
          sign=2*legacy_binomial(1,.5_dp,rng)-1
          self%components_(i,j)=magnitude*sign
        end do
      end do
    else
      allocate(counts(self%n_components_),indices(p,self%n_components_))
      do i=1,self%n_components_
        counts(i)=legacy_binomial(p,self%density_,rng)
        selected=sample_indices(p,counts(i),rng)
        indices(:counts(i),i)=selected
      end do
      magnitude=sqrt(ieee_quotient(1._dp,self%density_))/sqrt(real(self%n_components_,dp))
      do i=1,self%n_components_
        do j=1,counts(i)
          sign=2*legacy_binomial(1,.5_dp,rng)-1
          self%components_(i,indices(j,i))=magnitude*sign
        end do
      end do
    end if
    if(self%compute_inverse_components) self%inverse_components_=pseudoinverse(self%components_)
  end subroutine
  function sample_indices(n,k,rng) result(selected)
    integer, intent(in) :: n,k
    type(legacy_random_state), intent(inout) :: rng
    integer, allocatable :: selected(:),pool(:)
    real(dp) :: ratio
    integer :: i,j,swap
    allocate(selected(k));ratio=real(k,dp)/n
    if(ratio>.01_dp.and.ratio<.99_dp) then
      pool=[(i,i=1,n)]
      do i=n,2,-1
        j=rng%integer_index(i);swap=pool(i);pool(i)=pool(j);pool(j)=swap
      end do
      selected=pool(:k)
    else if(ratio<.2_dp) then
      do i=1,k
        do
          j=rng%integer_index(n)
          if(.not.any(selected(:i-1)==j)) exit
        end do
        selected(i)=j
      end do
    else
      selected=[(i,i=1,k)]
      do i=k+1,n
        j=rng%integer_index(i)
        if(j<=k) selected(j)=i
      end do
    end if
  end function
  function projection_transform(self,x) result(z)
    class(gaussian_random_projection), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call require(allocated(self%components_),'random projection is not fitted')
    call check_xy(transpose(x),size(self%components_,2))
    z=matmul(x,transpose(self%components_))
  end function
  function projection_fit_transform(self,x) result(z)
    class(gaussian_random_projection), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x); z=self%transform(x)
  end function
  function projection_inverse(self,z) result(x)
    class(gaussian_random_projection), intent(in) :: self
    real(dp), intent(in) :: z(:,:)
    real(dp), allocatable :: x(:,:)
    call require(allocated(self%components_),'random projection is not fitted')
    call check_xy(transpose(z),size(self%components_,1))
    if(allocated(self%inverse_components_)) then
      x=matmul(z,transpose(self%inverse_components_))
    else
      x=matmul(z,transpose(pseudoinverse(self%components_)))
    end if
  end function
end module
