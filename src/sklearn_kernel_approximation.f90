! Dense kernel feature maps following sklearn/kernel_approximation.py.
! See THIRD_PARTY_NOTICES.md for upstream BSD-3-Clause attribution.
module sklearn_kernel_approximation
  use sklearn_polynomial_count_sketch, only: polynomial_count_sketch
  use sklearn_common
  use sklearn_metrics, only: linear_kernel,rbf_kernel,polynomial_kernel,cosine_similarity
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: polynomial_count_sketch
  public :: rbf_sampler,nystroem,additive_chi2_sampler,skewed_chi2_sampler
  ! Chi2 map contracts follow sklearn 1.6.1 kernel_approximation.py.
  type :: additive_chi2_sampler
    integer :: sample_steps=2,n_features_in_=0
    real(dp) :: sample_interval=-1
  contains
    procedure :: fit=>additive_fit
    procedure :: transform=>additive_transform
    procedure :: fit_transform=>additive_fit_transform
  end type
  type :: skewed_chi2_sampler
    integer :: n_components=100,random_state=0,n_features_in_=0
    real(dp) :: skewedness=1
    real(dp), allocatable :: random_weights_(:,:),random_offset_(:)
  contains
    procedure :: fit=>skewed_fit
    procedure :: transform=>skewed_transform
    procedure :: fit_transform=>skewed_fit_transform
  end type
  type :: rbf_sampler
    integer :: n_components=100,random_state=0
    real(dp) :: gamma=1,gamma_=1
    logical :: gamma_scale=.false.
    real(dp), allocatable :: random_weights_(:,:),random_offset_(:)
  contains
    procedure :: fit=>rbfs_fit
    procedure :: transform=>rbfs_transform
    procedure :: fit_transform=>rbfs_fit_transform
  end type
  type :: nystroem
    integer :: n_components=100,random_state=0,degree=3
    character(16) :: kernel='rbf'
    real(dp) :: gamma=-1,coef0=1,gamma_=0
    real(dp), allocatable :: components_(:,:),normalization_(:,:)
    integer, allocatable :: component_indices_(:)
    character(16), private :: kernel_fit_='rbf'
    integer, private :: degree_fit_=3
    real(dp), private :: coef0_fit_=1
  contains
    procedure :: fit=>ny_fit
    procedure :: transform=>ny_transform
    procedure :: fit_transform=>ny_fit_transform
  end type
contains
  subroutine rbfs_fit(self,x)
    class(rbf_sampler), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: variance,mean
    call check_x(x)
    call require(self%n_components>0,'RBF sampler n_components must be positive')
    call require(ieee_is_finite(self%gamma).and.self%gamma>=0,'invalid RBF sampler gamma')
    self%gamma_=self%gamma
    if(self%gamma_scale) then
      mean=sum(x)/size(x); variance=sum((x-mean)**2)/size(x); self%gamma_=1
      if(variance>0) self%gamma_=1/(size(x,2)*variance)
    end if
    call seed_rng(self%random_state)
    self%random_weights_=sqrt(2*self%gamma_)*normal_random(size(x,2),self%n_components)
    if(allocated(self%random_offset_)) deallocate(self%random_offset_)
    allocate(self%random_offset_(self%n_components)); call random_number(self%random_offset_)
    self%random_offset_=self%random_offset_*2*acos(-1._dp)
  end subroutine
  function rbfs_transform(self,x) result(z)
    class(rbf_sampler), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call require(allocated(self%random_weights_),'RBF sampler is not fitted')
    call check_xy(transpose(x),size(self%random_weights_,1))
    z=cos(matmul(x,self%random_weights_)+spread(self%random_offset_,1,size(x,1)))* &
      sqrt(2._dp/size(self%random_weights_,2))
  end function
  function rbfs_fit_transform(self,x) result(z)
    class(rbf_sampler), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x); z=self%transform(x)
  end function
  function kernel_matrix(self,x,y) result(k)
    class(nystroem), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp), allocatable :: k(:,:)
    select case(self%kernel_fit_)
    case('rbf'); k=rbf_kernel(x,y,self%gamma_)
    case('linear'); k=linear_kernel(x,y)
    case('poly','polynomial'); k=polynomial_kernel(x,y,self%degree_fit_,self%gamma_,self%coef0_fit_)
    case('cosine'); k=cosine_similarity(x,y)
    end select
  end function
  subroutine ny_fit(self,x)
    class(nystroem), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: gram(:,:),values(:),vectors(:,:)
    integer :: n,k,i,j,tmp
    integer, allocatable :: order(:)
    real(dp) :: draw
    call check_x(x); n=size(x,1)
    call require(self%n_components>0,'Nystroem n_components must be positive')
    call require(self%kernel=='rbf'.or.self%kernel=='linear'.or.self%kernel=='poly'.or. &
      self%kernel=='polynomial'.or.self%kernel=='cosine','unsupported Nystroem kernel')
    call require(ieee_is_finite(self%gamma).and.(self%gamma>=0.or.self%gamma==-1), &
      'invalid Nystroem gamma')
    call require(self%degree>=0.and.ieee_is_finite(self%coef0).and.self%coef0>=0, &
      'invalid Nystroem polynomial parameters')
    self%kernel_fit_=self%kernel; self%degree_fit_=self%degree; self%coef0_fit_=self%coef0
    self%gamma_=self%gamma
    if(self%gamma_==-1) self%gamma_=1._dp/size(x,2)
    k=min(n,self%n_components); allocate(order(n)); order=[(i,i=1,n)]
    call seed_rng(self%random_state)
    do i=n,2,-1
      call random_number(draw); j=1+int(draw*i); tmp=order(i); order(i)=order(j); order(j)=tmp
    end do
    self%component_indices_=order(:k); self%components_=x(order(:k),:)
    gram=kernel_matrix(self,self%components_,self%components_)
    call require(all(ieee_is_finite(gram)),'Nystroem nonfinite kernel matrix')
    allocate(values(k),vectors(k,k)); call symmetric_eigen(gram,values,vectors)
    call require(minval(values)>=-1.e-10_dp*max(1._dp,maxval(abs(values))), &
      'Nystroem basis kernel must be positive semidefinite')
    values=1/sqrt(max(values,1.e-12_dp))
    self%normalization_=matmul(vectors*spread(values,1,k),transpose(vectors))
  end subroutine
  function ny_transform(self,x) result(z)
    class(nystroem), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call require(allocated(self%components_),'Nystroem is not fitted')
    call check_xy(transpose(x),size(self%components_,2))
    z=matmul(kernel_matrix(self,x,self%components_),transpose(self%normalization_))
  end function
  function ny_fit_transform(self,x) result(z)
    class(nystroem), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x); z=self%transform(x)
  end function
  real(dp) function additive_interval(self) result(h)
    class(additive_chi2_sampler), intent(in) :: self
    call require(self%sample_steps>=1,'AdditiveChi2 sample_steps must be positive')
    call require(ieee_is_finite(self%sample_interval),'nonfinite AdditiveChi2 sample_interval')
    call require(self%sample_interval==-1.or.self%sample_interval>=0,'invalid AdditiveChi2 sample_interval')
    h=self%sample_interval
    if(h==-1) then
      select case(self%sample_steps)
      case(1); h=.8_dp
      case(2); h=.5_dp
      case(3); h=.4_dp
      case default; call require(.false.,'AdditiveChi2 needs an explicit interval for these steps')
      end select
    end if
  end function

  subroutine additive_fit(self,x)
    class(additive_chi2_sampler), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: interval
    interval=additive_interval(self)
    call check_x(x); call require(all(x>=0),'AdditiveChi2 requires nonnegative input')
    self%n_features_in_=size(x,2)
  end subroutine

  function additive_transform(self,x) result(z)
    use, intrinsic :: iso_fortran_env, only: int64
    class(additive_chi2_sampler), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    real(dp) :: h,a,amplitude,phase,pi,log_amplitude
    integer :: n,p,i,j,k,offset,width
    integer(int64) :: wide
    h=additive_interval(self); call check_x(x)
    call require(all(x>=0),'AdditiveChi2 requires nonnegative input')
    n=size(x,1); p=size(x,2)
    if(self%n_features_in_>0) call require(p==self%n_features_in_,'AdditiveChi2 feature count mismatch')
    wide=(2_int64*self%sample_steps-1)*int(p,int64)
    call require(wide<=huge(1),'AdditiveChi2 output width overflow')
    width=int(wide); allocate(z(n,width)); z=0
    if(h==0) return
    pi=acos(-1._dp)
    do j=1,p
      do i=1,n
        if(x(i,j)==0) cycle
        z(i,j)=sqrt(x(i,j))*sqrt(h)
        do k=1,self%sample_steps-1
          ! Beyond this limit all harmonics underflow for any finite input.
          if(h>3000._dp/(pi*real(k,dp))) cycle
          a=pi*real(k,dp)*h
          log_amplitude=.5_dp*(log(4._dp)+log(x(i,j))+log(h)-a-log(1+exp(-2*a)))
          amplitude=exp(log_amplitude); phase=real(k,dp)*h*log(x(i,j))
          offset=(2*k-1)*p
          z(i,offset+j)=amplitude*cos(phase)
          z(i,offset+p+j)=amplitude*sin(phase)
        end do
      end do
    end do
  end function

  function additive_fit_transform(self,x) result(z)
    class(additive_chi2_sampler), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x); z=self%transform(x)
  end function

  subroutine skewed_fit(self,x)
    class(skewed_chi2_sampler), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: pi
    call check_x(x)
    call require(self%n_components>=1,'SkewedChi2 n_components must be positive')
    call require(ieee_is_finite(self%skewedness),'nonfinite SkewedChi2 skewedness')
    self%n_features_in_=size(x,2); pi=acos(-1._dp)
    if(allocated(self%random_weights_)) deallocate(self%random_weights_,self%random_offset_)
    allocate(self%random_weights_(size(x,2),self%n_components),self%random_offset_(self%n_components))
    call seed_rng(self%random_state); call random_number(self%random_weights_)
    ! Exclude the possible zero endpoint before inverse-CDF sampling.
    self%random_weights_=max(tiny(1._dp),self%random_weights_)
    self%random_weights_=log(tan(.5_dp*pi*self%random_weights_))/pi
    call random_number(self%random_offset_); self%random_offset_=self%random_offset_*2*pi
  end subroutine

  function skewed_transform(self,x) result(z)
    class(skewed_chi2_sampler), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:),shifted(:,:)
    integer :: i,j
    call require(allocated(self%random_weights_),'SkewedChi2 is not fitted')
    call require(ieee_is_finite(self%skewedness),'nonfinite SkewedChi2 skewedness')
    call check_xy(transpose(x),self%n_features_in_)
    call require(all(x>-self%skewedness),'SkewedChi2 input must exceed -skewedness')
    allocate(shifted(size(x,1),size(x,2)))
    do j=1,size(x,2)
      do i=1,size(x,1)
        if(x(i,j)>0.and.self%skewedness>0) then
          shifted(i,j)=log(max(x(i,j),self%skewedness))+ &
            log(1+min(x(i,j),self%skewedness)/max(x(i,j),self%skewedness))
        else
          shifted(i,j)=log(x(i,j)+self%skewedness)
        end if
      end do
    end do
    z=cos(matmul(shifted,self%random_weights_)+spread(self%random_offset_,1,size(x,1)))* &
      (sqrt(2._dp)/sqrt(real(size(self%random_offset_),dp)))
  end function

  function skewed_fit_transform(self,x) result(z)
    class(skewed_chi2_sampler), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x); z=self%transform(x)
  end function
end module
