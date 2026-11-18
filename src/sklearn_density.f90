! sklearn 1.6.1 neighbors/_kde.py and _binary_tree.pxi kernel conventions.
! See THIRD_PARTY_NOTICES.md. Dense exact Euclidean evaluation; no tree backend.
module sklearn_density
  use sklearn_common
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_negative_inf,ieee_is_finite
  implicit none
  private
  public :: kernel_density
  type :: kernel_density
    real(dp) :: bandwidth=1,bandwidth_=1
    character(16) :: kernel='gaussian',bandwidth_rule='fixed'
    real(dp), allocatable :: x_(:,:),weights_(:)
  contains
    procedure :: fit=>density_fit
    procedure :: score_samples=>density_scores
    procedure :: score=>density_score
    procedure :: sample=>density_sample
  end type
contains
  subroutine density_fit(self,x,sample_weight)
    class(kernel_density), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(in), optional :: sample_weight(:)
    integer :: n,p
    call check_x(x); n=size(x,1); p=size(x,2)
    call require(self%kernel=='gaussian'.or.self%kernel=='tophat'.or.self%kernel=='epanechnikov'.or. &
      self%kernel=='exponential'.or.self%kernel=='linear','unknown density kernel')
    select case(self%bandwidth_rule)
    case('fixed'); self%bandwidth_=self%bandwidth
    case('scott'); self%bandwidth_=real(n,dp)**(-1._dp/(p+4))
    case('silverman'); self%bandwidth_=(real(n,dp)*(p+2)/4)**(-1._dp/(p+4))
    case default; call require(.false.,'unknown density bandwidth rule')
    end select
    call require(self%bandwidth_>0.and.ieee_is_finite(self%bandwidth_),'density bandwidth must be positive and finite')
    self%weights_=spread(1._dp,1,n)
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'density sample weight size mismatch')
      call check_x(reshape(sample_weight,[n,1]))
      call require(all(sample_weight>=0).and.sum(sample_weight)>0,'invalid density sample weights')
      self%weights_=sample_weight
    end if
    self%weights_=self%weights_/maxval(self%weights_)
    self%weights_=self%weights_/sum(self%weights_); self%x_=x
  end subroutine
  function density_scores(self,x) result(scores)
    class(kernel_density), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: scores(size(x,1)),r,log_norm,log_volume,neg_inf
    real(dp), allocatable :: terms(:)
    integer :: i,j,p
    call require(allocated(self%x_),'KernelDensity is not fitted')
    call check_xy(transpose(x),size(self%x_,2)); p=size(x,2)
    allocate(terms(size(self%weights_)))
    neg_inf=ieee_value(0._dp,ieee_negative_inf)
    log_volume=0.5_dp*p*log(acos(-1._dp))-log_gamma(0.5_dp*p+1)
    select case(self%kernel)
    case('gaussian'); log_norm=-0.5_dp*p*log(2*acos(-1._dp))
    case('tophat'); log_norm=-log_volume
    case('epanechnikov'); log_norm=log((p+2)*0.5_dp)-log_volume
    case('exponential'); log_norm=-log(real(p,dp))-log_volume-log_gamma(real(p,dp))
    case('linear'); log_norm=log(real(p+1,dp))-log_volume
    case default; call require(.false.,'unknown density kernel')
    end select
    log_norm=log_norm-p*log(self%bandwidth_)
    do i=1,size(x,1)
      terms=neg_inf
      do j=1,size(self%x_,1)
        if(self%weights_(j)==0) cycle
        r=sqrt(sum((x(i,:)-self%x_(j,:))**2))/self%bandwidth_
        select case(self%kernel)
        case('gaussian'); terms(j)=-0.5_dp*r*r
        case('exponential'); terms(j)=-r
        case('tophat'); if(r<1) terms(j)=0
        case('epanechnikov'); if(r<1) terms(j)=log(1-r*r)
        case('linear'); if(r<1) terms(j)=log(1-r)
        end select
        terms(j)=terms(j)+log(self%weights_(j))
      end do
      scores(i)=neg_inf
      if(maxval(terms)>neg_inf) scores(i)=logsumexp(terms)+log_norm
    end do
  end function
  real(dp) function density_score(self,x) result(score)
    class(kernel_density), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    score=sum(self%score_samples(x))
  end function
  function density_sample(self,n_samples,random_state) result(x)
    class(kernel_density), intent(in) :: self
    integer, intent(in), optional :: n_samples,random_state
    real(dp), allocatable :: x(:,:)
    real(dp) :: coin,cumulative,radius,norm
    real(dp), allocatable :: direction(:,:)
    integer :: n,p,i,j,k
    call require(allocated(self%x_),'KernelDensity is not fitted')
    call require(self%kernel=='gaussian'.or.self%kernel=='tophat','density sampling supports gaussian/tophat only')
    n=1; if(present(n_samples)) n=n_samples
    call require(n>0,'density sample count must be positive')
    if(present(random_state)) call seed_rng(random_state)
    p=size(self%x_,2); allocate(x(n,p))
    do i=1,n
      call random_number(coin); cumulative=0; j=maxloc(self%weights_,dim=1)
      do k=1,size(self%x_,1)
        cumulative=cumulative+self%weights_(k)
        if(coin<cumulative) then
          j=k; exit
        end if
      end do
      direction=normal_random(1,p)
      if(self%kernel=='tophat') then
        norm=sqrt(sum(direction**2))
        if(norm==0) then
          direction=0; direction(1,1)=1; norm=1
        end if
        call random_number(radius); direction=direction/norm*radius**(1._dp/p)
      end if
      x(i,:)=self%x_(j,:)+self%bandwidth_*direction(1,:)
    end do
  end function
end module
