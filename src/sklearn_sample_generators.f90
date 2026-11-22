! Dense sample generators adapted from sklearn 1.6.1 datasets/_samples_generator.py.
! Upstream BSD-3-Clause attribution: THIRD_PARTY_NOTICES.md and LICENSES/scikit-learn-COPYING.
module sklearn_sample_generators
  use sklearn_common, only: dp,require,argsort
  use sklearn_legacy_rng, only: legacy_random_state
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: make_friedman1,make_friedman2,make_friedman3,make_sparse_uncorrelated
  public :: make_s_curve,make_swiss_roll,make_gaussian_quantiles
contains
  subroutine initialize(rng,n,random_state,noise,std)
    type(legacy_random_state), intent(out) :: rng
    integer, intent(in) :: n
    integer, optional, intent(in) :: random_state
    real(dp), optional, intent(in) :: noise
    real(dp), intent(out) :: std
    integer :: seed
    call require(n>=1,'sample generator n_samples must be positive')
    std=0;seed=0
    if(present(noise)) std=noise
    if(present(random_state)) seed=random_state
    call require(ieee_is_finite(std),'sample generator noise must be finite')
    call require(std>=0,'sample generator noise must be nonnegative')
    call rng%seed(seed)
  end subroutine
  subroutine draw_matrix(rng,x,normal)
    type(legacy_random_state), intent(inout) :: rng
    real(dp), intent(out) :: x(:,:)
    logical, intent(in) :: normal
    integer :: i,j
    ! NumPy fills these matrices in C order, unlike Fortran's array traversal.
    do i=1,size(x,1)
      do j=1,size(x,2)
        if(normal) then
          x(i,j)=rng%normal()
        else
          x(i,j)=rng%uniform()
        end if
      end do
    end do
  end subroutine
  subroutine make_friedman1(n_samples,n_features,x,y,noise,random_state)
    integer, intent(in) :: n_samples,n_features
    real(dp), allocatable, intent(out) :: x(:,:),y(:)
    real(dp), optional, intent(in) :: noise
    integer, optional, intent(in) :: random_state
    type(legacy_random_state) :: rng
    real(dp) :: std,pi
    integer :: i
    call require(n_features>=5,'make_friedman1 needs at least five features')
    call initialize(rng,n_samples,random_state,noise,std);pi=acos(-1._dp)
    allocate(x(n_samples,n_features),y(n_samples));call draw_matrix(rng,x,.false.)
    do i=1,n_samples
      y(i)=10*sin(pi*x(i,1)*x(i,2))+20*(x(i,3)-.5_dp)**2+10*x(i,4)+5*x(i,5)+std*rng%normal()
    end do
  end subroutine
  subroutine friedman23(n_samples,x,y,third,noise,random_state)
    integer, intent(in) :: n_samples
    real(dp), allocatable, intent(out) :: x(:,:),y(:)
    logical, intent(in) :: third
    real(dp), optional, intent(in) :: noise
    integer, optional, intent(in) :: random_state
    type(legacy_random_state) :: rng
    real(dp) :: std,pi,value
    integer :: i
    call initialize(rng,n_samples,random_state,noise,std);pi=acos(-1._dp)
    allocate(x(n_samples,4),y(n_samples));call draw_matrix(rng,x,.false.)
    x(:,1)=100*x(:,1);x(:,2)=520*pi*x(:,2)+40*pi;x(:,4)=10*x(:,4)+1
    do i=1,n_samples
      value=x(i,2)*x(i,3)-1/(x(i,2)*x(i,4))
      if(third) then
        if(x(i,1)>0) then
          y(i)=atan(value/x(i,1))
        else
          y(i)=sign(pi/2,value)
        end if
      else
        y(i)=sqrt(x(i,1)**2+value**2)
      end if
      y(i)=y(i)+std*rng%normal()
    end do
  end subroutine
  subroutine make_friedman2(n_samples,x,y,noise,random_state)
    integer, intent(in) :: n_samples
    real(dp), allocatable, intent(out) :: x(:,:),y(:)
    real(dp), optional, intent(in) :: noise
    integer, optional, intent(in) :: random_state
    call friedman23(n_samples,x,y,.false.,noise,random_state)
  end subroutine
  subroutine make_friedman3(n_samples,x,y,noise,random_state)
    integer, intent(in) :: n_samples
    real(dp), allocatable, intent(out) :: x(:,:),y(:)
    real(dp), optional, intent(in) :: noise
    integer, optional, intent(in) :: random_state
    call friedman23(n_samples,x,y,.true.,noise,random_state)
  end subroutine
  subroutine make_sparse_uncorrelated(n_samples,n_features,x,y,random_state)
    integer, intent(in) :: n_samples,n_features
    real(dp), allocatable, intent(out) :: x(:,:),y(:)
    integer, optional, intent(in) :: random_state
    type(legacy_random_state) :: rng
    real(dp) :: std
    integer :: i
    call require(n_features>=4,'make_sparse_uncorrelated needs at least four features')
    call initialize(rng,n_samples,random_state,std=std)
    allocate(x(n_samples,n_features),y(n_samples));call draw_matrix(rng,x,.true.)
    do i=1,n_samples
      y(i)=x(i,1)+2*x(i,2)-2*x(i,3)-1.5_dp*x(i,4)+rng%normal()
    end do
  end subroutine
  subroutine make_s_curve(n_samples,x,t,noise,random_state)
    integer, intent(in) :: n_samples
    real(dp), allocatable, intent(out) :: x(:,:),t(:)
    real(dp), optional, intent(in) :: noise
    integer, optional, intent(in) :: random_state
    type(legacy_random_state) :: rng
    real(dp) :: std
    integer :: i,j
    call initialize(rng,n_samples,random_state,noise,std)
    allocate(x(n_samples,3),t(n_samples))
    do i=1,n_samples
      t(i)=3*acos(-1._dp)*(rng%uniform()-.5_dp)
    end do
    x(:,1)=sin(t)
    do i=1,n_samples
      x(i,2)=2*rng%uniform()
      if(t(i)==0) then
        x(i,3)=0
      else
        x(i,3)=sign(1._dp,t(i))*(cos(t(i))-1)
      end if
    end do
    ! Source draws shape (3,n_samples), then transposes it.
    do j=1,3
      do i=1,n_samples
        x(i,j)=x(i,j)+std*rng%normal()
      end do
    end do
  end subroutine
  subroutine make_swiss_roll(n_samples,x,t,noise,random_state,hole)
    integer, intent(in) :: n_samples
    real(dp), allocatable, intent(out) :: x(:,:),t(:)
    real(dp), optional, intent(in) :: noise
    integer, optional, intent(in) :: random_state
    logical, optional, intent(in) :: hole
    type(legacy_random_state) :: rng
    real(dp) :: std,pi
    integer :: i,j,corner
    integer, allocatable :: corners(:)
    logical :: cut
    call initialize(rng,n_samples,random_state,noise,std);pi=acos(-1._dp)
    cut=.false.;if(present(hole)) cut=hole
    allocate(x(n_samples,3),t(n_samples))
    if(cut) then
      allocate(corners(n_samples))
      do i=1,n_samples
        corner=rng%integer_index(8)-1
        if(corner>=4) corner=corner+1
        corners(i)=corner
      end do
      do i=1,n_samples
        t(i)=pi*(1.5_dp+corners(i)/3)+pi*rng%uniform()
      end do
      do i=1,n_samples
        x(i,2)=7*mod(corners(i),3)+7*rng%uniform()
      end do
    else
      do i=1,n_samples
        t(i)=1.5_dp*pi*(1+2*rng%uniform())
      end do
      do i=1,n_samples
        x(i,2)=21*rng%uniform()
      end do
    end if
    x(:,1)=t*cos(t);x(:,3)=t*sin(t)
    do j=1,3
      do i=1,n_samples
        x(i,j)=x(i,j)+std*rng%normal()
      end do
    end do
  end subroutine
  subroutine make_gaussian_quantiles(n_samples,n_features,x,y,n_classes,mean,cov,shuffle,random_state)
    integer, intent(in) :: n_samples,n_features
    real(dp), allocatable, intent(out) :: x(:,:)
    integer, allocatable, intent(out) :: y(:)
    integer, optional, intent(in) :: n_classes,random_state
    real(dp), optional, intent(in) :: mean(:),cov
    logical, optional, intent(in) :: shuffle
    type(legacy_random_state) :: rng
    real(dp), allocatable :: center(:),distances(:)
    integer, allocatable :: order(:)
    real(dp) :: std,variance
    integer :: classes,step,i,j,tmp
    logical :: shuf
    call initialize(rng,n_samples,random_state,std=std)
    call require(n_features>=1,'make_gaussian_quantiles n_features must be positive')
    classes=3;variance=1;shuf=.true.
    if(present(n_classes)) classes=n_classes
    if(present(cov)) variance=cov
    if(present(shuffle)) shuf=shuffle
    call require(classes>=1,'make_gaussian_quantiles n_classes must be positive')
    call require(n_samples>=classes,'make_gaussian_quantiles n_samples must be at least n_classes')
    call require(ieee_is_finite(variance),'make_gaussian_quantiles cov must be finite')
    call require(variance>=0,'make_gaussian_quantiles cov must be nonnegative')
    allocate(center(n_features));center=0
    if(present(mean)) then
      call require(size(mean)==n_features,'make_gaussian_quantiles mean length mismatch')
      call require(all(ieee_is_finite(mean)),'make_gaussian_quantiles mean must be finite');center=mean
    end if
    allocate(x(n_samples,n_features),y(n_samples),distances(n_samples));call draw_matrix(rng,x,.true.)
    do i=1,n_samples
      x(i,:)=sqrt(variance)*x(i,:)+center
      distances(i)=sum((x(i,:)-center)**2)
    end do
    order=argsort(distances);x=x(order,:);step=n_samples/classes
    do i=1,n_samples
      y(i)=min((i-1)/step,classes-1)
    end do
    if(shuf) then
      order=[(i,i=1,n_samples)]
      do i=n_samples,2,-1
        j=rng%integer_index(i);tmp=order(i);order(i)=order(j);order(j)=tmp
      end do
      x=x(order,:);y=y(order)
    end if
  end subroutine
end module
