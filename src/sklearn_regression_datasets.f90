! sklearn 1.6.1 datasets/_samples_generator.py (BSD contracts).
! Native real64 QR, rank adapters and validation are MIT contributions.
module sklearn_regression_datasets
  use sklearn_common, only: dp,require
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_ieee_math, only: ieee_product,ieee_difference
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: make_low_rank_matrix,make_regression
  interface make_regression
    module procedure regression_vector,regression_matrix
  end interface
contains
  subroutine seed_generator(rng,random_state)
    type(legacy_random_state), intent(out) :: rng
    integer, optional, intent(in) :: random_state
    integer :: seed
    seed=0;if(present(random_state)) seed=random_state
    call rng%seed(seed)
  end subroutine
  subroutine draw_normal(rng,a)
    type(legacy_random_state), intent(inout) :: rng
    real(dp), intent(out) :: a(:,:)
    integer :: i,j
    do i=1,size(a,1)
      do j=1,size(a,2)
        a(i,j)=rng%normal()
      end do
    end do
  end subroutine
  real(dp) function scaled_norm(v) result(value)
    real(dp), intent(in) :: v(:)
    value=0;if(size(v)==0) return
    value=maxval(abs(v));if(value>0) value=value*sqrt(sum((v/value)**2))
  end function
  function thin_qr(a) result(q)
    real(dp), intent(in) :: a(:,:)
    real(dp), allocatable :: q(:,:),work(:,:),vectors(:,:),tau(:),v(:),product(:)
    real(dp) :: norm,alpha,beta
    integer :: m,r,j
    m=size(a,1);r=size(a,2);work=a
    allocate(q(m,r),vectors(m,r),tau(r));q=0;vectors=0;tau=0
    do j=1,r
      norm=scaled_norm(work(j+1:,j));alpha=work(j,j)
      ! LAPACK DLARFG leaves a length-one/zero-tail column unchanged.
      if(norm==0) cycle
      beta=-sign(scaled_norm([alpha,norm]),alpha)
      tau(j)=(beta-alpha)/beta;v=work(j:,j);v(1)=1
      v(2:)=v(2:)/(alpha-beta);vectors(j:,j)=v
      if(j<r) then
        product=matmul(v,work(j:,j+1:))
        work(j:,j+1:)=work(j:,j+1:)-tau(j)*spread(v,2,r-j)*spread(product,1,m-j+1)
      end if
    end do
    do j=1,r
      q(j,j)=1
    end do
    do j=r,1,-1
      if(tau(j)==0) cycle
      product=matmul(vectors(j:,j),q(j:,:))
      q(j:,:)=q(j:,:)-tau(j)*spread(vectors(j:,j),2,r)*spread(product,1,m-j+1)
    end do
  end function
  function low_rank(n,p,rank,tail,rng) result(x)
    integer, intent(in) :: n,p,rank
    real(dp), intent(in) :: tail
    type(legacy_random_state), intent(inout) :: rng
    real(dp), allocatable :: x(:,:),a(:,:),u(:,:),v(:,:),s(:)
    real(dp) :: index
    integer :: r,j
    r=min(n,p);allocate(a(n,r));call draw_normal(rng,a);u=thin_qr(a)
    deallocate(a);allocate(a(p,r));call draw_normal(rng,a);v=thin_qr(a)
    allocate(s(r))
    do j=1,r
      index=real(j-1,dp)/rank
      s(j)=(1-tail)*exp(-index**2)+tail*exp(-.1_dp*index)
    end do
    x=matmul(u*spread(s,1,n),transpose(v))
  end function
  function make_low_rank_matrix(n_samples,n_features,effective_rank,tail_strength,random_state) result(x)
    integer, optional, intent(in) :: n_samples,n_features,effective_rank,random_state
    real(dp), optional, intent(in) :: tail_strength
    real(dp), allocatable :: x(:,:)
    integer :: n,p,rank
    real(dp) :: tail
    type(legacy_random_state) :: rng
    n=100;p=100;rank=10;tail=.5_dp
    if(present(n_samples)) n=n_samples
    if(present(n_features)) p=n_features
    if(present(effective_rank)) rank=effective_rank
    if(present(tail_strength)) tail=tail_strength
    call require(n>=1.and.p>=1,'low rank matrix dimensions must be positive')
    call require(rank>=1,'effective_rank must be positive')
    call require(ieee_is_finite(tail),'tail_strength must be finite')
    call require(tail>=0.and.tail<=1,'tail_strength must be in [0,1]')
    call seed_generator(rng,random_state);x=low_rank(n,p,rank,tail,rng)
  end function
  function permutation(n,rng) result(order)
    integer, intent(in) :: n
    type(legacy_random_state), intent(inout) :: rng
    integer :: order(n),i,j,temp
    order=[(i,i=1,n)]
    do i=n,2,-1
      j=rng%integer_index(i);temp=order(i);order(i)=order(j);order(j)=temp
    end do
  end function
  subroutine regression_body(n,p,nt,x,y,coefficients,n_informative,noise,bias,random_state, &
      effective_rank,tail_strength,shuffle)
    integer, intent(in) :: n,p,nt
    real(dp), allocatable, intent(out) :: x(:,:),y(:,:),coefficients(:,:)
    integer, optional, intent(in) :: n_informative,random_state,effective_rank
    real(dp), optional, intent(in) :: noise,bias,tail_strength
    logical, optional, intent(in) :: shuffle
    integer :: info,i,j
    integer, allocatable :: order(:)
    real(dp) :: std,offset,tail
    logical :: shuf
    type(legacy_random_state) :: rng
    call require(n>=1.and.p>=1,'regression dimensions must be positive')
    call require(nt>=1,'n_targets must be positive')
    info=10;std=0;offset=0;tail=.5_dp;shuf=.true.
    if(present(n_informative)) info=n_informative
    if(present(noise)) std=noise
    if(present(bias)) offset=bias
    if(present(tail_strength)) tail=tail_strength
    if(present(shuffle)) shuf=shuffle
    call require(info>=0,'n_informative must be nonnegative');info=min(info,p)
    call require(ieee_is_finite(std),'regression noise must be finite')
    call require(std>=0,'regression noise must be nonnegative')
    call require(ieee_is_finite(offset),'regression bias must be finite')
    call require(ieee_is_finite(tail),'tail_strength must be finite')
    call require(tail>=0.and.tail<=1,'tail_strength must be in [0,1]')
    if(present(effective_rank)) call require(effective_rank>=1,'effective_rank must be positive')
    call seed_generator(rng,random_state)
    if(present(effective_rank)) then
      x=low_rank(n,p,effective_rank,tail,rng)
    else
      allocate(x(n,p));call draw_normal(rng,x)
    end if
    allocate(coefficients(p,nt));coefficients=0
    do i=1,info
      do j=1,nt
        coefficients(i,j)=100*rng%uniform()
      end do
    end do
    y=matmul(x,coefficients)+offset
    if(std>0) then
      do i=1,n
        do j=1,nt
          y(i,j)=ieee_difference(y(i,j),-ieee_product(std,rng%normal()))
        end do
      end do
    end if
    if(shuf) then
      order=permutation(n,rng);x=x(order,:);y=y(order,:)
      order=permutation(p,rng);x=x(:,order);coefficients=coefficients(order,:)
    end if
  end subroutine
  subroutine regression_vector(n_samples,n_features,x,y,n_informative,noise,bias,random_state,coef, &
      n_targets,effective_rank,tail_strength,shuffle)
    integer, intent(in) :: n_samples,n_features
    real(dp), allocatable, intent(out) :: x(:,:),y(:)
    integer, optional, intent(in) :: n_informative,random_state,n_targets,effective_rank
    real(dp), optional, intent(in) :: noise,bias,tail_strength
    real(dp), allocatable, optional, intent(out) :: coef(:)
    logical, optional, intent(in) :: shuffle
    real(dp), allocatable :: target(:,:),weights(:,:)
    if(present(n_targets)) then
      call require(n_targets>=1,'n_targets must be positive')
      call require(n_targets==1,'use matrix y for multiple regression targets')
    end if
    call regression_body(n_samples,n_features,1,x,target,weights,n_informative,noise,bias,random_state, &
      effective_rank,tail_strength,shuffle)
    y=target(:,1);if(present(coef)) coef=weights(:,1)
  end subroutine
  subroutine regression_matrix(n_samples,n_features,x,y,n_informative,noise,bias,random_state,coef, &
      n_targets,effective_rank,tail_strength,shuffle)
    integer, intent(in) :: n_samples,n_features
    real(dp), allocatable, intent(out) :: x(:,:),y(:,:)
    integer, optional, intent(in) :: n_informative,random_state,n_targets,effective_rank
    real(dp), optional, intent(in) :: noise,bias,tail_strength
    real(dp), allocatable, optional, intent(out) :: coef(:,:)
    logical, optional, intent(in) :: shuffle
    integer :: nt
    real(dp), allocatable :: weights(:,:)
    nt=1;if(present(n_targets)) nt=n_targets
    call regression_body(n_samples,n_features,nt,x,y,weights,n_informative,noise,bias,random_state, &
      effective_rank,tail_strength,shuffle)
    if(present(coef)) coef=weights
  end subroutine
end module
