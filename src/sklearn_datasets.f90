module sklearn_datasets
  ! sklearn 1.6.1 datasets/_samples_generator.py (BSD contracts).
  ! Native rank adapters and quiet IEEE arithmetic are MIT contributions.
  use sklearn_common, only: dp,require
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_ieee_math, only: ieee_product,ieee_difference
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan
  use, intrinsic :: iso_fortran_env, only: int64
  use sklearn_structured_datasets, only: make_sparse_coded_signal,make_sparse_spd_matrix
  use sklearn_bicluster_datasets, only: make_biclusters,make_checkerboard
  use sklearn_multilabel_datasets, only: make_multilabel_classification
  use sklearn_regression_datasets, only: make_regression,make_low_rank_matrix
  use sklearn_classification_datasets, only: make_classification,make_hastie_10_2,classification_dataset_options
  use sklearn_sample_generators, only: make_friedman1,make_friedman2,make_friedman3,make_sparse_uncorrelated, &
    make_s_curve,make_swiss_roll,make_gaussian_quantiles
  implicit none
  private
  public :: make_blobs,make_regression,make_low_rank_matrix,make_moons,make_circles
  public :: make_sparse_coded_signal,make_sparse_spd_matrix
  public :: make_biclusters,make_checkerboard
  public :: make_multilabel_classification
  public :: make_classification,make_hastie_10_2,classification_dataset_options
  public :: make_friedman1,make_friedman2,make_friedman3,make_sparse_uncorrelated
  public :: make_s_curve,make_swiss_roll,make_gaussian_quantiles
  interface make_blobs
    module procedure blobs_total,blobs_counts
  end interface
  interface make_moons
    module procedure moons_total,moons_counts
  end interface
  interface make_circles
    module procedure circles_total,circles_counts
  end interface
contains
  subroutine initialize(rng,random_state)
    type(legacy_random_state), intent(out) :: rng
    integer, optional, intent(in) :: random_state
    integer :: seed
    seed=0;if(present(random_state)) seed=random_state
    call rng%seed(seed)
  end subroutine
  function permutation(n,rng) result(order)
    integer, intent(in) :: n
    type(legacy_random_state), intent(inout) :: rng
    integer :: order(n),i,j,temp
    order=[(i,i=1,n)]
    do i=n,2,-1
      j=rng%integer_index(i);temp=order(i);order(i)=order(j);order(j)=temp
    end do
  end function
  integer function total_samples(counts) result(n)
    integer, intent(in) :: counts(:)
    integer(int64) :: total
    call require(size(counts)>0,'sample counts must be nonempty')
    call require(all(counts>=0),'sample counts must be nonnegative')
    total=sum(int(counts,int64))
    call require(total<=int(huge(n),int64),'sample count total is too large');n=int(total)
  end function
  subroutine blobs_total(n_samples,n_features,n_centers,x,y,random_state,cluster_std,centers, &
      center_box,shuffle,centers_out)
    integer, intent(in) :: n_samples,n_features
    integer, optional, intent(in) :: n_centers,random_state
    real(dp), allocatable, intent(out) :: x(:,:)
    integer, allocatable, intent(out) :: y(:)
    real(dp), optional, intent(in) :: cluster_std(..),centers(:,:),center_box(:)
    logical, optional, intent(in) :: shuffle
    real(dp), allocatable, optional, intent(out) :: centers_out(:,:)
    integer :: k
    integer, allocatable :: counts(:)
    call require(n_samples>=1,'blob n_samples must be positive')
    k=3;if(present(n_centers)) k=n_centers
    call require(k>=1,'blob n_centers must be positive')
    if(present(centers)) then
      if(present(n_centers)) call require(size(centers,1)==k,'blob centers count mismatch')
      k=size(centers,1);call require(k>=1,'blob centers must be nonempty')
    end if
    allocate(counts(k));counts=n_samples/k;counts(:mod(n_samples,k))=counts(:mod(n_samples,k))+1
    call blobs_body(counts,n_features,x,y,random_state,cluster_std,centers,center_box,shuffle,centers_out)
  end subroutine
  subroutine blobs_counts(n_samples,n_features,n_centers,x,y,random_state,cluster_std,centers, &
      center_box,shuffle,centers_out)
    integer, intent(in) :: n_samples(:),n_features
    integer, optional, intent(in) :: n_centers,random_state
    real(dp), allocatable, intent(out) :: x(:,:)
    integer, allocatable, intent(out) :: y(:)
    real(dp), optional, intent(in) :: cluster_std(..),centers(:,:),center_box(:)
    logical, optional, intent(in) :: shuffle
    real(dp), allocatable, optional, intent(out) :: centers_out(:,:)
    if(present(n_centers)) call require(n_centers==size(n_samples),'blob n_centers and count lengths differ')
    call blobs_body(n_samples,n_features,x,y,random_state,cluster_std,centers,center_box,shuffle,centers_out)
  end subroutine
  subroutine blobs_body(counts,n_features,x,y,random_state,cluster_std,centers,center_box,shuffle,centers_out)
    integer, intent(in) :: counts(:),n_features
    integer, optional, intent(in) :: random_state
    real(dp), allocatable, intent(out) :: x(:,:)
    integer, allocatable, intent(out) :: y(:)
    real(dp), optional, intent(in) :: cluster_std(..),centers(:,:),center_box(:)
    logical, optional, intent(in) :: shuffle
    real(dp), allocatable, optional, intent(out) :: centers_out(:,:)
    type(legacy_random_state) :: rng
    real(dp), allocatable :: means(:,:),stds(:)
    integer, allocatable :: order(:)
    real(dp) :: low,high,width
    logical :: shuf
    integer :: n,p,k,i,j,c,row
    n=total_samples(counts);k=size(counts);p=n_features
    call require(p>=1,'blob n_features must be positive')
    allocate(stds(k));stds=1
    if(present(cluster_std)) then
      select rank(scale=>cluster_std)
      rank(0)
        call require(ieee_is_finite(scale),'scalar cluster_std must be finite')
        call require(scale>=0,'cluster_std must be nonnegative');stds=scale
      rank(1)
        call require(size(scale)==k,'cluster_std length must match centers')
        do c=1,k
          if(.not.ieee_is_nan(scale(c))) call require(scale(c)>=0,'cluster_std must be nonnegative')
        end do
        stds=scale
      rank default
        call require(.false.,'cluster_std must be scalar or a vector')
      end select
    end if
    call initialize(rng,random_state)
    if(present(centers)) then
      call require(size(centers,1)==k,'blob centers count mismatch')
      call require(size(centers,2)>0,'blob centers must have features')
      call require(all(ieee_is_finite(centers)),'blob centers must be finite')
      p=size(centers,2);means=centers
    else
      low=-10;high=10
      if(present(center_box)) then
        call require(size(center_box)==2,'center_box must have two bounds')
        low=center_box(1);high=center_box(2)
      end if
      call require(ieee_is_finite(low).and.ieee_is_finite(high),'center_box must be finite')
      width=ieee_difference(high,low)
      call require(ieee_is_finite(width),'center_box range must be finite')
      call require(width>=0,'center_box upper bound must be at least its lower bound')
      allocate(means(k,p))
      do c=1,k
        do j=1,p
          means(c,j)=ieee_difference(low,-ieee_product(width,rng%uniform()))
        end do
      end do
    end if
    allocate(x(n,p),y(n));row=0
    do c=1,k
      do i=1,counts(c)
        row=row+1;y(row)=c-1
        do j=1,p
          x(row,j)=ieee_difference(means(c,j),-ieee_product(stds(c),rng%normal()))
        end do
      end do
    end do
    shuf=.true.;if(present(shuffle)) shuf=shuffle
    if(shuf) then
      order=permutation(n,rng);x=x(order,:);y=y(order)
    end if
    if(present(centers_out)) centers_out=means
  end subroutine
  subroutine moons_total(n_samples,x,y,noise,random_state,shuffle)
    integer, intent(in) :: n_samples
    real(dp), allocatable, intent(out) :: x(:,:)
    integer, allocatable, intent(out) :: y(:)
    real(dp), optional, intent(in) :: noise
    integer, optional, intent(in) :: random_state
    logical, optional, intent(in) :: shuffle
    call require(n_samples>=1,'moon n_samples must be positive')
    call curves_body(.false.,[n_samples/2,n_samples-n_samples/2],x,y,noise=noise, &
      random_state=random_state,shuffle=shuffle)
  end subroutine
  subroutine moons_counts(n_samples,x,y,noise,random_state,shuffle)
    integer, intent(in) :: n_samples(:)
    real(dp), allocatable, intent(out) :: x(:,:)
    integer, allocatable, intent(out) :: y(:)
    real(dp), optional, intent(in) :: noise
    integer, optional, intent(in) :: random_state
    logical, optional, intent(in) :: shuffle
    call curves_body(.false.,n_samples,x,y,noise=noise,random_state=random_state,shuffle=shuffle)
  end subroutine
  subroutine circles_total(n_samples,x,y,noise,factor,random_state,shuffle)
    integer, intent(in) :: n_samples
    real(dp), allocatable, intent(out) :: x(:,:)
    integer, allocatable, intent(out) :: y(:)
    real(dp), optional, intent(in) :: noise,factor
    integer, optional, intent(in) :: random_state
    logical, optional, intent(in) :: shuffle
    call require(n_samples>=1,'circle n_samples must be positive')
    call curves_body(.true.,[n_samples/2,n_samples-n_samples/2],x,y,noise,factor,random_state,shuffle)
  end subroutine
  subroutine circles_counts(n_samples,x,y,noise,factor,random_state,shuffle)
    integer, intent(in) :: n_samples(:)
    real(dp), allocatable, intent(out) :: x(:,:)
    integer, allocatable, intent(out) :: y(:)
    real(dp), optional, intent(in) :: noise,factor
    integer, optional, intent(in) :: random_state
    logical, optional, intent(in) :: shuffle
    call curves_body(.true.,n_samples,x,y,noise,factor,random_state,shuffle)
  end subroutine
  subroutine curves_body(circle,counts,x,y,noise,factor,random_state,shuffle)
    logical, intent(in) :: circle
    integer, intent(in) :: counts(:)
    real(dp), allocatable, intent(out) :: x(:,:)
    integer, allocatable, intent(out) :: y(:)
    real(dp), optional, intent(in) :: noise,factor
    integer, optional, intent(in) :: random_state
    logical, optional, intent(in) :: shuffle
    type(legacy_random_state) :: rng
    integer, allocatable :: order(:)
    integer :: n,c,i,j,m,row
    real(dp) :: pi,theta,step,std,f
    logical :: shuf
    call require(size(counts)==2,'curve sample counts must have two entries')
    n=total_samples(counts);std=0;f=.8_dp
    if(present(noise)) std=noise
    call require(ieee_is_finite(std),'curve noise must be finite')
    call require(std>=0,'curve noise must be nonnegative')
    if(present(factor)) f=factor
    if(circle) then
      call require(ieee_is_finite(f),'circle factor must be finite')
      call require(f>=0.and.f<1,'circle factor must be in [0,1)')
    end if
    call initialize(rng,random_state);pi=acos(-1._dp);allocate(x(n,2),y(n));row=0
    do c=1,2
      m=counts(c);step=0
      if(circle.and.m>0) step=2*pi/m
      if(.not.circle.and.m>1) step=pi/(m-1)
      do i=1,m
        theta=step*(i-1)
        if(.not.circle.and.m>1.and.i==m) theta=pi
        row=row+1;y(row)=c-1
        if(circle) then
          x(row,:)=[cos(theta),sin(theta)]
          if(c==2) x(row,:)=x(row,:)*f
        else if(c==1) then
          x(row,:)=[cos(theta),sin(theta)]
        else
          x(row,:)=[1-cos(theta),1-sin(theta)-.5_dp]
        end if
      end do
    end do
    shuf=.true.;if(present(shuffle)) shuf=shuffle
    if(shuf) then
      order=permutation(n,rng);x=x(order,:);y=y(order)
    end if
    if(present(noise)) then
      do i=1,n
        do j=1,2
          x(i,j)=ieee_difference(x(i,j),-ieee_product(std,rng%normal()))
        end do
      end do
    end if
  end subroutine
end module
