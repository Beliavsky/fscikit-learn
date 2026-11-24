! sklearn 1.6.1 datasets/_samples_generator.py bicluster contracts (BSD).
! NumPy 2.2.6 binomial inversion/BTPE kernels (BSD; retained distribution license).
module sklearn_bicluster_datasets
  use sklearn_common, only: dp,require
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_random_binomial, only: binomial=>legacy_binomial
  use sklearn_ieee_math, only: ieee_product,ieee_difference
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  private
  public :: make_biclusters,make_checkerboard
  interface make_checkerboard
    module procedure checker_scalar,checker_pair
  end interface
contains
  function partition(n,k,rng) result(labels)
    integer, intent(in) :: n,k
    type(legacy_random_state), intent(inout) :: rng
    integer, allocatable :: labels(:),counts(:)
    real(dp) :: probability,remaining_p
    integer :: remaining,i,first
    allocate(labels(n),counts(k));counts=0;remaining=n;remaining_p=1;probability=1._dp/k
    do i=1,k-1
      counts(i)=binomial(remaining,probability/remaining_p,rng);remaining=remaining-counts(i)
      if(remaining<=0) exit
      remaining_p=remaining_p-probability
    end do
    if(remaining>0) counts(k)=remaining
    first=1
    do i=1,k
      labels(first:first+counts(i)-1)=i-1;first=first+counts(i)
    end do
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
  subroutine make_biclusters(shape,n_clusters,data,rows,cols,noise,minval,maxval,shuffle,random_state)
    integer, intent(in) :: shape(:),n_clusters
    real(dp), allocatable, intent(out) :: data(:,:)
    logical, allocatable, intent(out) :: rows(:,:),cols(:,:)
    real(dp), optional, intent(in) :: noise,minval,maxval
    logical, optional, intent(in) :: shuffle
    integer, optional, intent(in) :: random_state
    call generate(shape,n_clusters,n_clusters,.true.,data,rows,cols,noise,minval,maxval,shuffle,random_state)
  end subroutine
  subroutine checker_scalar(shape,n_clusters,data,rows,cols,noise,minval,maxval,shuffle,random_state)
    integer, intent(in) :: shape(:),n_clusters
    real(dp), allocatable, intent(out) :: data(:,:)
    logical, allocatable, intent(out) :: rows(:,:),cols(:,:)
    real(dp), optional, intent(in) :: noise,minval,maxval
    logical, optional, intent(in) :: shuffle
    integer, optional, intent(in) :: random_state
    call generate(shape,n_clusters,n_clusters,.false.,data,rows,cols,noise,minval,maxval,shuffle,random_state)
  end subroutine
  subroutine checker_pair(shape,n_clusters,data,rows,cols,noise,minval,maxval,shuffle,random_state)
    integer, intent(in) :: shape(:),n_clusters(:)
    real(dp), allocatable, intent(out) :: data(:,:)
    logical, allocatable, intent(out) :: rows(:,:),cols(:,:)
    real(dp), optional, intent(in) :: noise,minval,maxval
    logical, optional, intent(in) :: shuffle
    integer, optional, intent(in) :: random_state
    call require(size(n_clusters)==2,'checkerboard cluster counts must have two entries')
    call generate(shape,n_clusters(1),n_clusters(2),.false.,data,rows,cols,noise,minval,maxval,shuffle,random_state)
  end subroutine
  subroutine generate(shape,kr,kc,diagonal,data,rows,cols,noise,minval,maxval,shuffle,random_state)
    integer, intent(in) :: shape(:),kr,kc
    logical, intent(in) :: diagonal
    real(dp), allocatable, intent(out) :: data(:,:)
    logical, allocatable, intent(out) :: rows(:,:),cols(:,:)
    real(dp), optional, intent(in) :: noise,minval,maxval
    logical, optional, intent(in) :: shuffle
    integer, optional, intent(in) :: random_state
    type(legacy_random_state) :: rng,shuffle_rng
    real(dp), allocatable :: constants(:)
    integer, allocatable :: row_labels(:),col_labels(:),ri(:),ci(:)
    real(dp) :: std,low,high,width,value
    logical :: shuf
    integer :: n,p,seed,i,j,r,c,blocks,index
    call require(size(shape)==2,'bicluster shape must have two entries')
    call require(all(shape>=0),'bicluster dimensions must be nonnegative')
    call require(kr>=1.and.kc>=1,'bicluster cluster counts must be positive')
    n=shape(1);p=shape(2);std=0;low=10;high=100;seed=0;shuf=.true.
    if(present(noise)) std=noise
    if(present(minval)) low=minval
    if(present(maxval)) high=maxval
    if(present(random_state)) seed=random_state
    if(present(shuffle)) shuf=shuffle
    call require(ieee_is_finite(std),'bicluster noise must be finite')
    call require(std>=0,'bicluster noise must be nonnegative')
    call require(ieee_is_finite(low).and.ieee_is_finite(high),'bicluster bounds must be finite')
    width=ieee_difference(high,low)
    call require(ieee_is_finite(width),'bicluster value range must be finite')
    call require(width>=0,'bicluster maxval must be at least minval')
    call rng%seed(seed)
    if(diagonal) then
      blocks=kr;allocate(constants(kr))
      do r=1,kr
        constants(r)=ieee_difference(low,-ieee_product(width,rng%uniform()))
      end do
    else
      call require(int(kr,int64)*kc<=int(huge(blocks),int64),'checkerboard block count exceeds index range')
      blocks=kr*kc
    end if
    row_labels=partition(n,kr,rng);col_labels=partition(p,kc,rng);allocate(data(n,p));data=0
    if(diagonal) then
      do i=1,n
        do j=1,p
          if(row_labels(i)==col_labels(j)) data(i,j)=constants(row_labels(i)+1)
        end do
      end do
    else
      do r=0,kr-1
        do c=0,kc-1
          value=ieee_difference(low,-ieee_product(width,rng%uniform()))
          do i=1,n
            if(row_labels(i)/=r) cycle
            do j=1,p
              if(col_labels(j)==c) data(i,j)=value
            end do
          end do
        end do
      end do
    end if
    if(std>0) then
      do i=1,n
        do j=1,p
          data(i,j)=ieee_difference(data(i,j),-ieee_product(std,rng%normal()))
        end do
      end do
    end if
    if(shuf) then
      ! Integer-seeded sklearn _shuffle starts its own stream at the original seed.
      call shuffle_rng%seed(seed);ri=permutation(n,shuffle_rng);ci=permutation(p,shuffle_rng)
      data=data(ri,ci);row_labels=row_labels(ri);col_labels=col_labels(ci)
    end if
    allocate(rows(blocks,n),cols(blocks,p))
    if(diagonal) then
      do r=0,kr-1
        rows(r+1,:)=row_labels==r;cols(r+1,:)=col_labels==r
      end do
    else
      index=0
      do r=0,kr-1
        do c=0,kc-1
          index=index+1;rows(index,:)=row_labels==r;cols(index,:)=col_labels==c
        end do
      end do
    end if
  end subroutine
end module
