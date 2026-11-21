! sklearn 1.6.1 histogram binning; BSD attribution in THIRD_PARTY_NOTICES.md.
module sklearn_histogram_binning
  use sklearn_common, only: dp,require
  use sklearn_legacy_rng, only: legacy_random_state,seed_legacy_uint32
  use sklearn_ieee_math, only: ieee_difference
  use, intrinsic :: iso_fortran_env, only: int16,int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
  implicit none
  private
  public :: histogram_bin_mapper,histogram_thresholds,histogram_argsort
  type :: histogram_thresholds
    real(dp), allocatable :: values(:)
  end type
  type :: histogram_bin_mapper
    integer :: max_bins=255,n_features_in_=0,missing_bin_=256
    type(histogram_thresholds), allocatable :: bin_thresholds_(:)
    integer, allocatable :: n_bins_non_missing_(:)
    logical, allocatable :: has_missing_values_(:)
  contains
    procedure :: fit=>bin_fit
    procedure :: transform=>bin_transform
  end type
contains
  function histogram_argsort(values) result(order)
    real(dp), intent(in) :: values(:)
    integer :: order(size(values))
    integer, allocatable :: work(:)
    integer :: n,width,start,left,right,mid,last,k
    n=size(values);order=[(k,k=1,n)];allocate(work(n));width=1
    do while(width<n)
      start=1
      do while(start<=n)
        mid=start+min(width,n-start+1)-1;last=mid+min(width,n-mid);left=start;right=mid+1
        do k=start,last
          if(left>mid) then
            work(k)=order(right);right=right+1
          else if(right>last) then
            work(k)=order(left);left=left+1
          else if(values(order(left))<=values(order(right))) then
            work(k)=order(left);left=left+1
          else
            work(k)=order(right);right=right+1
          end if
        end do
        if(last==n) exit
        start=last+1
      end do
      order=work
      if(width>=n-width) exit
      width=2*width
    end do
  end function
  subroutine sorted_values(values)
    real(dp), intent(inout) :: values(:)
    real(dp), allocatable :: work(:)
    integer :: n,width,start,left,right,mid,last,k
    n=size(values); allocate(work(n)); width=1
    do while(width<n)
      do start=1,n,2*width
        mid=min(n,start+width-1); last=min(n,start+2*width-1); left=start; right=mid+1
        do k=start,last
          if(left>mid) then
            work(k)=values(right); right=right+1
          else if(right>last) then
            work(k)=values(left); left=left+1
          else if(values(left)<=values(right)) then
            work(k)=values(left); left=left+1
          else
            work(k)=values(right); right=right+1
          end if
        end do
      end do
      values=work; width=2*width
    end do
  end subroutine
  subroutine bin_fit(self,x,seed)
    class(histogram_bin_mapper), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer(int64), intent(in) :: seed
    type(legacy_random_state) :: rng
    integer, allocatable :: subset(:)
    real(dp), allocatable :: values(:),unique(:),thresholds(:)
    real(dp) :: position,midpoint,percentile
    integer :: n,p,ns,j,k,m,count,a,b,t
    n=size(x,1);p=size(x,2)
    call require(n>0.and.p>0,'histogram binning X must be nonempty')
    call require(self%max_bins>=2.and.self%max_bins<=255,'histogram max_bins must be in [2,255]')
    ns=min(n,200000); subset=[(j,j=1,n)]
    if(n>ns) then
      call seed_legacy_uint32(rng,seed)
      do j=n,2,-1
        k=rng%integer_index(j);t=subset(j);subset(j)=subset(k);subset(k)=t
      end do
    end if
    self%n_features_in_=p;self%missing_bin_=self%max_bins+1
    if(allocated(self%bin_thresholds_)) deallocate(self%bin_thresholds_)
    if(allocated(self%n_bins_non_missing_)) deallocate(self%n_bins_non_missing_,self%has_missing_values_)
    allocate(self%bin_thresholds_(p),self%n_bins_non_missing_(p),self%has_missing_values_(p))
    self%has_missing_values_=any(ieee_is_nan(x),dim=1)
    do j=1,p
      values=pack(x(subset(:ns),j),.not.ieee_is_nan(x(subset(:ns),j)))
      call sorted_values(values);m=size(values)
      if(allocated(unique)) deallocate(unique)
      allocate(unique(m));count=0
      do k=1,m
        if(k>1) then
          if(values(k)==values(k-1)) cycle
        end if
        count=count+1;unique(count)=values(k)
      end do
      if(allocated(thresholds)) deallocate(thresholds)
      if(count<=self%max_bins) then
        allocate(thresholds(max(0,count-1)))
        do k=1,size(thresholds)
          thresholds(k)=.5_dp*ieee_difference(unique(k),-unique(k+1))
        end do
      else
        allocate(thresholds(self%max_bins-1))
        do k=1,size(thresholds)
          percentile=real(k,dp)*(100._dp/self%max_bins)
          position=real(m-1,dp)*(percentile/100._dp)
          a=floor(position)+1;b=ceiling(position)+1
          thresholds(k)=values(a)
          if(a/=b) thresholds(k)=ieee_difference(values(b),.5_dp*ieee_difference(values(b),values(a)))
        end do
      end if
      do k=1,size(thresholds)
        midpoint=thresholds(k)
        if(.not.ieee_is_nan(midpoint)) thresholds(k)=min(midpoint,1.e300_dp)
      end do
      self%bin_thresholds_(j)%values=thresholds
      self%n_bins_non_missing_(j)=size(thresholds)+1
    end do
  end subroutine
  function bin_transform(self,x) result(bins)
    class(histogram_bin_mapper), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer(int16), allocatable :: bins(:,:)
    integer :: i,j,lo,hi,mid
    logical :: go_left
    call require(allocated(self%bin_thresholds_),'histogram bin mapper is not fitted')
    call require(size(x,2)==self%n_features_in_,'histogram binning feature width mismatch')
    allocate(bins(size(x,1),size(x,2)))
    do j=1,size(x,2)
      do i=1,size(x,1)
        if(ieee_is_nan(x(i,j))) then
          bins(i,j)=int(self%missing_bin_,int16);cycle
        end if
        lo=1;hi=self%n_bins_non_missing_(j)
        do while(lo<hi)
          mid=(lo+hi)/2;go_left=.false.
          if(.not.ieee_is_nan(self%bin_thresholds_(j)%values(mid))) &
            go_left=x(i,j)<=self%bin_thresholds_(j)%values(mid)
          if(go_left) then
            hi=mid
          else
            lo=mid+1
          end if
        end do
        bins(i,j)=int(lo,int16)
      end do
    end do
  end function
end module
