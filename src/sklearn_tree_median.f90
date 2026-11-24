! Internal sklearn 1.6.1 tree/_utils.pyx weighted-median stream (BSD-3-Clause).
! See LICENSES/scikit-learn-COPYING and THIRD_PARTY_NOTICES.md.
module sklearn_tree_median
  use sklearn_common, only: dp,require
  use sklearn_ieee_math, only: ieee_difference,ieee_quotient
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan,ieee_value,ieee_quiet_nan
  implicit none
  private
  public :: tree_median_queue
  type :: tree_median_queue
    private
    real(dp), allocatable :: values(:),weights(:)
    integer :: n=0,k=0,watermark=0
    real(dp) :: total=0,prefix=0
  contains
    procedure :: init,clear,push,remove,pop,median,count
  end type
contains
  logical function below(a,b)
    real(dp), intent(in) :: a,b
    below=.false.
    if(ieee_is_nan(a).or.ieee_is_nan(b)) return
    below=a<b
  end function
  logical function at_least(a,b)
    real(dp), intent(in) :: a,b
    at_least=.false.
    if(ieee_is_nan(a).or.ieee_is_nan(b)) return
    at_least=a>=b
  end function
  subroutine init(self,capacity)
    class(tree_median_queue), intent(inout) :: self
    integer, intent(in) :: capacity
    if(allocated(self%values)) deallocate(self%values,self%weights)
    allocate(self%values(max(1,capacity)),self%weights(max(1,capacity)))
    self%watermark=0;call self%clear()
  end subroutine
  subroutine clear(self)
    class(tree_median_queue), intent(inout) :: self
    self%n=0;self%k=0;self%total=0;self%prefix=0
  end subroutine
  integer function count(self)
    class(tree_median_queue), intent(in) :: self
    count=self%n
  end function
  real(dp) function value_at(self,i) result(value)
    class(tree_median_queue), intent(in) :: self
    integer, intent(in) :: i
    ! The C source retains removed records and can read them during updates.
    ! Only indices outside initialized storage use a deterministic quiet NaN.
    value=ieee_value(0._dp,ieee_quiet_nan)
    if(i>=1.and.i<=self%watermark) value=self%values(i)
  end function
  real(dp) function weight_at(self,i) result(value)
    class(tree_median_queue), intent(in) :: self
    integer, intent(in) :: i
    value=ieee_value(0._dp,ieee_quiet_nan)
    if(i>=1.and.i<=self%watermark) value=self%weights(i)
  end function
  real(dp) function median(self) result(value)
    class(tree_median_queue), intent(in) :: self
    real(dp) :: half
    value=0;half=ieee_quotient(self%total,2._dp)
    if(ieee_is_nan(self%prefix).or.ieee_is_nan(half)) return
    if(self%prefix==half) then
      value=ieee_quotient(ieee_difference(value_at(self,self%k+1),-value_at(self,self%k)),2._dp)
    else if(self%prefix>half) then
      value=value_at(self,self%k)
    end if
  end function
  subroutine push(self,data,weight)
    class(tree_median_queue), intent(inout) :: self
    real(dp), intent(in) :: data,weight
    real(dp) :: original,held
    integer :: i
    original=0;if(self%n/=0) original=self%median()
    call require(self%n<size(self%values),'internal tree median queue capacity exceeded')
    self%n=self%n+1;self%watermark=max(self%watermark,self%n);i=self%n;self%values(i)=data;self%weights(i)=weight
    do while(i>1)
      if(.not.below(self%values(i),self%values(i-1))) exit
      held=self%values(i);self%values(i)=self%values(i-1);self%values(i-1)=held
      held=self%weights(i);self%weights(i)=self%weights(i-1);self%weights(i-1)=held;i=i-1
    end do
    if(self%n==1) then
      self%k=1;self%total=weight;self%prefix=weight;return
    end if
    self%total=ieee_difference(self%total,-weight)
    if(below(data,original)) then
      self%k=self%k+1;self%prefix=ieee_difference(self%prefix,-weight)
      do while(self%k>1)
        if(.not.at_least(ieee_difference(self%prefix,weight_at(self,self%k)),ieee_quotient(self%total,2._dp))) exit
        self%k=self%k-1;self%prefix=ieee_difference(self%prefix,weight_at(self,self%k+1))
      end do
    else if(at_least(data,original)) then
      do while(self%k<self%n)
        if(.not.below(self%prefix,ieee_quotient(self%total,2._dp))) exit
        self%k=self%k+1;self%prefix=ieee_difference(self%prefix,-weight_at(self,self%k))
      end do
    end if
  end subroutine
  subroutine post_remove(self,data,weight,original)
    class(tree_median_queue), intent(inout) :: self
    real(dp), intent(in) :: data,weight,original
    if(self%n==0) then
      call self%clear();return
    end if
    self%total=ieee_difference(self%total,weight)
    if(self%n==1) then
      self%k=1;self%prefix=self%total;return
    end if
    if(below(data,original)) then
      self%k=self%k-1;self%prefix=ieee_difference(self%prefix,weight)
      do while(self%k<self%n)
        if(.not.below(self%prefix,ieee_quotient(self%total,2._dp))) exit
        self%k=self%k+1;self%prefix=ieee_difference(self%prefix,-weight_at(self,self%k))
      end do
    else if(at_least(data,original)) then
      do while(self%k>1)
        if(.not.at_least(ieee_difference(self%prefix,weight_at(self,self%k)),ieee_quotient(self%total,2._dp))) exit
        self%k=self%k-1;self%prefix=ieee_difference(self%prefix,weight_at(self,self%k+1))
      end do
    end if
  end subroutine
  subroutine remove(self,data,weight)
    class(tree_median_queue), intent(inout) :: self
    real(dp), intent(in) :: data,weight
    real(dp) :: original
    integer :: i,at
    original=self%median();at=0
    do i=1,self%n
      if(self%values(i)==data.and.self%weights(i)==weight) then
        at=i;exit
      end if
    end do
    call require(at/=0,'internal tree median record missing')
    do i=at,self%n-1
      self%values(i)=self%values(i+1);self%weights(i)=self%weights(i+1)
    end do
    self%n=self%n-1;call post_remove(self,data,weight,original)
  end subroutine
  subroutine pop(self,data,weight)
    class(tree_median_queue), intent(inout) :: self
    real(dp), intent(out) :: data,weight
    call require(self%n>0,'internal tree median pop on empty queue')
    data=self%values(1);weight=self%weights(1);call self%remove(data,weight)
  end subroutine
end module
