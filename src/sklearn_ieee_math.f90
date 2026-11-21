! Internal quiet-IEEE primitives shared by weighted estimators and reductions.
module sklearn_ieee_math
  use sklearn_common, only: dp
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_is_negative,ieee_value, &
    ieee_quiet_nan,ieee_positive_inf,ieee_negative_inf
  implicit none
  private
  public :: ieee_difference,ieee_product,ieee_quotient,ieee_total,ieee_mean,ieee_argmax
contains
  real(dp) function ieee_difference(a,b) result(value)
    real(dp), intent(in) :: a,b
    value=ieee_value(0._dp,ieee_quiet_nan)
    if(ieee_is_nan(a).or.ieee_is_nan(b)) return
    if(.not.ieee_is_finite(a).or..not.ieee_is_finite(b)) then
      if(a==b) return
      if(a>0.and..not.ieee_is_finite(a).or.b<0.and..not.ieee_is_finite(b)) then
        value=ieee_value(0._dp,ieee_positive_inf)
      else
        value=ieee_value(0._dp,ieee_negative_inf)
      end if
    else
      if(b<0) then
        if(a>huge(a)+b) then
          value=ieee_value(0._dp,ieee_positive_inf); return
        end if
      else if(b>0) then
        if(a< -huge(a)+b) then
          value=ieee_value(0._dp,ieee_negative_inf); return
        end if
      end if
      value=a-b
    end if
  end function
  real(dp) function ieee_product(a,b) result(value)
    real(dp), intent(in) :: a,b
    value=ieee_value(0._dp,ieee_quiet_nan)
    if(ieee_is_nan(a).or.ieee_is_nan(b)) return
    if(.not.ieee_is_finite(a).or..not.ieee_is_finite(b)) then
      if(a==0.or.b==0) return
      value=ieee_value(0._dp,ieee_positive_inf)
      if(ieee_is_negative(a).neqv.ieee_is_negative(b)) value=-value
    else
      if(abs(a)>1) then
        if(abs(b)>huge(a)/abs(a)) then
          value=ieee_value(0._dp,ieee_positive_inf)
          if(ieee_is_negative(a).neqv.ieee_is_negative(b)) value=-value
          return
        end if
      end if
      value=a*b
    end if
  end function

  real(dp) function ieee_quotient(a,b) result(value)
    real(dp), intent(in) :: a,b
    value=ieee_value(0._dp,ieee_quiet_nan)
    if(ieee_is_nan(a).or.ieee_is_nan(b)) return
    if(.not.ieee_is_finite(a).and..not.ieee_is_finite(b)) return
    if(b==0) then
      if(a==0) return
      value=ieee_value(0._dp,ieee_positive_inf)
      if(ieee_is_negative(a).neqv.ieee_is_negative(b)) value=-value
      return
    end if
    if(abs(b)<1) then
      if(abs(a)>huge(a)*abs(b)) then
        value=ieee_value(0._dp,ieee_positive_inf)
        if(ieee_is_negative(a).neqv.ieee_is_negative(b)) value=-value
        return
      end if
    end if
    value=a/b
  end function
  real(dp) function ieee_total(values) result(value)
    real(dp), intent(in) :: values(:)
    integer :: j
    value=0
    do j=1,size(values)
      value=ieee_difference(value,-values(j))
    end do
  end function
  real(dp) function ieee_mean(values,weights) result(value)
    real(dp), intent(in) :: values(:)
    real(dp), intent(in), optional :: weights(:)
    integer :: j
    value=0
    if(present(weights)) then
      do j=1,size(values)
        value=ieee_difference(value,-ieee_product(values(j),weights(j)))
      end do
      value=ieee_quotient(value,ieee_total(weights))
    else
      value=ieee_quotient(ieee_total(values),real(size(values),dp))
    end if
  end function
  integer function ieee_argmax(values) result(index_)
    real(dp), intent(in) :: values(:)
    integer :: j
    index_=1
    do j=2,size(values)
      if(ieee_is_nan(values(index_))) return
      if(ieee_is_nan(values(j))) then
        index_=j; return
      end if
      if(values(j)>values(index_)) index_=j
    end do
  end function
end module
