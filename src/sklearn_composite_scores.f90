! Internal source-compatible scoring shared by native composite estimators.
module sklearn_composite_scores
  use sklearn_common, only: dp,require
  use sklearn_ieee_math, only: ieee_total,ieee_mean,ieee_difference,ieee_quotient,ieee_product
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value,ieee_quiet_nan
  implicit none
  private
  public :: score_weights,regression_score
contains
  subroutine score_weights(w,n,context)
    real(dp), intent(in), optional :: w(:)
    integer, intent(in) :: n
    character(*), intent(in), optional :: context
    character(:), allocatable :: prefix
    real(dp) :: mass
    prefix='bagging'; if(present(context)) prefix=trim(context)
    if(.not.present(w)) return
    call require(size(w)==n,prefix//' score weight length mismatch')
    call require(all(ieee_is_finite(w)),prefix//' score weights must be finite')
    mass=ieee_total(w)
    if(.not.ieee_is_nan(mass)) call require(mass/=0,prefix//' score weights must have a nonzero sum')
  end subroutine

  real(dp) function regression_score(y,predicted,w,context) result(value)
    real(dp), intent(in) :: y(:),predicted(:)
    real(dp), intent(in), optional :: w(:)
    character(*), intent(in), optional :: context
    character(:), allocatable :: prefix
    real(dp) :: residual(size(y)),centered(size(y)),center,num,den,difference
    integer :: i
    prefix='bagging'; if(present(context)) prefix=trim(context)
    call require(size(y)==size(predicted),prefix//' score target length mismatch')
    call require(all(ieee_is_finite(y)),prefix//' score targets must be finite')
    call require(all(ieee_is_finite(predicted)),prefix//' score predictions must be finite')
    if(size(y)<2) then
      if(present(w)) then
        call require(size(w)==size(y),prefix//' score weight length mismatch')
        call require(all(ieee_is_finite(w)),prefix//' score weights must be finite')
      end if
      value=ieee_value(0._dp,ieee_quiet_nan); return
    end if
    call score_weights(w,size(y),prefix); center=ieee_mean(y,w)
    do i=1,size(y)
      difference=ieee_difference(y(i),predicted(i)); residual(i)=ieee_product(difference,difference)
      difference=ieee_difference(y(i),center); centered(i)=ieee_product(difference,difference)
    end do
    num=ieee_mean(residual,w); den=ieee_mean(centered,w)
    value=ieee_value(0._dp,ieee_quiet_nan)
    if(ieee_is_nan(num).or.ieee_is_nan(den)) return
    value=1
    if(num/=0) then
      value=0
      if(den/=0) value=ieee_difference(1._dp,ieee_quotient(num,den))
    end if
  end function

end module
