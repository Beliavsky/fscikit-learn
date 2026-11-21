program test_inspection
  use sklearn, only: dp,permutation_importance,permutation_importance_multi,permutation_importance_result
  use, intrinsic :: iso_fortran_env, only: int64
  use sklearn_common, only: require
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_is_nan
  implicit none
  type(permutation_importance_result) :: result,again
  type(permutation_importance_result), allocatable :: multi(:)
  real(dp) :: x(12,3),weights(12)=1
  integer :: i,calls=0
  x(:,1)=[(real(i,dp),i=1,12)]; x(:,2)=3; x(:,3)=x(:,1)**2
  result=permutation_importance(x,score,n_repeats=7,random_state=42)
  call require(calls==22,'permutation importance calls baseline once plus repeats for each feature')
  call require(all(shape(result%importances)==[3,7]),'permutation importance raw layout')
  call require(result%importances_mean(1)>0.and.all(result%importances(2:,:)==0),'permutation importance isolates feature')
  again=permutation_importance(x,score,n_repeats=7,random_state=42)
  call require(all(result%importances==again%importances),'permutation importance reproduces seeded shuffles')
  multi=permutation_importance_multi(x,multiple,n_repeats=7,random_state=42)
  call require(size(multi)==2.and.all(multi(1)%importances==result%importances),'multiple scores share permutations')
  call require(all(multi(2)%importances==2*result%importances),'multiple score order is preserved')
  weights(1)=-.1_dp
  result=permutation_importance(x,score,n_repeats=1,max_samples=.5_dp,sample_weight=weights)
  call require(all(result%importances_std==0),'one repeat has zero population deviation')
  again=permutation_importance(x,score,n_repeats=1,max_samples=.5,sample_weight=weights)
  call require(all(result%importances==again%importances),'real32 sample fractions are supported')
  again=permutation_importance(x,score,n_repeats=1,max_samples=6_int64,sample_weight=weights)
  call require(all(result%importances==again%importances),'int64 sample counts are supported')
  result=permutation_importance(x(:10,:),score,n_repeats=2,random_state=42,max_samples=.7)
  again=permutation_importance(x(:10,:),score,n_repeats=2,random_state=42,max_samples=6)
  call require(all(result%importances==again%importances),'real32 fractions promote their represented value')
  result=permutation_importance(x(:10,:),score,n_repeats=2,random_state=42,max_samples=.7_dp)
  again=permutation_importance(x(:10,:),score,n_repeats=2,random_state=42,max_samples=7)
  call require(all(result%importances==again%importances),'real64 fractions retain their precision')
  x(1,2)=ieee_value(0._dp,ieee_quiet_nan)
  result=permutation_importance(x,score,n_repeats=2,max_samples=1)
  call require(ieee_is_nan(x(1,2)),'permutation importance preserves input NaNs')
  result=permutation_importance(x,score,n_repeats=2,max_samples=.001_dp)
  call require(all(result%importances==1),'fraction rounded to zero forwards empty samples to the scorer')
  print *, 'permutation importance: PASS'
contains
  function score(features,rows,sample_weight) result(value)
    real(dp), intent(in) :: features(:,:)
    integer, intent(in) :: rows(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp) :: value,errors(size(rows))
    calls=calls+1
    if(size(rows)==0) then
      value=-1; return
    end if
    errors=(features(:,1)-x(rows,1))**2
    if(present(sample_weight)) then
      value=-sum(errors*sample_weight)/sum(sample_weight)
    else
      value=-sum(errors)/size(rows)
    end if
  end function
  function multiple(features,rows,sample_weight) result(values)
    real(dp), intent(in) :: features(:,:)
    integer, intent(in) :: rows(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: values(:)
    real(dp) :: value
    value=score(features,rows,sample_weight); values=[value,2*value]
  end function
end program
