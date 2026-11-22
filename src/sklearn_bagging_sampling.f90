! Sampling contracts from sklearn utils/_random.pyx and NumPy RandomState.
! BSD attribution retained in THIRD_PARTY_NOTICES.md and LICENSES.
module sklearn_bagging_sampling
  use sklearn_common, only: dp
  use sklearn_legacy_rng, only: legacy_random_state
  implicit none
  private
  public :: sample_indices
contains
  function sample_indices(rng,population,count,replace) result(ids)
    type(legacy_random_state), intent(inout) :: rng
    integer, intent(in) :: population,count
    logical, intent(in) :: replace
    integer :: ids(count),i,j,tmp
    integer, allocatable :: pool(:)
    real(dp) :: ratio
    if(replace) then
      do i=1,count
        ids(i)=rng%integer_index(population)
      end do
      return
    end if
    ratio=real(count,dp)/population
    if(ratio>.01_dp.and.ratio<.99_dp) then
      pool=[(i,i=1,population)]
      do i=population,2,-1
        j=rng%integer_index(i); tmp=pool(i); pool(i)=pool(j); pool(j)=tmp
      end do
      ids=pool(:count)
    else if(ratio<.2_dp) then
      do i=1,count
        do
          j=rng%integer_index(population)
          if(.not.any(ids(:i-1)==j)) exit
        end do
        ids(i)=j
      end do
    else
      ids=[(i,i=1,count)]
      do i=count+1,population
        j=rng%integer_index(i)
        if(j<=count) ids(j)=i
      end do
    end if
  end function
end module
