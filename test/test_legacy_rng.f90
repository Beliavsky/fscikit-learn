program test_legacy_rng
  use sklearn_common, only: dp,require
  use sklearn_legacy_rng, only: legacy_random_state
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  type(legacy_random_state) :: rng
  integer :: i,index
  call rng%seed(5489)
  call require(rng%raw_uint32()==3499211612_int64,'MT19937 published initial value')
  call rng%seed(0)
  call require(abs(rng%uniform()-.5488135039273248_dp)<1.e-16_dp,'RandomState initial uniform')
  call rng%seed(0)
  call require(abs(rng%normal()-1.764052345967664_dp)<1.e-14_dp,'RandomState initial Gaussian')
  call require(abs(rng%normal()-.4001572083672233_dp)<1.e-14_dp,'cached Gaussian')
  call rng%seed(0); index=rng%integer_index(1)
  call require(index==1.and.abs(rng%uniform()-.5488135039273248_dp)<1.e-16_dp,'unit bound preserves stream')
  do i=1,10000
    index=rng%integer_index(17)
    call require(index>=1.and.index<=17,'bounded integer draw')
  end do
  print *, 'Legacy RandomState tests passed'
end program
