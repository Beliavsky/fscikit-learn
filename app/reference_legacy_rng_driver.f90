program reference_legacy_rng_driver
  use sklearn_common, only: dp
  use sklearn_legacy_rng, only: legacy_random_state,seed_legacy_uint32
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  type(legacy_random_state) :: rng
  integer(int64) :: seed
  integer :: output,i
  real(dp) :: values(2048),indices(1000),normals(1000),uniforms(1000)
  character(512) :: path
  call get_command_argument(1,path); read(path,*) seed
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  call seed_legacy_uint32(rng,seed)
  do i=1,2048
    values(i)=real(rng%raw_uint32(),dp)
  end do
  call emit('raw',values); call seed_legacy_uint32(rng,seed)
  do i=1,2048
    values(i)=rng%uniform()
  end do
  call emit('uniform',values); call seed_legacy_uint32(rng,seed)
  do i=1,2048
    values(i)=rng%normal()
  end do
  call emit('normal',values); call seed_legacy_uint32(rng,seed)
  do i=1,2048
    values(i)=real(rng%integer_index(17),dp)
  end do
  call emit('index',values); call seed_legacy_uint32(rng,seed)
  do i=1,1000
    indices(i)=real(rng%integer_index(mod(i,57)+1),dp)
    normals(i)=rng%normal(); uniforms(i)=rng%uniform()
  end do
  call emit('mixed_index',indices); call emit('mixed_normal',normals); call emit('mixed_uniform',uniforms)
  close(output)
contains
  subroutine emit(name,values)
    character(*), intent(in) :: name
    real(dp), intent(in) :: values(:)
    write(output,'(a,*(1x,es25.16e3))') name,values
  end subroutine
end program
