program image_invalid_driver
  use sklearn_image
  use sklearn_common, only: dp
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(patch_extractor) :: model
  real(dp) :: x(1,4,5),image(4,5),cp(2,2,3,2)
  real(dp), allocatable :: p(:,:,:),r(:,:),rc(:,:,:)
  character(40) :: arg
  call get_command_argument(1,arg)
  x=1;image=1;cp=1;model%patch_size=[2,3]
  select case(trim(arg))
  case('negative_size');model%patch_size=[-1,2]
  case('mixed_auto');model%patch_size=[0,2]
  case('negative_count');model%max_patches=-1
  case('negative_fraction');model%patch_fraction=-.1_dp
  case('unit_fraction');model%patch_fraction=1
  case('large_fraction');model%patch_fraction=2
  case('nan_fraction');model%patch_fraction=ieee_value(0._dp,ieee_quiet_nan)
  case('inf_fraction');model%patch_fraction=ieee_value(0._dp,ieee_positive_inf)
  case('both_limits');model%max_patches=2;model%patch_fraction=.5_dp
  case('negative_seed');model%random_state=-1
  case('tall');model%patch_size=[5,2]
  case('wide');model%patch_size=[2,6]
  case('auto_zero');model%patch_size=0
  case('nan_image');x(1,1,1)=ieee_value(0._dp,ieee_quiet_nan)
  case('inf_image');x(1,1,1)=ieee_value(0._dp,ieee_positive_inf)
  case('empty_batch');p=model%transform(x(:0,:,:));stop 0
  case('empty_height');p=model%transform(x(:,:0,:));stop 0
  case('empty_width');p=model%transform(x(:,:,:0));stop 0
  case('zero_extract');p=extract_patches_2d(image,[0,2]);stop 0
  case('bad_recon_image');r=reconstruct_from_patches_2d(reshape([1._dp],[1,1,1]),[0,5]);stop 0
  case('bad_recon_patch');r=reconstruct_from_patches_2d(reshape([1._dp],[1,1,1]),[1,-1]);stop 0
  case('large_recon_patch');rc=reconstruct_from_patches_2d(cp,[1,2,2]);stop 0
  case('recon_channels');rc=reconstruct_from_patches_2d(cp,[4,5,3]);stop 0
  case('zero_recon_patch');rc=reconstruct_from_patches_2d(cp(:,:0,:,:),[4,5,2]);stop 0
  case default;stop 2
  end select
  p=model%transform(x)
end program
