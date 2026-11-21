program image_patches
  use sklearn, only: dp,extract_patches_2d,reconstruct_from_patches_2d
  implicit none
  real(dp) :: image(6,7)
  real(dp), allocatable :: patches(:,:,:),reconstructed(:,:)
  integer :: i,j
  do i=1,6
    do j=1,7
      image(i,j)=sin(real(i+j,dp))
    end do
  end do
  patches=extract_patches_2d(image,[2,3])
  reconstructed=reconstruct_from_patches_2d(patches,[6,7])
  print *, 'Extracted patches:',size(patches,1)
  print *, 'Maximum reconstruction error:',maxval(abs(image-reconstructed))
end program
