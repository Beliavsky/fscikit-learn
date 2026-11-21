program test_image
  use sklearn, only: dp,patch_extractor,extract_patches_2d,reconstruct_from_patches_2d
  use sklearn_common, only: require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  type(patch_extractor) :: model,copy
  real(dp) :: image(4,5),batch(2,4,5),color(4,5,3)
  real(dp), allocatable :: patches(:,:,:),again(:,:,:),out(:,:),cp(:,:,:,:),back(:,:,:)
  integer :: i,j,c
  do i=1,4
    do j=1,5
      image(i,j)=10*i+j
    end do
  end do
  patches=extract_patches_2d(image,[2,3])
  call require(all(shape(patches)==[9,2,3]),'image patch shape')
  call require(all(patches(2,:,:)==image(1:2,2:4)),'row-major patch ordering')
  out=reconstruct_from_patches_2d(patches,[4,5])
  call require(maxval(abs(out-image))<1.e-13_dp,'small patch image reconstruction')
  patches(1,1,1)=-1
  call require(image(1,1)==11,'patch allocation independent of input')
  batch(1,:,:)=image;batch(2,:,:)=image+100
  model%patch_size=[2,3];model%max_patches=4;model%random_state=42
  patches=model%fit_transform(batch);again=model%transform(batch)
  call require(all(patches==again),'integer random_state resets each transform')
  call require(all(shape(patches)==[8,2,3]),'batch sample count')
  copy=model;copy%max_patches=0;copy%patch_fraction=.01_dp
  again=copy%transform(batch)
  call require(size(again,1)==0,'fraction may round to zero patches')
  do c=1,3
    color(:,:,c)=image*c
  end do
  cp=extract_patches_2d(color,[2,3]);back=reconstruct_from_patches_2d(cp,[4,5,3])
  call require(maxval(abs(back-color))<1.e-13_dp,'multichannel reconstruction')
  patches=extract_patches_2d(image,[4,5]);out=reconstruct_from_patches_2d(patches,[4,5])
  call require(out(2,3)==image(2,3)/6,'source denominator for oversized patches')
  patches=extract_patches_2d(image*0+1.e308_dp,[2,3])
  out=reconstruct_from_patches_2d(patches,[4,5])
  call require(.not.ieee_is_finite(out(2,3)),'overflow reconstruction retains source infinity without trap')
  print *, 'image patch extraction and reconstruction: PASS'
end program
