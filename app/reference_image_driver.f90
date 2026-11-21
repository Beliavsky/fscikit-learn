program reference_image_driver
  use sklearn_image
  use sklearn_common, only: dp
  implicit none
  type(patch_extractor) :: model
  real(dp), allocatable :: x(:,:,:,:),p(:,:,:,:),gray(:,:,:),rec(:,:,:),one(:,:,:,:),rgray(:,:)
  integer :: n,h,w,c,ph,pw,limit,seed,kind,i,j,k,used
  real(dp) :: fraction
  read(*,*) n,h,w,c,ph,pw,limit,fraction,seed,kind,used
  allocate(x(n,h,w,c))
  read(*,*) x
  model%patch_size=[ph,pw];model%max_patches=limit;model%patch_fraction=fraction;model%random_state=seed
  if(kind==0) then
    gray=model%fit_transform(x(:,:,:,1))
    p=reshape(gray,[size(gray,1),size(gray,2),size(gray,3),1])
  else if(kind==1) then
    p=model%fit_transform(x)
  else if(kind==2) then
    gray=extract_patches_2d(x(1,:,:,1),[ph,pw],limit,fraction,seed)
    p=reshape(gray,[size(gray,1),size(gray,2),size(gray,3),1])
  else
    p=extract_patches_2d(x(1,:,:,:),[ph,pw],limit,fraction,seed)
  end if
  print *,shape(p)
  print '(*(es25.16e3,1x))',p
  if(used<0) used=size(p,1)
  used=min(used,size(p,1))
  if(kind==0.or.kind==2) then
    rgray=reconstruct_from_patches_2d(p(:used,:,:,1),[h,w])
    rec=reshape(rgray,[h,w,1])
  else
    rec=reconstruct_from_patches_2d(p(:used,:,:,:),[h,w,c])
  end if
  print '(*(es25.16e3,1x))',rec
  ! Repeat calls exercise stateless seeded transforms; singleton channels retain rank.
  if(kind==0) then
    gray=model%transform(x(:,:,:,1));one=reshape(gray,[shape(gray),1])
  else if(kind==1) then
    one=model%transform(x)
  else
    one=p
  end if
  print '(*(es25.16e3,1x))',one
end program
