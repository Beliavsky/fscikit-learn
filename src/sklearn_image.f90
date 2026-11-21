! Image patch algorithms adapted from sklearn 1.6.1 feature_extraction/image.py.
! BSD-3-Clause attribution is retained in THIRD_PARTY_NOTICES.md.
module sklearn_image
  use sklearn_common, only: dp,require
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_ieee_math, only: ieee_difference,ieee_quotient
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: patch_extractor,extract_patches_2d,reconstruct_from_patches_2d
  type :: patch_extractor
    integer :: patch_size(2)=0,max_patches=0,random_state=0
    real(dp) :: patch_fraction=0
  contains
    procedure, private :: fit_gray,fit_color,transform_gray,transform_color,fit_transform_gray,fit_transform_color
    generic :: fit=>fit_gray,fit_color
    generic :: transform=>transform_gray,transform_color
    generic :: fit_transform=>fit_transform_gray,fit_transform_color
  end type
  interface extract_patches_2d
    module procedure extract_gray,extract_color
  end interface
  interface reconstruct_from_patches_2d
    module procedure reconstruct_gray,reconstruct_color
  end interface
contains
  subroutine controls(self)
    class(patch_extractor), intent(in) :: self
    call require(all(self%patch_size==0).or.all(self%patch_size>0),'patch size must be positive or both zero')
    call require(self%max_patches>=0,'max_patches must be nonnegative (zero means all)')
    call require(ieee_is_finite(self%patch_fraction),'patch_fraction must be finite')
    call require(self%patch_fraction>=0.and.self%patch_fraction<1,'patch_fraction must be in [0,1)')
    call require(self%max_patches==0.or.self%patch_fraction==0,'choose max_patches or patch_fraction')
    call require(self%random_state>=0,'patch random_state must be nonnegative')
  end subroutine
  subroutine fit_gray(self,x)
    class(patch_extractor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:,:)
    call controls(self)
  end subroutine
  subroutine fit_color(self,x)
    class(patch_extractor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:,:,:)
    call controls(self)
  end subroutine
  function fit_transform_gray(self,x) result(patches)
    class(patch_extractor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:,:)
    real(dp), allocatable :: patches(:,:,:)
    call self%fit(x);patches=self%transform(x)
  end function
  function fit_transform_color(self,x) result(patches)
    class(patch_extractor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:,:,:)
    real(dp), allocatable :: patches(:,:,:,:)
    call self%fit(x);patches=self%transform(x)
  end function
  function transform_gray(self,x) result(patches)
    class(patch_extractor), intent(in) :: self
    real(dp), intent(in) :: x(:,:,:)
    real(dp), allocatable :: patches(:,:,:),colored(:,:,:,:)
    colored=self%transform(reshape(x,[size(x,1),size(x,2),size(x,3),1]))
    patches=colored(:,:,:,1)
  end function
  function transform_color(self,x) result(patches)
    class(patch_extractor), intent(in) :: self
    real(dp), intent(in) :: x(:,:,:,:)
    real(dp), allocatable :: patches(:,:,:,:)
    integer, allocatable :: rows(:),cols(:)
    integer :: ph,pw,nh,nw,n,m,i,j,k,pos,img
    integer(int64) :: total,capacity
    logical :: sampled
    type(legacy_random_state) :: rng
    call controls(self)
    call require(all(shape(x)>0),'patch image dimensions must be nonempty')
    call require(all(ieee_is_finite(x)),'patch images must be finite')
    ph=self%patch_size(1);pw=self%patch_size(2)
    if(all(self%patch_size==0)) then
      ph=size(x,2)/10;pw=size(x,3)/10
    end if
    call require(ph>0.and.pw>0,'automatic patch size is zero; specify positive patch_size')
    call require(ph<=size(x,2).and.pw<=size(x,3),'patch size exceeds image dimensions')
    nh=size(x,2)-ph+1;nw=size(x,3)-pw+1
    total=int(nh,int64)*nw
    call require(total<=huge(n),'patch count exceeds integer capacity')
    n=int(total)
    if(self%max_patches>0) n=min(n,self%max_patches)
    if(self%patch_fraction>0) n=int(self%patch_fraction*real(total,dp))
    capacity=int(n,int64)*size(x,1)
    call require(capacity<=huge(n),'batch patch count exceeds integer capacity')
    m=int(capacity)
    allocate(patches(m,ph,pw,size(x,4)))
    sampled=self%max_patches>0.or.self%patch_fraction>0
    call rng%seed(self%random_state)
    allocate(rows(n),cols(n));pos=0
    do img=1,size(x,1)
      if(sampled) then
        ! NumPy draws every row first, then every column, with replacement.
        do k=1,n
          rows(k)=rng%integer_index(nh)
        end do
        do k=1,n
          cols(k)=rng%integer_index(nw)
        end do
      else
        k=0
        do i=1,nh
          do j=1,nw
            k=k+1;rows(k)=i;cols(k)=j
          end do
        end do
      end if
      do k=1,n
        i=rows(k);j=cols(k);pos=pos+1
        patches(pos,:,:,:)=x(img,i:i+ph-1,j:j+pw-1,:)
      end do
    end do
  end function
  function extract_gray(image,patch_size,max_patches,patch_fraction,random_state) result(patches)
    real(dp), intent(in) :: image(:,:)
    integer, intent(in) :: patch_size(2)
    integer, optional, intent(in) :: max_patches,random_state
    real(dp), optional, intent(in) :: patch_fraction
    real(dp), allocatable :: patches(:,:,:),colored(:,:,:,:)
    colored=extract_color(reshape(image,[size(image,1),size(image,2),1]), &
      patch_size,max_patches,patch_fraction,random_state)
    patches=colored(:,:,:,1)
  end function
  function extract_color(image,patch_size,max_patches,patch_fraction,random_state) result(patches)
    real(dp), intent(in) :: image(:,:,:)
    integer, intent(in) :: patch_size(2)
    integer, optional, intent(in) :: max_patches,random_state
    real(dp), optional, intent(in) :: patch_fraction
    real(dp), allocatable :: patches(:,:,:,:)
    type(patch_extractor) :: model
    call require(all(patch_size>0),'extract_patches_2d patch size must be positive')
    model%patch_size=patch_size
    if(present(max_patches)) model%max_patches=max_patches
    if(present(patch_fraction)) model%patch_fraction=patch_fraction
    if(present(random_state)) model%random_state=random_state
    patches=model%transform(reshape(image,[1,size(image,1),size(image,2),size(image,3)]))
  end function
  function reconstruct_gray(patches,image_size) result(image)
    real(dp), intent(in) :: patches(:,:,:)
    integer, intent(in) :: image_size(2)
    real(dp), allocatable :: image(:,:),colored(:,:,:)
    colored=reconstruct_color(reshape(patches,[size(patches,1),size(patches,2),size(patches,3),1]), &
      [image_size,1])
    image=colored(:,:,1)
  end function
  function reconstruct_color(patches,image_size) result(image)
    real(dp), intent(in) :: patches(:,:,:,:)
    integer, intent(in) :: image_size(3)
    real(dp), allocatable :: image(:,:,:)
    integer :: ph,pw,ih,iw,i,j,k,a,b,c
    call require(all(image_size>0),'reconstruction image dimensions must be positive')
    ph=size(patches,2);pw=size(patches,3);ih=image_size(1);iw=image_size(2)
    call require(ph>0.and.pw>0.and.ph<=ih.and.pw<=iw,'invalid reconstruction patch size')
    call require(size(patches,4)==image_size(3),'reconstruction channel count mismatch')
    allocate(image(ih,iw,image_size(3)));image=0;k=0
    do i=1,ih-ph+1
      do j=1,iw-pw+1
        k=k+1
        if(k>size(patches,1)) exit
        do c=1,image_size(3)
          do b=1,pw
            do a=1,ph
              image(i+a-1,j+b-1,c)=ieee_difference(image(i+a-1,j+b-1,c),-patches(k,a,b,c))
            end do
          end do
        end do
      end do
      if(k>size(patches,1)) exit
    end do
    ! Preserve sklearn's denominator, including incomplete and oversized patches.
    do i=1,ih
      do j=1,iw
        do c=1,image_size(3)
          image(i,j,c)=ieee_quotient(image(i,j,c),real(min(i,ph,ih-i+1),dp)*real(min(j,pw,iw-j+1),dp))
        end do
      end do
    end do
  end function
end module
