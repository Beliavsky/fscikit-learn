program reference_image_graph_driver
  use sklearn_image_graph, only: grid_to_graph,img_to_graph
  use sklearn_sparse, only: csr_matrix
  use sklearn_common, only: dp,require
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan,ieee_is_negative
  implicit none
  type(csr_matrix) :: graph
  real(dp), allocatable :: image(:,:,:),before(:,:,:),line(:),right(:,:),dense(:,:)
  logical, allocatable :: mask(:,:,:),mask_before(:,:,:),flat(:),plane(:,:)
  integer, allocatable :: bits(:,:,:)
  integer(int64), allocatable :: signs(:)
  integer :: mode,rank,nx,ny,nz,masked,mask_rank,summary,i,j,k,at
  character(40) :: argument
  call get_command_argument(1,argument)
  if(len_trim(argument)>0) then
    call reject(trim(argument));error stop 'invalid graph input accepted'
  end if
  read(*,*) mode,rank,nx,ny,nz,masked,mask_rank,summary
  allocate(image(nx,ny,nz),mask(nx,ny,nz),bits(nx,ny,nz))
  read(*,*) image;read(*,*) bits;mask=bits/=0;before=image;mask_before=mask
  if(mode==0) then
    if(masked==0) then
      if(nz==1) then
        graph=grid_to_graph(nx,ny)
      else
        graph=grid_to_graph(nx,ny,nz)
      end if
    else
      allocate(flat(size(mask)),plane(nx,ny*nz));at=0
      do i=1,nx
        do j=1,ny
          do k=1,nz
            at=at+1;flat(at)=mask(i,j,k);plane(i,(j-1)*nz+k)=mask(i,j,k)
          end do
        end do
      end do
      select case(mask_rank)
      case(0);graph=grid_to_graph(nx,ny,nz,mask(1,1,1))
      case(1);graph=grid_to_graph(nx,ny,nz,flat)
      case(2)
        if(nz==1) then
          graph=grid_to_graph(nx,ny,mask=plane)
        else
          graph=grid_to_graph(nx,ny,nz,plane)
        end if
      case(3);graph=grid_to_graph(nx,ny,nz,mask)
      case default;error stop 'unknown grid mask rank'
      end select
    end if
  else
    select case(rank)
    case(0)
      if(masked==0) then
        graph=img_to_graph(image(1,1,1))
      else
        graph=img_to_graph(image(1,1,1),mask(1,1,1))
      end if
    case(1)
      if(masked==0) then
        graph=img_to_graph(image(1,:,1))
      else
        graph=img_to_graph(image(1,:,1),mask(1,:,1))
      end if
    case(2)
      if(masked==0) then
        graph=img_to_graph(image(:,:,1))
      else
        graph=img_to_graph(image(:,:,1),mask(:,:,1))
      end if
    case(3)
      if(masked==0) then
        graph=img_to_graph(image)
      else
        graph=img_to_graph(image,mask)
      end if
    case default;error stop 'unknown image rank'
    end select
  end if
  call graph%validate()
  call integers(reshape(int([graph%n_rows,graph%n_cols,graph%nnz()],int64),[3,1]))
  call integers(reshape(int(graph%indptr,int64),[size(graph%indptr),1]))
  call integers(reshape(int(graph%indices,int64),[size(graph%indices),1]))
  call matrix(reshape(graph%data,[size(graph%data),1]))
  allocate(signs(size(graph%data)));signs=0
  do i=1,size(signs)
    if(.not.ieee_is_nan(graph%data(i))) signs(i)=merge(1_int64,0_int64,ieee_is_negative(graph%data(i)))
  end do
  call integers(reshape(signs,[size(signs),1]))
  if(summary==0) then
    call matrix(graph%to_dense())
    call integers(int(merge(1,0,graph%stored_mask()),int64))
  else
    allocate(right(graph%n_cols,2))
    do i=1,graph%n_cols
      right(i,:)=[1._dp,real(modulo(i,17)-8,dp)]
    end do
    call matrix(graph%matmul(right))
  end if
  call require(all(ieee_is_nan(image).eqv.ieee_is_nan(before)),'image graph input NaN mask changed')
  where(ieee_is_nan(before))
    image=0;before=0
  end where
  call require(all(image==before).and.all(mask.eqv.mask_before),'image graph input changed')
contains
  subroutine matrix(a)
    real(dp), intent(in) :: a(:,:)
    print *, 'F',shape(a,kind=int64)
    print '(8(es25.16e3,1x))',a
  end subroutine
  subroutine integers(a)
    integer(int64), intent(in) :: a(:,:)
    print *, 'I',shape(a,kind=int64)
    print '(20(i0,1x))',a
  end subroutine
  subroutine reject(name)
    character(*), intent(in) :: name
    type(csr_matrix) :: result
    real(dp) :: data(2,3),cube(2,3,2)
    data=1;cube=1
    select case(name)
    case('zero_x');result=grid_to_graph(0,2)
    case('negative_y');result=grid_to_graph(2,-1)
    case('zero_z');result=grid_to_graph(2,2,0)
    case('volume_overflow');result=grid_to_graph(huge(1),huge(1),huge(1))
    case('index_range');result=grid_to_graph(huge(1),1)
    case('stored_range');result=grid_to_graph(1000000000,1)
    case('grid_mask_scalar');result=grid_to_graph(2,1,mask=.true.)
    case('grid_mask_1');result=grid_to_graph(2,3,mask=[.true.,.false.])
    case('grid_mask_2');result=grid_to_graph(2,3,mask=reshape([.true.,.true.,.false.,.true.],[2,2]))
    case('grid_mask_3');result=grid_to_graph(2,3,mask=reshape([.true.,.true.],[1,1,2]))
    case('image_mask_1');result=img_to_graph(data(1,:),[.true.,.true.])
    case('image_mask_2');result=img_to_graph(data,reshape([.true.,.true.,.true.,.true.,.true.,.true.],[3,2]))
    case('image_mask_3');result=img_to_graph(cube,reshape([(.true.,i=1,12)],[3,2,2]))
    case default;error stop 'unknown invalid graph case'
    end select
  end subroutine
end program
