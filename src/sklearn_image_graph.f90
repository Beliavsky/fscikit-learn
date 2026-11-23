! Pixel/voxel graphs: sklearn 1.6.1 feature_extraction/image.py (BSD-3-Clause).
! Native CSR construction/storage are MIT; see THIRD_PARTY_NOTICES.md.
module sklearn_image_graph
  use sklearn_common, only: dp,require
  use sklearn_sparse, only: csr_matrix
  use sklearn_ieee_math, only: ieee_difference
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  private
  public :: grid_to_graph,img_to_graph
  interface grid_to_graph
    module procedure grid_graph_3,grid_graph_2,grid_graph_1,grid_graph_0
  end interface
  interface img_to_graph
    module procedure image_graph_3,image_graph_2,image_graph_1,image_graph_0
  end interface
contains
  integer function volume(dims) result(n)
    integer, intent(in) :: dims(3)
    integer(int64) :: total
    integer :: i
    call require(all(dims>=0),'image graph dimensions must be nonnegative')
    n=0
    if(any(dims==0)) return
    total=1
    do i=1,3
      call require(total<=huge(total)/dims(i),'image graph volume exceeds int64 range')
      total=total*dims(i)
    end do
    call require(total<huge(1),'image graph volume exceeds native index range')
    n=int(total)
  end function
  function grid_dimensions(n_x,n_y,n_z) result(dims)
    integer, intent(in) :: n_x,n_y
    integer, optional, intent(in) :: n_z
    integer :: dims(3),n
    dims=[n_x,n_y,1]
    if(present(n_z)) dims(3)=n_z
    call require(all(dims>0),'grid graph dimensions must be positive')
    n=volume(dims)
  end function
  function flatten_mask(mask) result(flat)
    logical, intent(in) :: mask(:,:,:)
    logical, allocatable :: flat(:)
    integer :: i,j,k,at,status
    allocate(flat(size(mask)),stat=status);call require(status==0,'image graph mask allocation failed')
    at=0
    do i=1,size(mask,1)
      do j=1,size(mask,2)
        do k=1,size(mask,3)
          at=at+1;flat(at)=mask(i,j,k)
        end do
      end do
    end do
  end function
  function grid_graph_3(n_x,n_y,n_z,mask) result(graph)
    integer, intent(in) :: n_x,n_y
    integer, optional, intent(in) :: n_z
    logical, optional, intent(in) :: mask(:,:,:)
    type(csr_matrix) :: graph
    integer :: dims(3)
    dims=grid_dimensions(n_x,n_y,n_z)
    if(present(mask)) then
      call require(size(mask,kind=int64)==volume(dims),'grid graph mask size mismatch')
      graph=build_graph(dims,flatten_mask(mask))
    else
      graph=build_graph(dims)
    end if
  end function
  function grid_graph_2(n_x,n_y,n_z,mask) result(graph)
    integer, intent(in) :: n_x,n_y
    integer, optional, intent(in) :: n_z
    logical, intent(in) :: mask(:,:)
    type(csr_matrix) :: graph
    integer :: dims(3)
    dims=grid_dimensions(n_x,n_y,n_z)
    call require(size(mask,kind=int64)==volume(dims),'grid graph mask size mismatch')
    graph=build_graph(dims,reshape(transpose(mask),[size(mask)]))
  end function
  function grid_graph_1(n_x,n_y,n_z,mask) result(graph)
    integer, intent(in) :: n_x,n_y
    integer, optional, intent(in) :: n_z
    logical, intent(in) :: mask(:)
    type(csr_matrix) :: graph
    integer :: dims(3)
    dims=grid_dimensions(n_x,n_y,n_z)
    call require(size(mask,kind=int64)==volume(dims),'grid graph mask size mismatch')
    graph=build_graph(dims,mask)
  end function
  function grid_graph_0(n_x,n_y,n_z,mask) result(graph)
    integer, intent(in) :: n_x,n_y
    integer, optional, intent(in) :: n_z
    logical, intent(in) :: mask
    type(csr_matrix) :: graph
    integer :: dims(3)
    dims=grid_dimensions(n_x,n_y,n_z)
    call require(volume(dims)==1,'grid graph mask size mismatch')
    graph=build_graph(dims,[mask])
  end function
  function image_graph_3(image,mask) result(graph)
    real(dp), intent(in) :: image(:,:,:)
    logical, optional, intent(in) :: mask(:,:,:)
    type(csr_matrix) :: graph
    real(dp), allocatable :: values(:)
    integer :: dims(3),i,j,k,at,n,status
    dims=shape(image);n=volume(dims)
    if(present(mask)) call require(all(shape(mask)==dims),'image graph mask shape mismatch')
    allocate(values(n),stat=status);call require(status==0,'image graph values allocation failed')
    at=0
    do i=1,dims(1)
      do j=1,dims(2)
        do k=1,dims(3)
          at=at+1;values(at)=image(i,j,k)
        end do
      end do
    end do
    if(present(mask)) then
      graph=build_graph(dims,flatten_mask(mask),values)
    else
      graph=build_graph(dims,values=values)
    end if
  end function
  function image_graph_2(image,mask) result(graph)
    real(dp), intent(in) :: image(:,:)
    logical, optional, intent(in) :: mask(:,:)
    type(csr_matrix) :: graph
    integer :: dims(3),n
    dims=[shape(image),1];n=volume(dims)
    if(present(mask)) then
      call require(all(shape(mask)==shape(image)),'image graph mask shape mismatch')
      graph=image_graph_3(reshape(image,dims),reshape(mask,dims))
    else
      graph=image_graph_3(reshape(image,dims))
    end if
  end function
  function image_graph_1(image,mask) result(graph)
    real(dp), intent(in) :: image(:)
    logical, optional, intent(in) :: mask(:)
    type(csr_matrix) :: graph
    integer :: dims(3),n
    dims=[1,size(image),1];n=volume(dims)
    if(present(mask)) then
      call require(size(mask)==size(image),'image graph mask shape mismatch')
      graph=image_graph_3(reshape(image,dims),reshape(mask,dims))
    else
      graph=image_graph_3(reshape(image,dims))
    end if
  end function
  function image_graph_0(image,mask) result(graph)
    real(dp), intent(in) :: image
    logical, optional, intent(in) :: mask
    type(csr_matrix) :: graph
    if(present(mask)) then
      graph=image_graph_3(reshape([image],[1,1,1]),reshape([mask],[1,1,1]))
    else
      graph=image_graph_3(reshape([image],[1,1,1]))
    end if
  end function
  function build_graph(dims,mask,values) result(graph)
    integer, intent(in) :: dims(3)
    logical, optional, intent(in) :: mask(:)
    real(dp), optional, intent(in) :: values(:)
    type(csr_matrix) :: graph
    integer, allocatable :: mapping(:)
    integer :: n,nodes,voxel,row,neighbors(7),degree,j,at,status,stride
    integer(int64) :: entries,edges
    n=volume(dims);nodes=n
    if(present(mask)) then
      call require(size(mask)==n,'image graph internal mask size mismatch');nodes=count(mask)
    end if
    if(present(values)) call require(size(values)==n,'image graph internal values size mismatch')
    graph%n_rows=nodes;graph%n_cols=nodes
    if(nodes==0) then
      allocate(graph%indptr(1),graph%indices(0),graph%data(0));graph%indptr=1;return
    end if
    if(.not.present(mask)) then
      edges=int(dims(1)-1,int64)*dims(2)*dims(3)+int(dims(1),int64)*(dims(2)-1)*dims(3)+ &
        int(dims(1),int64)*dims(2)*(dims(3)-1)
      call require(2*edges+n<huge(1),'image graph stored count exceeds native index range')
    end if
    allocate(mapping(n),graph%indptr(nodes+1),stat=status)
    call require(status==0,'image graph index allocation failed');row=0
    do voxel=1,n
      mapping(voxel)=0
      if(present(mask)) then
        if(.not.mask(voxel)) cycle
      end if
      row=row+1;mapping(voxel)=row
    end do
    stride=dims(2)*dims(3);entries=0;graph%indptr(1)=1
    do voxel=1,n
      row=mapping(voxel);if(row==0) cycle
      call row_neighbors(voxel,neighbors,degree)
      entries=entries+degree
      call require(entries<huge(1),'image graph stored count exceeds native index range')
      graph%indptr(row+1)=int(entries)+1
    end do
    allocate(graph%indices(entries),graph%data(entries),stat=status)
    call require(status==0,'image graph values allocation failed')
    do voxel=1,n
      row=mapping(voxel);if(row==0) cycle
      call row_neighbors(voxel,neighbors,degree);at=graph%indptr(row)
      do j=1,degree
        graph%indices(at)=mapping(neighbors(j));graph%data(at)=1
        if(present(values)) then
          if(neighbors(j)==voxel) then
            graph%data(at)=values(voxel)
          else
            graph%data(at)=abs(ieee_difference(values(voxel),values(neighbors(j))))
          end if
        end if
        at=at+1
      end do
    end do
    call graph%validate()
  contains
    subroutine append(candidate,list,used)
      integer, intent(in) :: candidate
      integer, intent(inout) :: list(7),used
      if(mapping(candidate)==0) return
      used=used+1;list(used)=candidate
    end subroutine
    subroutine row_neighbors(index,list,used)
      integer, intent(in) :: index
      integer, intent(out) :: list(7),used
      integer :: x,y,z
      x=(index-1)/stride;y=modulo((index-1)/dims(3),dims(2));z=modulo(index-1,dims(3));used=0
      ! C-order voxel numbers make this axis order sorted within each CSR row.
      if(x>0) call append(index-stride,list,used)
      if(y>0) call append(index-dims(3),list,used)
      if(z>0) call append(index-1,list,used)
      call append(index,list,used)
      if(z<dims(3)-1) call append(index+1,list,used)
      if(y<dims(2)-1) call append(index+dims(3),list,used)
      if(x<dims(1)-1) call append(index+stride,list,used)
    end subroutine
  end function
end module
