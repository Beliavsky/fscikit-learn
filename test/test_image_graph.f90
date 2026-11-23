program test_image_graph
  use sklearn_image_graph, only: grid_to_graph,img_to_graph
  use sklearn_sparse, only: csr_matrix
  use sklearn_common, only: dp,require
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf,ieee_is_nan,ieee_is_finite
  implicit none
  type(csr_matrix) :: graph
  real(dp) :: image(2,3),cube(2,2,2),line(5),scalar
  real(dp), allocatable :: dense(:,:),empty(:,:),product(:,:),ones(:,:)
  logical :: mask(2,3),mask3(2,2,2)
  integer :: i,j,k
  graph=grid_to_graph(2,3)
  call require(graph%n_rows==6.and.graph%nnz()==20,'grid graph dimensions and stored count')
  call require(all(graph%indices(:3)==[1,2,4]).and.all(graph%indices(18:)==[3,5,6]),'C-order sorted grid indices')
  image(1,:)=[1._dp,2._dp,4._dp];image(2,:)=[8._dp,16._dp,32._dp]
  graph=img_to_graph(image)
  call require(all(graph%data(:7)==[1._dp,1._dp,7._dp,1._dp,2._dp,2._dp,14._dp]),'pixel gradients and diagonal')
  mask=.false.;mask(1,1)=.true.;mask(1,3)=.true.;mask(2,1)=.true.
  graph=img_to_graph(image,mask)
  call require(graph%n_rows==3.and.graph%nnz()==5,'masked image graph dimensions')
  call require(all(graph%indices==[1,3,2,1,3]).and.all(graph%data==[1._dp,7._dp,4._dp,7._dp,8._dp]), &
    'masked graph compression and isolated voxel')
  graph=grid_to_graph(2,3,mask=mask)
  call require(graph%nnz()==5.and.all(graph%data==1),'two-dimensional grid mask overload')
  do i=1,2
    do j=1,2
      do k=1,2
        cube(i,j,k)=100*i+10*j+k
      end do
    end do
  end do
  graph=img_to_graph(cube)
  call require(graph%nnz()==32.and.all(graph%indices(:4)==[1,2,3,5]),'six-connected voxel graph')
  mask3=.false.;mask3(1,1,1)=.true.;mask3(2,2,2)=.true.
  graph=img_to_graph(cube,mask3)
  call require(graph%nnz()==2.and.all(graph%data==[111._dp,222._dp]),'disconnected diagonal-only masked graph')
  graph=grid_to_graph(2,2,2,mask3)
  call require(all(graph%data==1),'three-dimensional grid mask overload')
  graph=grid_to_graph(2,3,mask=[.false.,.false.,.false.,.false.,.false.,.false.])
  call require(graph%n_rows==0.and.graph%nnz()==0.and.all(graph%indptr==1),'empty selection remains a zero-node CSR')
  allocate(empty(0,2));graph=img_to_graph(empty)
  call require(graph%n_rows==0.and.size(graph%indptr)==1,'zero-volume image graph')
  graph=img_to_graph(reshape([0._dp,0._dp,0._dp,1._dp],[2,2]))
  call require(graph%nnz()==12.and.count(graph%data==0)==7,'explicit zero gradients and diagonal retained')
  scalar=ieee_value(0._dp,ieee_quiet_nan);graph=img_to_graph(scalar)
  call require(graph%nnz()==1.and.ieee_is_nan(graph%data(1)),'scalar nonfinite image diagonal')
  line=[huge(1._dp),-huge(1._dp),ieee_value(0._dp,ieee_positive_inf), &
    ieee_value(0._dp,ieee_positive_inf),scalar]
  graph=img_to_graph(line)
  call require(graph%nnz()==13.and.any(ieee_is_nan(graph%data)).and.any(.not.ieee_is_finite(graph%data)), &
    'quiet IEEE gradients without arithmetic traps')
  graph=grid_to_graph(200,300)
  call require(graph%n_rows==60000.and.graph%nnz()==299000,'large sparse grid without dense adjacency allocation')
  allocate(ones(60000,1));ones=1;product=graph%matmul(ones)
  call require(count(product(:,1)==3)==4.and.count(product(:,1)==4)==992.and.count(product(:,1)==5)==59004, &
    'large sparse grid boundary and interior degrees')
  print *, 'Image and grid graphs: PASS'
end program
