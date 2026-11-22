program test_truncated_sparse
  use sklearn_common, only: dp,require
  use sklearn_decomposition, only: truncated_svd
  use sklearn_sparse, only: csr_matrix,csr_from_dense,csr_from_arrays
  implicit none
  type(truncated_svd) :: dense,sparse,legacy
  type(csr_matrix) :: xcsr,query,duplicates
  real(dp) :: x(30,12)
  real(dp), allocatable :: z(:,:),other(:,:),before(:),back(:,:)
  integer :: i,j
  do j=1,12
    do i=1,30
      x(i,j)=0
      if(mod(i*j,7)<2) x(i,j)=sin(.13_dp*i)+cos(.31_dp*j)
    end do
  end do
  xcsr=csr_from_dense(x);query=csr_from_dense(x(:4,:));before=xcsr%data
  dense%algorithm='randomized';dense%n_components=3;dense%random_state=42
  z=dense%fit_transform(x);sparse=dense;other=sparse%fit_transform(xcsr)
  call require(maxval(abs(z-other))<1.e-9_dp,'dense and CSR randomized SVD agree')
  call require(all(xcsr%data==before),'SVD preserves sparse input values')
  call require(maxval(abs(dense%singular_values_-sparse%singular_values_))<1.e-9_dp,'CSR SVD singular spectrum')
  call require(maxval(abs(dense%explained_variance_ratio_-sparse%explained_variance_ratio_))<1.e-10_dp, &
    'CSR SVD accounts for implicit-zero variance')
  other=sparse%transform(query);call require(maxval(abs(other-z(:4,:)))<1.e-9_dp,'CSR SVD query transform')
  back=sparse%inverse_transform(other);call require(all(shape(back)==[4,12]),'sparse SVD inverse is dense')
  duplicates=csr_from_arrays(4,3,[1,4,6,8,10],[3,1,3,2,2,1,3,2,1], &
    [1._dp,2._dp,-.5_dp,1._dp,2._dp,1._dp,3._dp,0._dp,2._dp])
  dense%n_components=3;sparse=dense;z=dense%fit_transform(duplicates%to_dense());other=sparse%fit_transform(duplicates)
  call require(maxval(abs(dense%explained_variance_ratio_-sparse%explained_variance_ratio_))<1.e-10_dp, &
    'CSR SVD aggregates duplicates before feature variance')
  call require(maxval(abs(z-other))<1.e-9_dp,'CSR SVD unsorted duplicate algebra')
  dense%n_components=3;z=dense%fit_transform(x(:2,:))
  call require(dense%n_components_==2.and.size(z,2)==2,'randomized SVD clips requested rank to sample count')
  legacy%n_components=1;z=legacy%fit_transform(x(:,:1))
  call require(size(z,2)==1,'legacy exact single-feature SVD remains supported')
  x=0;xcsr=csr_from_dense(x);sparse%n_components=3;z=sparse%fit_transform(xcsr)
  call require(all(z==0).and.all(sparse%explained_variance_ratio_==0),'zero CSR matrix SVD')
  call require(maxval(abs(matmul(sparse%components_,transpose(sparse%components_))-eye3()))<1.e-12_dp, &
    'CSR SVD completes orthogonal null components')
  print *, 'Dense and sparse randomized TruncatedSVD: PASS'
contains
  function eye3() result(eye)
    real(dp) :: eye(3,3)
    integer :: k
    eye=0;do k=1,3
      eye(k,k)=1
    end do
  end function
end program
