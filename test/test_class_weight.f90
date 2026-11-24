program test_class_weight
  use sklearn_class_weight, only: compute_class_weight,compute_sample_weight,class_weight_map
  use sklearn_common, only: dp,require
  use sklearn_sparse, only: csr_matrix,csr_from_dense,csr_from_arrays
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_positive_inf,ieee_is_nan
  implicit none
  integer :: y(6)=[1,1,1,1,0,0],targets(6,2),i
  real(dp), allocatable :: w(:),v(:)
  type(class_weight_map) :: map,maps(2)
  type(csr_matrix) :: sparse
  w=compute_class_weight([0,1],y,'balanced')
  call require(maxval(abs(w-[1.5_dp,.75_dp]))<1.e-14_dp,'Balanced class frequencies')
  v=compute_class_weight([1,1,0],y,'balanced')
  call require(maxval(abs(v-[.75_dp,.75_dp,1.5_dp]))<1.e-14_dp,'Unsorted repeated class selectors')
  w=compute_sample_weight(y,'balanced')
  call require(abs(sum(w)-6)<1.e-14_dp.and.abs(sum(pack(w,y==0))-3)<1.e-14_dp, &
    'Balanced sample weights equalize class mass')
  map=class_weight_map([1],[2._dp]);w=compute_sample_weight(y,class_weight=map)
  call require(all(w==[2._dp,2._dp,2._dp,2._dp,1._dp,1._dp]),'Partial class map defaults missing labels to one')
  w=compute_sample_weight(y,'balanced',indices=[-1,-2])
  call require(all(w==[0._dp,0._dp,0._dp,0._dp,1._dp,1._dp]),'Subsample absent classes get zero mass')
  targets(:,1)=y;targets(:,2)=[0,1,0,1,0,1]
  w=compute_sample_weight(targets,'balanced')
  v=compute_sample_weight(y,'balanced')
  call require(all(w==v),'Multioutput weights multiply per-output factors')
  sparse=csr_from_dense(real(targets,dp))
  v=compute_sample_weight(sparse,'balanced')
  call require(all(w==v),'Sparse class weights match dense targets')
  sparse=csr_from_arrays(2,2,[1,5,7],[2,1,2,1,2,2],[.5_dp,.5_dp,.5_dp,.5_dp,.5_dp,.5_dp])
  v=compute_sample_weight(sparse,'balanced')
  call require(all(v==1),'Sparse duplicate fragments are summed before class counting')
  maps(1)=class_weight_map([0,1],[0._dp,2._dp])
  maps(2)=class_weight_map([0,1],[ieee_value(0._dp,ieee_positive_inf),1._dp])
  w=compute_sample_weight(targets,class_weight=maps)
  call require(ieee_is_nan(w(5)),'Zero times infinity is a quiet NaN')
  v=compute_sample_weight(y==1,'balanced')
  call require(maxval(abs(v-[.75_dp,.75_dp,.75_dp,.75_dp,1.5_dp,1.5_dp]))<1.e-14_dp,'Logical labels')
  w=compute_sample_weight([integer ::])
  call require(size(w)==0,'Empty unweighted target vector')
  w=compute_sample_weight(reshape([integer ::],[3,0]),'balanced')
  call require(size(w)==1.and.w(1)==1,'Zero-output source scalar represented by a singleton result')
  print *, 'Class and sample weights, subsamples, multioutput products and sparse labels: PASS'
end program
