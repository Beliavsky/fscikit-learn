program test_stateless_preprocessing
  use sklearn_stateless_preprocessing, only: normalizer,binarizer,normalize,binarize
  use sklearn_common, only: dp,require
  use sklearn_sparse, only: csr_matrix,csr_from_arrays,csr_from_dense
  use sklearn_feature_extraction, only: feature_string
  implicit none
  type(normalizer) :: unit
  type(binarizer) :: binary
  type(csr_matrix) :: sparse,zs
  type(feature_string) :: names(2)
  type(feature_string), allocatable :: output_names(:)
  real(dp) :: x(2,2)
  real(dp), allocatable :: z(:,:),scales(:)
  x(1,:)=[3._dp,4._dp];x(2,:)=0
  z=unit%transform(x)
  call require(maxval(abs(z(1,:)-[.6_dp,.8_dp]))<1.e-15_dp.and.all(z(2,:)==0),'unfitted dense transform')
  call require(unit%n_features_in_==0,'transform before fit does not learn width')
  z=normalize(x,axis=0,norms=scales)
  call require(all(scales==[3._dp,4._dp]).and.all(z(1,:)==1),'dense column norms')
  z=normalize(x,norms=scales)
  call require(all(scales==[5._dp,1._dp]),'dense zero norm returned as one')
  x=1.e-17_dp;z=normalize(x,'max')
  call require(all(z==x),'dense tiny norms use sklearn near-zero scale guard')
  sparse=csr_from_dense(x);zs=normalize(sparse,'max',norms=scales)
  call require(all(zs%data==1).and.all(scales==1.e-17_dp),'sparse tiny norms remain nonzero')
  sparse=csr_from_arrays(1,3,[1,4],[1,1,2],[3._dp,-3._dp,2._dp])
  zs=normalize(sparse,'l1')
  call require(all(zs%indices==[1,1,2]).and.all(zs%data==[.375_dp,-.375_dp,.25_dp]), &
    'sparse L1 counts stored duplicate entries')
  zs=normalize(sparse,'max',norms=scales)
  call require(all(zs%indices==[1,2]).and.all(zs%data==[0._dp,1._dp]).and.scales(1)==2, &
    'sparse max canonicalizes duplicates and retains cancellation zero')
  zs=normalize(sparse,'max',axis=0,norms=scales)
  call require(all(scales==[0._dp,2._dp,0._dp]),'sparse returned zero column norms')
  zs=binarize(sparse)
  call require(all(zs%data==1).and.all(zs%indices==[1,2]),'sparse binarization removes zero and negative entries')
  call require(all(sparse%data==[3._dp,-3._dp,2._dp]),'sparse input is preserved')
  sparse=csr_from_arrays(1,2,[1,3],[1,1],[2._dp,3._dp]);zs=binarize(sparse)
  call require(zs%nnz()==2.and.all(zs%indices==1),'binarizer retains qualifying duplicate entries')
  x(1,:)=[3._dp,4._dp];x(2,:)=0
  names(1)%text='east';names(2)%text='north'
  call unit%fit(x,names);output_names=unit%get_feature_names_out()
  call require(unit%n_features_in_==2.and.output_names(1)%text=='east','fit captures feature metadata')
  call unit%fit(x);output_names=unit%get_feature_names_out()
  call require(.not.allocated(unit%feature_names_in_).and.output_names(2)%text=='x1','unnamed refit clears names')
  binary%threshold=-.5_dp;call binary%fit(sparse)
  call require(binary%n_features_in_==2,'sparse negative threshold fit validates data without transforming')
  binary%threshold=.5_dp;z=binary%fit_transform(x)
  call require(all(z(1,:)==1).and.all(z(2,:)==0),'dense binarizer estimator')
  call require(.not.unit%can_inverse().and..not.binary%can_inverse(),'normalization and binarization have no inverse')
  print *, 'Normalizer and Binarizer: PASS'
end program
