program stateless_feature_preprocessing
  use sklearn, only: dp,normalizer,binarizer,csr_matrix,csr_from_dense,pipeline,bernoulli_nb
  implicit none
  type(normalizer) :: unit
  type(binarizer) :: binary
  type(bernoulli_nb) :: classifier
  type(pipeline) :: model
  type(csr_matrix) :: sparse,features
  real(dp) :: x(4,2),weights(4)=[1._dp,2._dp,1._dp,2._dp]
  integer :: labels(4)=[1,1,-1,-1]
  x(1,:)=[3._dp,4._dp];x(2,:)=[4._dp,3._dp]
  x(3,:)=-x(1,:);x(4,:)=-x(2,:)
  sparse=csr_from_dense(x);features=unit%fit_transform(sparse)
  print '(a,2f10.6)', 'Normalized first CSR row: ',features%data(:2)
  binary%threshold=.5_dp
  allocate(model%steps(3))
  model%steps(1)%name='unit';allocate(model%steps(1)%model,source=unit)
  model%steps(2)%name='binary';allocate(model%steps(2)%model,source=binary)
  model%steps(3)%name='classifier';allocate(model%steps(3)%model,source=classifier)
  call model%fit(x,labels,sample_weight=weights)
  print *, 'Weighted pipeline predictions:',model%predict_class(x)
end program
