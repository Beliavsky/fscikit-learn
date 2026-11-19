program test_feature_agglomeration
  use sklearn, only: dp,feature_agglomeration
  use sklearn_common, only: require
  implicit none
  type(feature_agglomeration) :: model
  real(dp) :: x(3,4)
  real(dp), allocatable :: z(:,:),back(:,:),again(:,:)
  x(:,1)=[1._dp,2._dp,3._dp]; x(:,2)=x(:,1)+.1_dp
  x(:,3)=[20._dp,30._dp,40._dp]; x(:,4)=x(:,3)+.2_dp
  z=model%fit_transform(x)
  call require(all(model%labels_==[0,0,1,1]),'correlated features grouped')
  call require(maxval(abs(z(:,1)-x(:,1)-.05_dp))<1.e-12_dp,'mean pooling first cluster')
  call require(maxval(abs(z(:,2)-x(:,3)-.1_dp))<1.e-12_dp,'mean pooling second cluster')
  back=model%inverse_transform(z)
  call require(all(back(:,1)==back(:,2)).and.all(back(:,3)==back(:,4)),'inverse replicates cluster values')
  again=model%transform(back); call require(maxval(abs(again-z))<1.e-12_dp,'mean pooling is a projection')
  call require(size(model%children_,1)==3.and.model%n_leaves_==4,'full feature hierarchy')
  model%n_clusters=1; model%pooling_func='median'; z=model%fit_transform(x)
  call require(maxval(abs(z(:,1)-.5_dp*(x(:,2)+x(:,3))))<1.e-12_dp,'even median pooling')
  model%pooling_func='max'; z=model%transform(x)
  call require(all(z(:,1)==x(:,4)),'maximum pooling')
  model%n_clusters=4; model%pooling_func='mean'; z=model%fit_transform(x)
  call require(maxval(abs(z-x))<1.e-12_dp,'one feature per cluster is identity')
  print *, 'FeatureAgglomeration tests passed'
end program
