program example_random_feature_maps
  use sklearn, only: dp,random_trees_embedding,kernel_centerer,logistic_regression
  implicit none
  type(random_trees_embedding) :: embedding
  type(kernel_centerer) :: centerer
  type(logistic_regression) :: classifier
  real(dp) :: x(120,2)
  real(dp), allocatable :: train(:,:),test(:,:),kernel(:,:),centered(:,:),query_kernel(:,:)
  integer :: y(120),i
  do i=1,120
    x(i,:)=[sin(real(i,dp)),cos(.37_dp*i)]
    y(i)=merge(7,-3,x(i,1)*x(i,2)>0)
  end do
  embedding%n_estimators=5; embedding%max_depth=3; embedding%random_state=7
  train=embedding%fit_transform(x(:80,:)); test=embedding%transform(x(81:,:))
  classifier%tol=1.e-10_dp; classifier%max_iter=1000
  call classifier%fit(train,y(:80))
  kernel=matmul(train,transpose(train)); centered=centerer%fit_transform(kernel)
  query_kernel=centerer%transform(matmul(test,transpose(train)))
  print '(a,i0)', 'Leaf features: ',embedding%n_features_out_
  print '(a,f10.6)', 'Held-out accuracy: ',classifier%score(test,y(81:))
  print '(a,es12.4)', 'Centered kernel column residual: ',maxval(abs(sum(centered,dim=1)))
  print '(a,i0,a,i0)', 'Centered query kernel: ',size(query_kernel,1),' x ',size(query_kernel,2)
end program
