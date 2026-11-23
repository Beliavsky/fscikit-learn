program polynomial_feature_pipeline
  use sklearn, only: dp,polynomial_features,pipeline,ridge,csr_matrix,csr_from_dense,feature_string
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  type(polynomial_features) :: poly
  type(ridge) :: regression
  type(pipeline) :: model
  type(csr_matrix) :: sparse,expanded
  type(feature_string), allocatable :: names(:)
  real(dp) :: x(5,1),y(5),query(3,1)
  integer :: i
  x(:,1)=[-2._dp,-1._dp,0._dp,1._dp,2._dp];y=1+2*x(:,1)+3*x(:,1)**2
  query(:,1)=[-.5_dp,0._dp,.5_dp]
  poly%include_bias=.false.;regression%alpha=0
  allocate(model%steps(2));model%steps(1)%name='poly';model%steps(2)%name='ridge'
  allocate(model%steps(1)%model,source=poly);allocate(model%steps(2)%model,source=regression)
  call model%fit(x,y)
  print '(a,3f12.6)', 'Quadratic predictions: ',model%predict(query)
  sparse=csr_from_dense(x);expanded=poly%fit_transform(sparse);names=poly%get_feature_names_out()
  print *, 'Sparse polynomial shape:',expanded%n_rows,expanded%n_cols
  print *, 'Stored entries:',expanded%nnz()
  print '(a)', 'Feature names: '//names(1)%text//', '//names(2)%text
end program
