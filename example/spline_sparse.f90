program spline_sparse
 use sklearn,only:dp,spline_transformer,csr_matrix,feature_string
 implicit none
 type(spline_transformer)::model
 type(csr_matrix)::features
 type(feature_string)::input(1)
 type(feature_string),allocatable::names(:)
 real(dp)::x(8,1),weight(8)
 integer::i
 x(:,1)=[-99._dp,0._dp,1._dp,2._dp,3._dp,4._dp,5._dp,99._dp]
 weight=[0._dp,1._dp,2._dp,1._dp,1._dp,2._dp,1._dp,0._dp]
 model%n_knots=4;model%knots='quantile';model%degree=2;model%include_bias=.false.
 input(1)%text='temperature'
 features=model%fit_transform_csr(x,sample_weight=weight,feature_names=input)
 names=model%get_feature_names_out()
 print *, 'Weighted base knots:',model%base_knots_(:,1)
 print *, 'CSR shape and stored entries:',features%n_rows,features%n_cols,features%nnz()
 do i=1,size(names)
  print *, names(i)%text
 end do
end program
