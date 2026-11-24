program kbins_sparse
 use sklearn,only:dp,kbins_discretizer,csr_matrix,feature_string
 implicit none
 type(kbins_discretizer)::model
 type(csr_matrix)::features
 type(feature_string)::input(1)
 type(feature_string),allocatable::names(:)
 real(dp)::x(8,1)
 real(dp),allocatable::reconstructed(:,:)
 integer::i
 x(:,1)=[0._dp,1._dp,2._dp,4._dp,8._dp,16._dp,32._dp,64._dp]
 model%strategy='quantile';model%encode='onehot';model%n_bins=3;model%subsample=6;model%random_state=42
 input(1)%text='measurement'
 features=model%fit_transform_csr(x,feature_names=input)
 reconstructed=model%inverse_transform(features);names=model%get_feature_names_out()
 print *, 'CSR shape and entries:',features%n_rows,features%n_cols,features%nnz()
 print *, 'Sampled edges:',model%bin_edges_(1)%values
 print *, 'Midpoint reconstruction:',reconstructed(:,1)
 do i=1,size(names)
  print *, names(i)%text
 end do
end program
