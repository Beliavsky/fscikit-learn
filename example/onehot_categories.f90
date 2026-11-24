program onehot_categories
 use sklearn,only:dp,one_hot_encoder,onehot_inverse_result,feature_string,csr_matrix
 implicit none
 type(one_hot_encoder)::encoder
 type(onehot_inverse_result)::restored
 type(csr_matrix)::features
 type(feature_string)::input(1)
 type(feature_string),allocatable::names(:)
 real(dp)::train(7,1),query(3,1)
 integer::i
 train(:,1)=[0._dp,0._dp,0._dp,1._dp,1._dp,2._dp,3._dp]
 query(:,1)=[1._dp,3._dp,99._dp]
 encoder%min_frequency_count=2;encoder%handle_unknown='infrequent_if_exist';input(1)%text='category'
 call encoder%fit(train,input);features=encoder%transform_csr(query)
 names=encoder%get_feature_names_out();restored=encoder%inverse_transform(features)
 print *, 'CSR shape and stored entries:',features%n_rows,features%n_cols,features%nnz()
 do i=1,size(names)
  print *, names(i)%text
 end do
 print *, 'Numeric values:',restored%values(:,1)
 print *, 'Infrequent category flags:',restored%is_infrequent(:,1)
 print *, 'None flags:',restored%is_none(:,1)
end program
