program structured_features
  use sklearn, only: dp,dict_vectorizer,feature_hasher,feature_row,feature_item,csr_matrix
  implicit none
  type(dict_vectorizer) :: vectorizer
  type(feature_hasher) :: hasher
  type(feature_row) :: samples(2)
  type(csr_matrix) :: features,hashed
  real(dp), allocatable :: dense(:,:)
  samples(1)%items=[feature_item(key='temperature',value=18._dp),feature_item(key='city',category='Paris')]
  samples(2)%items=[feature_item(key='temperature',value=25._dp),feature_item(key='city',category='Rome')]
  features=vectorizer%fit_transform(samples);dense=features%to_dense()
  print *, 'Dictionary matrix shape:',shape(dense)
  print *, 'Temperature feature column:',vectorizer%vocabulary_index('temperature')
  hasher%n_features=32
  hashed=hasher%fit_transform(samples)
  print *, 'Hashed matrix shape:',hashed%n_rows,hashed%n_cols
  print *, 'Hashed stored entries:',hashed%nnz()
end program
