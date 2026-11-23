program hashed_text_documents
  use sklearn, only: dp,hashing_vectorizer,feature_string,csr_matrix,multinomial_nb
  implicit none
  type(hashing_vectorizer) :: vectorizer
  type(multinomial_nb) :: classifier
  type(csr_matrix) :: features,query
  integer :: labels(4)=[0,0,1,1]
  vectorizer%n_features=1024
  ! Nonnegative features are required by MultinomialNB.
  vectorizer%alternate_sign=.false.;vectorizer%norm='none'
  features=vectorizer%fit_transform([character(30):: &
    'red apple sweet fruit','green pear sweet fruit','blue ocean salty water','deep sea salty water'])
  call classifier%fit(features,labels)
  query=vectorizer%transform([character(24)::'sweet apple fruit','salty ocean water'])
  print *, 'Hashed CSR shape:',features%n_rows,features%n_cols
  print *, 'Stored entries:',features%nnz()
  print *, 'Predicted classes:',classifier%predict(query)
end program
