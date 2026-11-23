program sparse_sgd_text
  use sklearn, only: dp,hashing_vectorizer,csr_matrix,sgd_classifier
  implicit none
  type(hashing_vectorizer) :: vectorizer
  type(sgd_classifier) :: classifier
  type(csr_matrix) :: features,query
  real(dp), allocatable :: probabilities(:,:)
  vectorizer%n_features=4096
  features=vectorizer%fit_transform([character(32):: &
    'red apple sweet fruit','green pear fresh fruit','sweet orange apple fruit', &
    'blue ocean salty water','deep sea ocean water','salty sea blue water'])
  classifier%loss='log_loss';classifier%learning_rate='constant';classifier%eta0=.2_dp
  classifier%penalty='elasticnet';classifier%l1_ratio=.3_dp;classifier%random_state=42
  classifier%shuffle=.true.;classifier%max_iter=200;classifier%tol=-1
  call classifier%fit(features,[0,0,0,1,1,1])
  query=vectorizer%transform([character(24)::'sweet apple fruit','salty ocean water'])
  probabilities=classifier%predict_proba(query)
  print *, 'CSR training shape:',features%n_rows,features%n_cols
  print *, 'Stored training entries:',features%nnz()
  print *, 'Predicted classes:',classifier%predict(query)
  print *, 'Probability row sums:',sum(probabilities,dim=2)
end program
