program text_classification_example
  use sklearn
  implicit none
  type(tfidf_vectorizer) :: features
  type(multinomial_nb) :: classifier
  character(48) :: documents(6)=[character(48):: &
    'Fortran compiler numerical arrays', 'Fortran scientific numerical computing', &
    'compiler optimizes Fortran arrays', 'garden flowers trees grow', &
    'plant flowers in the garden', 'trees plants and garden flowers']
  character(48) :: queries(2)=[character(48)::'Fortran numerical compiler','flowers in the garden']
  real(dp), allocatable :: x(:,:),query(:,:)
  x=features%fit_transform(documents); call classifier%fit(x,[0,0,0,1,1,1])
  query=features%transform(queries)
  print *, 'Predictions (computing=0, gardening=1): ',classifier%predict(query)
end program
