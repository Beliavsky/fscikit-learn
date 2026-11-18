program test_text
  use sklearn_common
  use sklearn_feature_extraction_text
  implicit none
  type(count_vectorizer) :: cv
  type(tfidf_transformer) :: tf
  type(tfidf_vectorizer) :: tv
  character(40) :: documents(4)=[character(40):: &
    'This is the first document.','This document is the second document.', &
    'And this is the third one.','Is this the first document?']
  real(dp), allocatable :: counts(:,:),z(:,:),combined(:,:)
  character(128), allocatable :: names(:)
  counts=cv%fit_transform(documents); names=cv%get_feature_names_out()
  call require(size(names)==9,'text vocabulary size')
  call require(names(1)=='and'.and.names(9)=='this','lexicographic vocabulary')
  call require(all(counts(2,:)==[0,2,0,1,0,1,1,0,1]),'document counts')
  z=tf%fit_transform(counts); combined=tv%fit_transform(documents)
  call require(maxval(abs(z-combined))<1.e-12_dp,'TF-IDF vectorizer composition')
  call require(maxval(abs(sum(z**2,dim=2)-1))<1.e-12_dp,'TF-IDF normalization')
  counts=cv%transform([character(40)::'unseen vocabulary tokens'])
  call require(all(counts==0),'unknown words ignored')
  print *, 'text feature extraction: PASS'
end program
