program categorical_classification
  use sklearn, only: dp,categorical_nb
  implicit none
  type(categorical_nb) :: model
  real(dp) :: x(4,2),query(1,2)
  x(1,:)=[0._dp,0._dp]; x(2,:)=[1._dp,0._dp]
  x(3,:)=[1._dp,1._dp]; x(4,:)=[1._dp,1._dp]
  call model%partial_fit(x(:2,:),[-3,-3],classes=[-3,9])
  call model%partial_fit(x(3:,:),[9,9])
  query(1,:)=[0._dp,1._dp]
  print *, 'Predicted class:',model%predict(query)
  print *, 'Class probabilities:',model%predict_proba(query)
end program
