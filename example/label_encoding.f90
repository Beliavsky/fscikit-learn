program label_encoding
  use sklearn, only: label_binarizer,multi_label_binarizer,label_set
  implicit none
  type(label_binarizer) :: target
  type(multi_label_binarizer) :: tags
  type(label_set) :: samples(3)
  type(label_set), allocatable :: restored(:)
  integer, allocatable :: encoded(:,:),decoded(:)
  integer :: i
  encoded=target%fit_transform([-3,7,7,-3])
  decoded=target%inverse_transform(encoded)
  print *, 'Binary class order:',target%classes_
  print *, 'Encoded positive-class column:',encoded(:,1)
  print *, 'Restored targets:',decoded
  samples(1)%values=[10,30,10]; samples(2)%values=[20,30]
  tags%classes=[30,10,20] ! Explicit ordering; sample 3 has no labels.
  encoded=tags%fit_transform(samples); restored=tags%inverse_transform(encoded)
  print *, 'Multilabel class order:',tags%classes_
  do i=1,size(samples)
    print *, 'Sample',i,'indicator:',encoded(i,:),'labels:',restored(i)%values
  end do
end program
