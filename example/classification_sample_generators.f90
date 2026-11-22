program classification_sample_generators
  use sklearn, only: dp,make_classification,make_hastie_10_2,classification_dataset_options
  implicit none
  type(classification_dataset_options) :: options
  real(dp), allocatable :: x(:,:)
  integer, allocatable :: labels(:)
  options%n_informative=4;options%n_redundant=2;options%n_repeated=1;options%n_classes=3
  options%weights=[.2_dp,.3_dp];options%random_state=42;options%random_shift=.true.
  call make_classification(120,9,x,labels,options)
  print '(a,3i5)', 'Classification class counts: ',count(labels==0),count(labels==1),count(labels==2)
  print '(a,f16.9)', 'Classification feature mean: ',sum(x)/size(x)
  call make_hastie_10_2(120,x,labels,42)
  print '(a,2i5)', 'Hastie negative/positive counts: ',count(labels==-1),count(labels==1)
end program
