program synthetic_sample_generators
  use sklearn, only: dp,make_friedman1,make_swiss_roll,make_gaussian_quantiles
  implicit none
  real(dp), allocatable :: x(:,:),target(:)
  integer, allocatable :: labels(:)
  call make_friedman1(100,8,x,target,noise=.2_dp,random_state=42)
  print '(a,f14.8)', 'Friedman1 mean target: ',sum(target)/size(target)
  call make_swiss_roll(100,x,target,noise=.1_dp,random_state=42,hole=.true.)
  print '(a,f14.8)', 'Swiss roll mean coordinate: ',sum(target)/size(target)
  call make_gaussian_quantiles(100,3,x,labels,n_classes=4,random_state=42)
  print '(a,4i5)', 'Gaussian quantile class counts: ',count(labels==0),count(labels==1),count(labels==2),count(labels==3)
end program
