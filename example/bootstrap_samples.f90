program bootstrap_samples
  use sklearn, only: dp,resample,shuffle
  implicit none
  integer :: y(6)=[0,0,1,1,1,1],i
  integer, allocatable :: ids(:),sample_y(:)
  real(dp) :: x(6,2)
  real(dp), allocatable :: sample_x(:,:)
  x(:,1)=[(real(i,dp),i=1,6)];x(:,2)=10*x(:,1)
  ids=resample(6,n_samples=5,random_state=7,stratify=y)
  sample_x=x(ids,:);sample_y=y(ids)
  print *, 'Bootstrap one-based rows:',ids
  print *, 'Class counts:',count(sample_y==0),count(sample_y==1)
  print *, 'Consistent paired rows:',all(sample_x(:,2)==10*sample_x(:,1))
  ids=shuffle(6,n_samples=3,random_state=7)
  print *, 'Shuffled holdout rows:',ids
end program
