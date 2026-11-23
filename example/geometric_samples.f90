program geometric_samples
  use sklearn, only: dp,make_blobs,make_moons,make_circles
  implicit none
  real(dp), allocatable :: x(:,:),returned(:,:)
  integer, allocatable :: labels(:)
  real(dp) :: centers(3,2)
  centers=reshape([-3._dp,0._dp,3._dp,-3._dp,3._dp,-3._dp],[3,2])
  call make_blobs([0,10,15],2,x=x,y=labels,random_state=42, &
    cluster_std=[.1_dp,.2_dp,.3_dp],centers=centers,centers_out=returned)
  print *, 'Blob class populations:',count(labels==0),count(labels==1),count(labels==2)
  print *, 'Returned centers unchanged:',all(returned==centers)
  call make_moons([1,7],x,labels,noise=.05_dp,random_state=42)
  print *, 'Moon class populations:',count(labels==0),count(labels==1)
  call make_circles([0,6],x,labels,factor=0._dp,shuffle=.false.)
  print *, 'Zero-factor inner-circle maximum squared radius:',maxval(sum(x*x,dim=2))
end program
