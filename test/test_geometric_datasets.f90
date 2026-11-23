program test_geometric_datasets
  use sklearn_common, only: dp,require
  use sklearn_datasets, only: make_blobs,make_moons,make_circles
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf,ieee_is_nan,ieee_is_finite
  implicit none
  real(dp), allocatable :: x(:,:),other(:,:),returned(:,:)
  integer, allocatable :: labels(:),again(:),before(:),after(:)
  real(dp) :: centers(3,3),original(3,3),scales(3)
  integer :: i,nseed
  do i=1,3
    centers(i,:)=[real(i,dp),-real(i,dp),2._dp*i]
  end do
  original=centers
  call make_blobs([0,2,5],99,x=x,y=labels,cluster_std=[0._dp,0._dp,0._dp], &
    centers=centers,shuffle=.false.,centers_out=returned)
  call require(all(shape(x)==[7,3]).and.all(returned==centers),'blob center feature inference and returned centers')
  call require(all(labels(:2)==1).and.all(labels(3:)==2),'unequal blob counts preserve zero-based labels and empty groups')
  call require(all(x(1,:)==centers(2,:)).and.all(x(7,:)==centers(3,:)),'zero scale places every row on its center')
  call require(all(centers==original),'explicit centers are not mutated')
  call make_blobs(2,2,5,x,labels,random_state=42,shuffle=.false.)
  call require(all(labels==[0,1]),'fewer scalar samples than blob centers')
  call make_blobs([0,0],2,x=x,y=labels,random_state=7,centers_out=returned)
  call require(all(shape(x)==[0,2]).and.size(labels)==0.and.all(shape(returned)==[2,2]), &
    'all-empty blob counts still return centers and retain output feature width')
  call make_blobs(19,3,4,x,labels,random_state=42,center_box=[2._dp,2._dp],centers_out=returned)
  call require(all(returned==2),'equal center-box bounds are valid')
  call make_blobs(19,3,4,other,again,random_state=42,center_box=[2._dp,2._dp])
  call require(all(x==other).and.all(labels==again),'center output presence preserves draws and shuffling')
  call make_moons(1,x,labels,shuffle=.false.)
  call require(all(x(1,:)==[0._dp,.5_dp]).and.labels(1)==1,'singleton moon uses the inner class')
  call make_moons([0,0],x,labels)
  call require(all(shape(x)==[0,2]).and.size(labels)==0,'empty explicit moon counts')
  call make_moons([2,1],x,labels,shuffle=.false.)
  call require(all(labels==[0,0,1]).and.all(x(3,:)==[0._dp,.5_dp]),'unequal moon counts and singleton inner arc')
  call make_circles([0,5],x,labels,factor=0._dp,shuffle=.false.)
  call require(all(x==0).and.all(labels==1),'zero-factor circle and empty outer class')
  call make_circles(1,x,labels,shuffle=.false.)
  call require(all(x(1,:)==[.8_dp,0._dp]).and.labels(1)==1,'singleton circle')
  call make_circles([4,7],x,labels,factor=.3_dp,shuffle=.false.)
  call require(maxval(abs(sum(x(:4,:)**2,dim=2)-1))<1.e-14_dp,'outer circle radius')
  call require(maxval(abs(sum(x(5:,:)**2,dim=2)-.09_dp))<1.e-14_dp,'inner circle radius')
  call random_seed(size=nseed);allocate(before(nseed),after(nseed));call random_seed(get=before)
  call make_moons(19,x,labels,noise=.2_dp,random_state=42)
  call random_seed(get=after);call require(all(before==after),'curve generator leaves intrinsic RNG unchanged')
  call make_moons(19,other,again,noise=.2_dp,random_state=42)
  call require(all(x==other).and.all(labels==again),'curve seeded reproducibility')
  scales=[0._dp,ieee_value(0._dp,ieee_positive_inf),ieee_value(0._dp,ieee_quiet_nan)]
  call make_blobs([1,3,2],2,x=x,y=labels,cluster_std=scales,shuffle=.false.)
  call require(all(ieee_is_finite(x(1,:))).and.all(.not.ieee_is_finite(x(2:4,:))), &
    'array-valued infinite scales preserve source IEEE outputs')
  call require(all(ieee_is_nan(x(5:,:))),'array-valued NaN scales preserve source IEEE outputs')
  print *, 'Seeded blobs, unequal geometric counts, center outputs and quiet IEEE values: PASS'
end program
