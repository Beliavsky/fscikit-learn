program test_regression_datasets
  use sklearn_common, only: dp,require,dense_svd
  use sklearn_datasets, only: make_low_rank_matrix,make_regression
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  real(dp), allocatable :: x(:,:),other(:,:),y(:),coef(:),targets(:,:),weights(:,:),s(:),u(:,:),vt(:,:)
  real(dp) :: wanted(7),index
  integer, allocatable :: before(:),after(:)
  integer :: i,count_seed
  x=make_low_rank_matrix(13,7,effective_rank=2,tail_strength=.3_dp,random_state=42)
  allocate(s(7),u(13,7),vt(7,7));call dense_svd(x,s,u,vt)
  do i=1,7
    index=real(i-1,dp)/2;wanted(i)=.7_dp*exp(-index**2)+.3_dp*exp(-.1_dp*index)
  end do
  call require(maxval(abs(s-wanted))<1.e-12_dp,'low rank singular spectrum')
  other=make_low_rank_matrix(13,7,effective_rank=2,tail_strength=.3_dp,random_state=42)
  call require(all(x==other),'low rank seed reproducibility')
  x=make_low_rank_matrix(1,1,random_state=7)
  call require(x(1,1)==1,'length-one QR leaves the scalar basis unchanged')
  x=make_low_rank_matrix(1,9,random_state=7)
  call require(abs(sum(x*x)-1)<1.e-12_dp,'wide one-row spectrum')
  call make_regression(23,5,x,y,n_informative=99,bias=2._dp,random_state=7,coef=coef,shuffle=.false.)
  call require(size(coef)==5.and.count(coef/=0)==5,'informative count is clamped to feature count')
  call require(maxval(abs(y-matmul(x,coef)-2))<1.e-12_dp,'noiseless scalar regression identity')
  call make_regression(23,5,other,targets,n_targets=1,n_informative=99,bias=2._dp,random_state=7, &
    coef=weights,shuffle=.false.)
  call require(all(x==other).and.all(y==targets(:,1)).and.all(coef==weights(:,1)), &
    'scalar and one-column matrix adapters share generator state')
  call make_regression(19,7,x,targets,n_targets=3,n_informative=2,bias=-3._dp,random_state=42, &
    effective_rank=3,tail_strength=.1_dp,coef=weights)
  call require(maxval(abs(targets-matmul(x,weights)+3))<1.e-12_dp,'low rank multiple-target identity')
  call require(count(any(weights/=0,dim=2))==2,'feature shuffling preserves informative rows')
  call make_regression(11,4,x,targets,n_targets=2,n_informative=0,bias=1.5_dp,random_state=0)
  call require(all(targets==1.5_dp),'zero informative features give bias-only targets')
  call random_seed(size=count_seed);allocate(before(count_seed),after(count_seed));call random_seed(get=before)
  call make_regression(9,3,x,y,random_state=42)
  call random_seed(get=after);call require(all(before==after),'seeded generator leaves intrinsic RNG state unchanged')
  call make_regression(100,5,x,targets,n_targets=3,noise=1.e308_dp,random_state=0)
  call require(any(.not.ieee_is_finite(targets)),'source-compatible overflow is quiet under floating point traps')
  print *, 'Low-rank matrices and seeded scalar/multiple-target regression: PASS'
end program
