program test_mean_shift_seeding
  use sklearn_common, only: dp,require
  use sklearn_cluster, only: mean_shift,get_bin_seeds,estimate_bandwidth
  implicit none
  type(mean_shift) :: model,explicit_model
  real(dp) :: x(12,2),ties(10,1),original(12,2),unique(3,1),same(4,2),before,after
  real(dp), allocatable :: seeds(:,:),a(:,:),b(:,:)
  integer :: i
  logical :: fallback
  ties(:,1)=[-2.5_dp,-1.5_dp,-.5_dp,.5_dp,1.5_dp,2.5_dp,3.5_dp,4.5_dp,-3.5_dp,-4.5_dp]
  seeds=get_bin_seeds(ties,1._dp,2,fallback)
  call require(.not.fallback.and.size(seeds,1)==5,'five nearest-even seed bins')
  call require(all(seeds(:,1)==[-2._dp,0._dp,2._dp,4._dp,-4._dp]),'first-seen bin order and symmetric ties')
  unique(:,1)=[.11_dp,2.13_dp,4.17_dp];seeds=get_bin_seeds(unique,.3_dp,binning_failed=fallback)
  call require(fallback.and.all(seeds==unique),'singleton bins fall back to original unrounded rows')
  seeds=get_bin_seeds(unique,0._dp,100,fallback)
  call require(.not.fallback.and.all(seeds==unique),'zero bin width ignores frequency filtering')
  seeds=get_bin_seeds(unique,1._dp,100)
  call require(size(seeds,1)==0.and.size(seeds,2)==1,'no qualifying bins retain feature dimension')
  a=get_bin_seeds(reshape([.9_dp,.9_dp,.9_dp],[3,1]),.3_dp,float64_width=.true.)
  b=get_bin_seeds(reshape([.9_dp,.9_dp,.9_dp],[3,1]),.3_dp)
  call require(a(1,1)==3*.3_dp.and.abs(a(1,1)-b(1,1))>1.e-9_dp, &
    'NumPy float64 and Python-float width promotion remain distinct')
  do i=1,12
    x(i,:)=[real((i-1)/4,dp)*3,.1_dp*real(mod(i,4),dp)]
  end do
  original=x;model%bin_seeding=.true.;model%min_bin_freq=2;model%bandwidth=.6_dp
  seeds=get_bin_seeds(x,model%bandwidth,2);explicit_model%bandwidth=model%bandwidth
  call model%fit(x);call explicit_model%fit(x,seeds=seeds)
  call require(all(model%cluster_centers_==explicit_model%cluster_centers_),'bin seeds equal explicit seed fit')
  call require(all(model%labels_==explicit_model%labels_).and.all(x==original),'bin fit labels and input nonmutation')
  model%min_bin_freq=100;call model%fit(x,seeds=x)
  call require(size(model%cluster_centers_,1)==3,'explicit seeds override bin frequency filtering')
  model%min_bin_freq=1;model%bandwidth=.01_dp;call model%fit(x)
  call require(model%binning_failed_,'fit exposes singleton fallback warning state')
  same=0;model%bandwidth=-1;call model%fit(same)
  call require(model%bandwidth_==0.and..not.model%binning_failed_,'zero automatic bandwidth resets fallback state')
  call require(size(model%cluster_centers_,1)==1,'zero automatic bandwidth supports bin seeding')
  before=estimate_bandwidth(x,.6_dp,n_samples=7,random_state=42)
  after=estimate_bandwidth(x,.6_dp,n_samples=7,random_state=42)
  call require(before==after.and.before>0,'seeded subsampling is reproducible')
  call require(estimate_bandwidth(x,n_samples=1)==0,'one sampled row gives zero bandwidth')
  call require(all(x==original),'bandwidth sampling preserves input')
  print *, 'MeanShift bin seeding, ties, fallbacks and bandwidth subsampling: PASS'
end program
