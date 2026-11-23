program test_density_extended
  use sklearn_common, only: dp,require
  use sklearn_density, only: kernel_density
  use sklearn_legacy_rng, only: legacy_random_state
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
  implicit none
  type(kernel_density) :: model,other
  type(legacy_random_state) :: rng
  integer, allocatable :: indices(:),spans(:,:)
  real(dp), allocatable :: bounds(:,:,:),radii(:),masses(:)
  real(dp) :: x(2,2),expected(3,2),z(1,16),query(1,2)
  real(dp), allocatable :: a(:,:),b(:,:),scores(:)
  x(1,:)=[0._dp,0._dp];x(2,:)=[4._dp,-2._dp]
  model%bandwidth=.7_dp;call model%fit(x,[0._dp,1._dp])
  expected(:,1)=[3.2216839173671556_dp,4.195328904540096_dp,3.5933853061835395_dp]
  expected(:,2)=[-1.7767684707174316_dp,-1.2926393006354315_dp,-2.3676188650247036_dp]
  a=model%sample(3,42)
  call require(maxval(abs(a-expected))<2.e-14_dp,'Gaussian samples match sklearn RandomState draws')
  model%kernel='tophat'
  expected(:,1)=[3.5300616802501334_dp,4.121147707626122_dp,3.7331740354992724_dp]
  expected(:,2)=[-1.8652153641340978_dp,-1.5612777975968386_dp,-2.241236383535996_dp]
  a=model%sample(3,42)
  call require(maxval(abs(a-expected))<2.e-14_dp,'source incomplete-gamma uniform-ball mapping')
  call rng%seed(42);b=model%sample(3,rng)
  call require(all(a==b),'integer and object RNG entry points agree')
  b=model%sample(3,rng);call require(any(a/=b),'object RNG state advances')
  a=model%sample(0,42);call require(all(shape(a)==[0,2]),'empty sample request')
  model%kernel='gaussian';call model%fit(x,[0._dp,0._dp]);scores=model%score_samples(x)
  call require(all(ieee_is_nan(scores)),'zero-weight scores preserve source NaN')
  a=model%sample(2,42);call require(all(abs(a)<3),'zero-weight source sampling uses the first point')
  call model%fit(x);call require(model%weight_sum_==2.and.all(model%sample_weight_==1),'unweighted refit resets weights')
  model%kernel='cosine';model%bandwidth=1;z=0;call model%fit(z);scores=model%score_samples(z)
  call require(all(ieee_is_nan(scores)),'source high-dimensional cosine normalization is retained quietly')
  other%algorithm='ball_tree';other%metric='minkowski';other%p=1;other%metric_weights=[1._dp,2._dp];call other%fit(x(:1,:))
  query(1,:)=[1._dp,1._dp];scores=other%score_samples(query)
  call require(abs(scores(1)+4.5_dp+log(2*acos(-1._dp)))<1.e-14_dp,'weighted L1 distance in Gaussian kernel')
  other%metric='invalid';other%p=-1;other%metric_weights=-9
  scores=other%score_samples(query)
  call require(abs(scores(1)+4.5_dp+log(2*acos(-1._dp)))<1.e-14_dp,'queries use the fitted metric parameters')
  other%metric='minkowski';other%p=1;other%metric_weights=[1._dp,2._dp];other%algorithm='auto'
  call other%fit(x(:1,:));scores=other%score_samples(query)
  call require(abs(scores(1)+2+log(2*acos(-1._dp)))<1.e-14_dp,'source KD-tree weighted-Minkowski bounding behavior')
  call other%get_tree_arrays(indices,spans,bounds,radii,masses)
  call require(other%n_nodes_==1.and.all(spans(1,:)==[1,1,1]).and.indices(1)==1,'tree snapshot indices and spans')
  print *, 'Extended KernelDensity and seeded sampling: PASS'
end program
