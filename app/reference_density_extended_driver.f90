program reference_density_extended_driver
  use sklearn_common, only: dp,require
  use sklearn_density, only: kernel_density
  use sklearn_legacy_rng, only: legacy_random_state
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_positive_inf,ieee_quiet_nan
  implicit none
  type(kernel_density) :: model
  type(legacy_random_state) :: rng
  real(dp), allocatable :: x(:,:),q(:,:),weights(:),feature_weights(:),before(:,:),qb(:,:),wb(:),a(:,:),b(:,:),c(:,:),d(:,:)
  integer, allocatable :: tree_indices(:),tree_spans(:,:)
  real(dp), allocatable :: tree_bounds(:,:,:),tree_radii(:),tree_masses(:)
  real(dp) :: warmed,after(2)
  integer :: n,p,m,kernel,metric,rule,weighted,feature_weighted,draws,seed,mutate,algorithm,breadth
  character(16), parameter :: kernels(6)=[character(16)::'gaussian','tophat','epanechnikov','exponential','linear','cosine']
  character(16), parameter :: metrics(9)=[character(16)::'euclidean','l2','minkowski','p','manhattan','cityblock','l1', &
    'chebyshev','infinity']
  character(40) :: argument
  call get_command_argument(1,argument)
  if(len_trim(argument)>0) then
    call reject(trim(argument));error stop 'invalid density input accepted'
  end if
  read(*,*) n,p,m,kernel,metric,model%p,model%bandwidth,rule,weighted,feature_weighted,draws,seed,mutate, &
    algorithm,model%leaf_size,breadth,model%atol,model%rtol
  model%algorithm='auto';if(algorithm==1) model%algorithm='kd_tree'
  if(algorithm==2) model%algorithm='ball_tree'
  model%breadth_first=breadth/=0
  model%kernel=kernels(kernel+1);model%metric=metrics(metric+1)
  if(rule==1) model%bandwidth_rule='scott'
  if(rule==2) model%bandwidth_rule='silverman'
  allocate(x(n,p),q(m,p),weights(n),feature_weights(p))
  read(*,*) x;read(*,*) q;read(*,*) weights;read(*,*) feature_weights
  before=x;qb=q;wb=weights
  if(feature_weighted/=0) model%metric_weights=feature_weights
  if(weighted/=0) then
    call model%fit(x,weights)
  else
    call model%fit(x)
  end if
  if(mutate/=0) then
    model%metric='invalid';model%p=-1;model%bandwidth=-1
    if(allocated(model%metric_weights)) model%metric_weights=-1
  end if
  if(mutate==2) model%kernel='cosine'
  call rng%seed(seed);warmed=rng%normal()
  if(model%kernel=='gaussian'.or.model%kernel=='tophat') then
    if(draws<0) then
      a=model%sample(random_state=seed);b=model%sample(random_state=seed)
    else
      a=model%sample(draws,seed);b=model%sample(draws,seed)
    end if
    c=model%sample(3,rng);d=model%sample(4,rng)
  else
    allocate(a(0,p),b(0,p),c(0,p),d(0,p))
  end if
  after(1)=rng%uniform();after(2)=rng%normal()
  call integers(reshape(int([model%n_features_in_,model%n_samples_fit_,model%n_nodes_,merge(1,2,model%algorithm_=='kd_tree')],int64),[4,1]))
  call matrix(reshape([model%bandwidth_,model%weight_sum_],[2,1]))
  call matrix(reshape(model%weights_,[n,1]));call matrix(reshape(model%sample_weight_,[n,1]))
  call matrix(reshape(model%score_samples(x),[n,1]));call matrix(reshape(model%score_samples(q),[m,1]))
  call matrix(reshape([model%score(x),model%score(q)],[2,1]))
  call matrix(a);call matrix(b);call matrix(c);call matrix(d);call matrix(reshape(after,[2,1]))
  call require(all(x==before).and.all(q==qb).and.all(weights==wb),'density input changed')
  call integers(reshape([1_int64],[1,1]))
  call model%get_tree_arrays(tree_indices,tree_spans,tree_bounds,tree_radii,tree_masses)
  call integers(reshape(int(tree_indices,int64),[n,1]));call integers(int(tree_spans,int64))
  call matrix(reshape(tree_bounds,[2*model%n_nodes_,p]))
  call matrix(reshape(tree_radii,[model%n_nodes_,1]));call matrix(reshape(tree_masses,[model%n_nodes_,1]))
contains
  subroutine matrix(a)
    real(dp), intent(in) :: a(:,:)
    print *, 'F',shape(a,kind=int64)
    print '(8(es25.16e3,1x))',a
  end subroutine
  subroutine integers(a)
    integer(int64), intent(in) :: a(:,:)
    print *, 'I',shape(a,kind=int64)
    print '(20(i0,1x))',a
  end subroutine
  subroutine reject(name)
    character(*), intent(in) :: name
    real(dp) :: small(4,2),weights(4)
    real(dp), allocatable :: value(:),sample(:,:)
    small(:,1)=[0._dp,1._dp,2._dp,3._dp];small(:,2)=0;weights=1
    select case(name)
    case('algorithm');model%algorithm='bad'
    case('leaf_size');model%leaf_size=0
    case('atol_negative');model%atol=-1
    case('rtol_negative');model%rtol=-1
    case('atol_nan');model%atol=ieee_value(0._dp,ieee_quiet_nan)
    case('rtol_inf');model%rtol=ieee_value(0._dp,ieee_positive_inf)
    case('bandwidth_zero');model%bandwidth=0
    case('bandwidth_negative');model%bandwidth=-1
    case('bandwidth_nan');model%bandwidth=ieee_value(0._dp,ieee_quiet_nan)
    case('bandwidth_inf');model%bandwidth=ieee_value(0._dp,ieee_positive_inf)
    case('rule');model%bandwidth_rule='bad'
    case('kernel');model%kernel='bad'
    case('metric');model%metric='cosine'
    case('power_zero');model%metric='minkowski';model%p=0
    case('power_negative');model%metric='minkowski';model%p=-2
    case('power_nan');model%metric='minkowski';model%p=ieee_value(0._dp,ieee_quiet_nan)
    case('weighted_power_inf');model%metric='minkowski';model%p=ieee_value(0._dp,ieee_positive_inf)
      model%metric_weights=[1._dp,1._dp]
    case('metric_weights_type');model%metric_weights=[1._dp,1._dp]
    case('metric_weights_length');model%metric='minkowski';model%metric_weights=[1._dp]
    case('metric_weights_negative');model%metric='minkowski';model%metric_weights=[1._dp,-1._dp]
    case('metric_weights_nan');model%metric='minkowski';model%metric_weights=[1._dp,ieee_value(0._dp,ieee_quiet_nan)]
    case('weights_negative');weights(1)=-1
    case('weights_nan');weights(1)=ieee_value(0._dp,ieee_quiet_nan)
    case('weights_inf');weights(1)=ieee_value(0._dp,ieee_positive_inf)
    case('weights_length');call model%fit(small,weights(:3));return
    case('unfitted_tree');call model%get_tree_arrays(tree_indices,tree_spans,tree_bounds,tree_radii,tree_masses);return
    case('unfitted_scores');value=model%score_samples(small);return
    case('unfitted_sample');sample=model%sample(1,0);return
    case('query_width');call model%fit(small);value=model%score_samples(small(:,:1));return
    case('query_nan');call model%fit(small);small(1,1)=ieee_value(0._dp,ieee_quiet_nan)
      value=model%score_samples(small);return
    case('sample_negative');call model%fit(small);sample=model%sample(-1,0);return
    case('sample_seed');call model%fit(small);sample=model%sample(1,-1);return
    case('sample_kernel');model%kernel='cosine';call model%fit(small);sample=model%sample(1,0);return
    case('rng_unseeded');call model%fit(small);sample=model%sample(1,rng);return
    case default;error stop 'unknown density rejection case'
    end select
    call model%fit(small,weights)
  end subroutine
end program
