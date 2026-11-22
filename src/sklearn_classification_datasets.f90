! Adapted from sklearn 1.6.1 datasets/_samples_generator.py and utils/_random.pyx.
! BSD-3-Clause attribution: THIRD_PARTY_NOTICES.md and LICENSES/scikit-learn-COPYING.
module sklearn_classification_datasets
  use sklearn_common, only: dp,require
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_bagging_sampling, only: sample_indices
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: classification_dataset_options,make_classification,make_hastie_10_2
  type :: classification_dataset_options
    integer :: n_informative=2,n_redundant=2,n_repeated=0,n_classes=2,n_clusters_per_class=2,random_state=0
    real(dp) :: flip_y=.01_dp,class_sep=1,shift=0,scale=1
    real(dp), allocatable :: weights(:),feature_shift(:),feature_scale(:)
    logical :: hypercube=.true.,shuffle=.true.,random_shift=.false.,random_scale=.false.
  end type
contains
  subroutine normal_matrix(rng,x)
    type(legacy_random_state), intent(inout) :: rng
    real(dp), intent(out) :: x(:,:)
    integer :: i,j
    do i=1,size(x,1)
      do j=1,size(x,2)
        x(i,j)=rng%normal()
      end do
    end do
  end subroutine
  subroutine uniform_matrix(rng,x)
    type(legacy_random_state), intent(inout) :: rng
    real(dp), intent(out) :: x(:,:)
    integer :: i,j
    do i=1,size(x,1)
      do j=1,size(x,2)
        x(i,j)=2*rng%uniform()-1
      end do
    end do
  end subroutine
  subroutine shuffle_order(rng,order)
    type(legacy_random_state), intent(inout) :: rng
    integer, intent(out) :: order(:)
    integer :: i,j,tmp
    order=[(i,i=1,size(order))]
    do i=size(order),2,-1
      j=rng%integer_index(i);tmp=order(i);order(i)=order(j);order(j)=tmp
    end do
  end subroutine
  function hypercube_vertices(rng,n,dimensions) result(vertices)
    type(legacy_random_state), intent(inout) :: rng
    integer, intent(in) :: n,dimensions
    real(dp), allocatable :: vertices(:,:)
    integer, allocatable :: ids(:)
    integer :: i,j,bits,leading
    allocate(vertices(n,dimensions));bits=min(30,dimensions);leading=dimensions-bits
    do i=1,n
      do j=1,leading
        vertices(i,j)=rng%integer_index(2)-1
      end do
    end do
    ids=sample_indices(rng,2**bits,n,.false.)-1
    do i=1,n
      do j=1,bits
        vertices(i,leading+j)=merge(1._dp,0._dp,btest(ids(i),bits-j))
      end do
    end do
  end function
  subroutine validate_options(settings,n,p,clusters)
    type(classification_dataset_options), intent(in) :: settings
    integer, intent(in) :: n,p
    integer, intent(out) :: clusters
    integer(int64) :: total
    call require(n>=1.and.p>=1,'make_classification sample and feature counts must be positive')
    call require(settings%n_informative>=1,'make_classification informative count must be positive')
    call require(settings%n_redundant>=0.and.settings%n_repeated>=0,'make_classification feature counts must be nonnegative')
    total=int(settings%n_informative,int64)+settings%n_redundant+settings%n_repeated
    call require(total<=p,'make_classification feature blocks exceed n_features')
    call require(settings%n_classes>=1.and.settings%n_clusters_per_class>=1, &
      'make_classification class and cluster counts must be positive')
    total=int(settings%n_classes,int64)*settings%n_clusters_per_class
    call require(total<=huge(1),'make_classification cluster count exceeds native integer range')
    clusters=int(total)
    if(settings%n_informative<31) call require(clusters<=2**settings%n_informative, &
      'make_classification needs more informative dimensions for its clusters')
    call require(ieee_is_finite(settings%flip_y),'make_classification flip_y must be finite')
    call require(settings%flip_y>=0.and.settings%flip_y<=1,'make_classification flip_y must be in [0,1]')
    call require(ieee_is_finite(settings%class_sep),'make_classification class_sep must be finite')
    call require(settings%class_sep>0,'make_classification class_sep must be positive')
    call require(.not.(settings%random_shift.and.allocated(settings%feature_shift)), &
      'make_classification random and explicit feature shifts conflict')
    call require(.not.(settings%random_scale.and.allocated(settings%feature_scale)), &
      'make_classification random and explicit feature scales conflict')
    if(allocated(settings%feature_shift)) then
      call require(size(settings%feature_shift)==p,'make_classification feature_shift length mismatch')
      call require(all(ieee_is_finite(settings%feature_shift)),'make_classification feature shifts must be finite')
    else if(.not.settings%random_shift) then
      call require(ieee_is_finite(settings%shift),'make_classification shift must be finite')
    end if
    if(allocated(settings%feature_scale)) then
      call require(size(settings%feature_scale)==p,'make_classification feature_scale length mismatch')
      call require(all(ieee_is_finite(settings%feature_scale)),'make_classification feature scales must be finite')
    else if(.not.settings%random_scale) then
      call require(ieee_is_finite(settings%scale),'make_classification scale must be finite')
      call require(settings%scale>0,'make_classification scalar scale must be positive')
    end if
  end subroutine
  subroutine make_classification(n_samples,n_features,x,y,options)
    integer, intent(in) :: n_samples,n_features
    real(dp), allocatable, intent(out) :: x(:,:)
    integer, allocatable, intent(out) :: y(:)
    type(classification_dataset_options), optional, intent(in) :: options
    type(classification_dataset_options) :: settings
    type(legacy_random_state) :: rng
    real(dp), allocatable :: weights(:),centers(:,:),matrix(:,:),features(:,:),offset(:),scale(:),factors(:)
    integer, allocatable :: counts(:),order(:)
    logical, allocatable :: flips(:)
    integer :: i,j,k,clusters,ni,nr,np,ns,nf,extra,first,last,stop,n,m
    integer(int64) :: assigned
    if(present(options)) settings=options
    call validate_options(settings,n_samples,n_features,clusters);call rng%seed(settings%random_state)
    ns=n_samples;nf=n_features;ni=settings%n_informative;nr=settings%n_redundant;np=settings%n_repeated
    allocate(weights(settings%n_classes));weights=1._dp/settings%n_classes
    if(allocated(settings%weights)) then
      n=size(settings%weights)
      call require(n==settings%n_classes.or.n==settings%n_classes-1,'make_classification weight count mismatch')
      weights(:n)=settings%weights
      if(n==settings%n_classes-1) weights(n+1)=1-sum(weights(:n))
    end if
    call require(all(ieee_is_finite(weights)),'make_classification weights must be finite')
    call require(all(weights>=0),'make_classification weights must be nonnegative including inferred class')
    call require(all(real(ns,dp)*weights/settings%n_clusters_per_class<real(huge(1),dp)), &
      'make_classification weighted cluster size exceeds native integer range')
    allocate(counts(clusters))
    do k=1,clusters
      counts(k)=int(ns*weights(mod(k-1,settings%n_classes)+1)/settings%n_clusters_per_class)
    end do
    assigned=sum(int(counts,int64))
    do i=1,int(max(0_int64,int(ns,int64)-assigned))
      j=mod(i-1,clusters)+1;counts(j)=counts(j)+1
    end do
    centers=hypercube_vertices(rng,clusters,ni);centers=centers*(2*settings%class_sep)-settings%class_sep
    if(.not.settings%hypercube) then
      do i=1,clusters
        centers(i,:)=centers(i,:)*rng%uniform()
      end do
      allocate(factors(ni))
      do j=1,ni
        factors(j)=rng%uniform()
      end do
      do i=1,clusters
        centers(i,:)=centers(i,:)*factors
      end do
    end if
    allocate(x(ns,nf),y(ns),features(ns,ni),matrix(ni,ni));x=0;y=0
    call normal_matrix(rng,features);x(:,:ni)=features;stop=0
    do k=1,clusters
      first=stop+1;last=min(ns,stop+counts(k));stop=last
      call uniform_matrix(rng,matrix)
      if(first<=last) then
        y(first:last)=mod(k-1,settings%n_classes)
        x(first:last,:ni)=matmul(features(first:last,:),matrix)
        do i=first,last
          x(i,:ni)=x(i,:ni)+centers(k,:)
        end do
      end if
    end do
    if(nr>0) then
      deallocate(matrix);allocate(matrix(ni,nr));call uniform_matrix(rng,matrix)
      x(:,ni+1:ni+nr)=matmul(x(:,:ni),matrix)
    end if
    n=ni+nr
    do j=1,np
      m=int((n-1)*rng%uniform()+.5_dp)+1;x(:,n+j)=x(:,m)
    end do
    extra=nf-ni-nr-np
    if(extra>0) call normal_matrix(rng,x(:,nf-extra+1:))
    allocate(flips(ns))
    do i=1,ns
      flips(i)=rng%uniform()<settings%flip_y
    end do
    do i=1,ns
      if(flips(i)) y(i)=rng%integer_index(settings%n_classes)-1
    end do
    allocate(offset(nf),scale(nf));offset=settings%shift;scale=settings%scale
    if(settings%random_shift) then
      do j=1,nf
        offset(j)=(2*rng%uniform()-1)*settings%class_sep
      end do
    else if(allocated(settings%feature_shift)) then
      offset=settings%feature_shift
    end if
    if(settings%random_scale) then
      do j=1,nf
        scale(j)=1+100*rng%uniform()
      end do
    else if(allocated(settings%feature_scale)) then
      scale=settings%feature_scale
    end if
    do i=1,ns
      x(i,:)=(x(i,:)+offset)*scale
    end do
    if(settings%shuffle) then
      allocate(order(ns));call shuffle_order(rng,order);x=x(order,:);y=y(order)
      deallocate(order);allocate(order(nf));call shuffle_order(rng,order);x=x(:,order)
    end if
  end subroutine
  subroutine make_hastie_10_2(n_samples,x,y,random_state)
    integer, intent(in) :: n_samples
    real(dp), allocatable, intent(out) :: x(:,:)
    integer, allocatable, intent(out) :: y(:)
    integer, optional, intent(in) :: random_state
    type(legacy_random_state) :: rng
    integer :: seed,i
    call require(n_samples>=1,'make_hastie_10_2 sample count must be positive')
    seed=0;if(present(random_state)) seed=random_state
    call rng%seed(seed);allocate(x(n_samples,10),y(n_samples));call normal_matrix(rng,x)
    do i=1,n_samples
      y(i)=merge(1,-1,sum(x(i,:)**2)>9.34_dp)
    end do
  end subroutine
end module
