! Permutation importance adapted from sklearn 1.6.1 inspection/_permutation_importance.py.
! BSD attribution: THIRD_PARTY_NOTICES.md and LICENSES/scikit-learn-COPYING.
module sklearn_inspection
  use sklearn_partial_dependence, only: partial_dependence,partial_dependence_result,dependence_axis,dependence_predictor
  use sklearn_common, only: dp,require
  use sklearn_legacy_rng, only: legacy_random_state
  use, intrinsic :: iso_fortran_env, only: int64,real32
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value, &
    ieee_quiet_nan,ieee_positive_inf,ieee_negative_inf
  implicit none
  private
  public :: partial_dependence,partial_dependence_result,dependence_axis,dependence_predictor
  public :: permutation_importance,permutation_importance_multi,permutation_importance_result
  public :: permutation_score,permutation_scores
  type :: permutation_importance_result
    real(dp), allocatable :: importances(:,:),importances_mean(:),importances_std(:)
  end type
  abstract interface
    function permutation_score(x,rows,sample_weight) result(value)
      import dp
      real(dp), intent(in) :: x(:,:)
      integer, intent(in) :: rows(:)
      real(dp), intent(in), optional :: sample_weight(:)
      real(dp) :: value
    end function
    function permutation_scores(x,rows,sample_weight) result(values)
      import dp
      real(dp), intent(in) :: x(:,:)
      integer, intent(in) :: rows(:)
      real(dp), intent(in), optional :: sample_weight(:)
      real(dp), allocatable :: values(:)
    end function
  end interface
contains
  function permutation_importance(x,scorer,n_repeats,random_state,sample_weight,max_samples) result(results)
    real(dp), intent(in) :: x(:,:)
    procedure(permutation_score) :: scorer
    integer, intent(in), optional :: n_repeats,random_state
    real(dp), intent(in), optional :: sample_weight(:)
    class(*), intent(in), optional :: max_samples
    type(permutation_importance_result) :: results
    type(permutation_importance_result), allocatable :: multiple(:)
    multiple=permutation_importance_multi(x,vector_score,n_repeats,random_state,sample_weight,max_samples)
    results=multiple(1)
  contains
    function vector_score(features,rows,weights) result(values)
      real(dp), intent(in) :: features(:,:)
      integer, intent(in) :: rows(:)
      real(dp), intent(in), optional :: weights(:)
      real(dp), allocatable :: values(:)
      values=[scorer(features,rows,weights)]
    end function
  end function

  function permutation_importance_multi(x,scorer,n_repeats,random_state,sample_weight,max_samples) result(results)
    real(dp), intent(in) :: x(:,:)
    procedure(permutation_scores) :: scorer
    integer, intent(in), optional :: n_repeats,random_state
    real(dp), intent(in), optional :: sample_weight(:)
    class(*), intent(in), optional :: max_samples
    type(permutation_importance_result), allocatable :: results(:)
    type(legacy_random_state) :: seed_rng,rng
    real(dp), allocatable :: baseline(:),score(:),features(:,:),weights(:),column(:),deviation(:)
    integer, allocatable :: rows(:),order(:)
    integer :: repeats,seed,worker_seed,n,m,p,i,j,k,r,swap
    call require(size(x,1)>0.and.size(x,2)>0,'permutation importance features must be nonempty')
    call require(all(ieee_is_finite(x).or.ieee_is_nan(x)),'permutation importance features contain infinity')
    n=size(x,1); p=size(x,2); repeats=5; seed=0
    if(present(n_repeats)) repeats=n_repeats
    if(present(random_state)) seed=random_state
    call require(repeats>=1,'permutation importance repeats must be positive')
    call require(seed>=0,'permutation importance random_state must be nonnegative')
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'permutation importance sample weight length mismatch')
      call require(all(ieee_is_finite(sample_weight)),'permutation importance sample weights must be finite')
    end if
    m=sample_count(n,max_samples)
    call seed_rng%seed(seed); worker_seed=int(iand(seed_rng%raw_uint32(),int(huge(1),int64)))
    rows=[(i,i=1,n)]; baseline=scorer(x,rows,sample_weight)
    call require(allocated(baseline),'permutation importance scorer returned an unallocated result')
    call require(size(baseline)>0,'permutation importance scorer returned no scores')
    allocate(results(size(baseline)))
    do k=1,size(results)
      allocate(results(k)%importances(p,repeats),results(k)%importances_mean(p),results(k)%importances_std(p))
    end do
    allocate(order(m),features(m,p),column(m),deviation(repeats))
    do j=1,p
      call rng%seed(worker_seed)
      if(m<n) then
        rows=sample_indices(rng,n,m)
      else
        rows=[(i,i=1,n)]
      end if
      features=x(rows,:); order=[(i,i=1,m)]
      if(present(sample_weight)) weights=sample_weight(rows)
      do r=1,repeats
        do i=m,2,-1
          swap=rng%integer_index(i); k=order(i); order(i)=order(swap); order(swap)=k
        end do
        column=features(order,j); features(:,j)=column
        if(present(sample_weight)) then
          score=scorer(features,rows,weights)
        else
          score=scorer(features,rows)
        end if
        call require(allocated(score),'permutation importance scorer returned an unallocated result')
        call require(size(score)==size(baseline),'permutation importance scorer changed score count')
        do k=1,size(results)
          results(k)%importances(j,r)=difference(baseline(k),score(k))
        end do
      end do
      do k=1,size(results)
        results(k)%importances_mean(j)=average(results(k)%importances(j,:))
        do r=1,repeats
          deviation(r)=square(difference(results(k)%importances(j,r),results(k)%importances_mean(j)))
        end do
        results(k)%importances_std(j)=sqrt(average(deviation))
      end do
    end do
  end function

  integer function sample_count(n,max_samples) result(count)
    integer, intent(in) :: n
    class(*), intent(in), optional :: max_samples
    real(dp) :: fraction
    count=n
    if(.not.present(max_samples)) return
    select type(max_samples)
    type is(integer)
      call require(max_samples>=1.and.max_samples<=n,'permutation importance max_samples count must be in 1..n_samples')
      count=max_samples; return
    type is(integer(int64))
      call require(max_samples>=1.and.max_samples<=n,'permutation importance max_samples count must be in 1..n_samples')
      count=int(max_samples); return
    type is(real(dp))
      fraction=max_samples
    type is(real(real32))
      fraction=real(max_samples,dp)
    class default
      call require(.false.,'permutation importance max_samples must be an integer count or real fraction')
      return
    end select
    call require(ieee_is_finite(fraction),'permutation importance max_samples fraction must be finite')
    call require(fraction>0.and.fraction<=1,'permutation importance max_samples fraction must be in (0,1]')
    count=int(n*fraction)
  end function

  real(dp) function difference(a,b) result(value)
    real(dp), intent(in) :: a,b
    value=ieee_value(0._dp,ieee_quiet_nan)
    if(ieee_is_nan(a).or.ieee_is_nan(b)) return
    if(.not.ieee_is_finite(a).or..not.ieee_is_finite(b)) then
      if(a==b) return
      if(a>0.and..not.ieee_is_finite(a).or.b<0.and..not.ieee_is_finite(b)) then
        value=ieee_value(0._dp,ieee_positive_inf)
      else
        value=ieee_value(0._dp,ieee_negative_inf)
      end if
    else
      if(b<0) then
        if(a>huge(a)+b) then
          value=ieee_value(0._dp,ieee_positive_inf); return
        end if
      else if(b>0) then
        if(a< -huge(a)+b) then
          value=ieee_value(0._dp,ieee_negative_inf); return
        end if
      end if
      value=a-b
    end if
  end function

  real(dp) function square(a) result(value)
    real(dp), intent(in) :: a
    if(ieee_is_nan(a)) then
      value=a
    else if(abs(a)>sqrt(huge(a))) then
      value=ieee_value(0._dp,ieee_positive_inf)
    else
      value=a*a
    end if
  end function

  real(dp) function average(values) result(value)
    real(dp), intent(in) :: values(:)
    integer :: i
    value=0
    do i=1,size(values)
      value=difference(value,-values(i))
    end do
    value=value/size(values)
  end function
  function sample_indices(rng,population,count) result(ids)
    type(legacy_random_state), intent(inout) :: rng
    integer, intent(in) :: population,count
    integer :: ids(count),pool(population),i,j,tmp
    real(dp) :: ratio
    ratio=real(count,dp)/population
    if(ratio>.01_dp.and.ratio<.99_dp) then
      pool=[(i,i=1,population)]
      do i=population,2,-1
        j=rng%integer_index(i); tmp=pool(i); pool(i)=pool(j); pool(j)=tmp
      end do
      ids=pool(:count)
    else if(ratio<.2_dp) then
      do i=1,count
        do
          j=rng%integer_index(population)
          if(.not.any(ids(:i-1)==j)) exit
        end do
        ids(i)=j
      end do
    else
      ids=[(i,i=1,count)]
      do i=count+1,population
        j=rng%integer_index(i)
        if(j<=count) ids(j)=i
      end do
    end if
  end function
end module
