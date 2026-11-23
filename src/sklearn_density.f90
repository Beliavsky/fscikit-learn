! sklearn 1.6.1 neighbors/_kde.py, _binary_tree.pxi.tp and distance contracts.
! Source seeded sampling: NumPy RandomState and scipy.special.gammainc.
! Native KD/Ball tree bounds and traversals; retained BSD notices in THIRD_PARTY_NOTICES.md.
module sklearn_density
  use sklearn_common, only: dp,require,check_x
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_ieee_math, only: ieee_difference,ieee_product,ieee_quotient
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_negative_inf,ieee_positive_inf,ieee_quiet_nan, &
    ieee_is_finite,ieee_is_nan
  implicit none
  private
  public :: kernel_density
  type(legacy_random_state), save :: default_rng
  logical, save :: default_initialized=.false.
  type :: density_node
    integer :: first=0,last=0
    real(dp) :: mass=0,radius=0
    real(dp), allocatable :: low(:),high(:),center(:)
  end type
  type :: kernel_density
    real(dp) :: bandwidth=1,bandwidth_=1
    character(16) :: kernel='gaussian',bandwidth_rule='fixed'
    real(dp), allocatable :: x_(:,:),weights_(:)
    character(16) :: metric='euclidean'
    real(dp) :: p=2,weight_sum_=0
    real(dp), allocatable :: metric_weights(:),sample_weight_(:)
    integer :: n_features_in_=0,n_samples_fit_=0
    character(16), private :: metric_fit_='euclidean'
    real(dp), private :: p_fit_=2
    real(dp), allocatable, private :: metric_weights_fit_(:)
    logical, private :: weighted_=.false.
    character(16) :: algorithm='auto',algorithm_='kd_tree'
    integer :: leaf_size=40,n_nodes_=0
    real(dp) :: atol=0,rtol=0
    logical :: breadth_first=.true.
    type(density_node), allocatable, private :: nodes_(:)
    integer, allocatable, private :: order_(:)
  contains
    procedure :: fit=>density_fit
    procedure :: score_samples=>density_scores
    procedure :: score=>density_score
    procedure :: get_tree_arrays=>density_tree_arrays
    procedure, private :: sample_seed=>density_sample,sample_state=>density_sample_state
    generic :: sample=>sample_seed,sample_state
  end type
contains
  real(dp) function plus(a,b) result(value)
    real(dp), intent(in) :: a,b
    value=ieee_difference(a,-b)
  end function
  recursive real(dp) function numpy_sum(values) result(total)
    real(dp), intent(in) :: values(:)
    real(dp) :: partial(8)
    integer :: n,i,j,whole,split
    n=size(values);total=0
    if(n==0) return
    if(n<8) then
      total=-0._dp
      do i=1,n
        total=plus(total,values(i))
      end do
    else if(n<=128) then
      partial=values(:8);whole=n-mod(n,8)
      do i=9,whole,8
        do j=1,8
          partial(j)=plus(partial(j),values(i+j-1))
        end do
      end do
      total=plus(plus(plus(partial(1),partial(2)),plus(partial(3),partial(4))), &
        plus(plus(partial(5),partial(6)),plus(partial(7),partial(8))))
      do i=whole+1,n
        total=plus(total,values(i))
      end do
    else
      split=n/2;split=split-mod(split,8)
      total=plus(numpy_sum(values(:split)),numpy_sum(values(split+1:)))
    end if
  end function
  real(dp) function quiet_log(x) result(value)
    real(dp), intent(in) :: x
    value=ieee_value(0._dp,ieee_quiet_nan)
    if(ieee_is_nan(x)) return
    if(x==0) then
      value=ieee_value(0._dp,ieee_negative_inf)
    else if(x>0) then
      value=log(x)
    end if
  end function
  real(dp) function quiet_power(x,power) result(value)
    real(dp), intent(in) :: x,power
    value=ieee_value(0._dp,ieee_quiet_nan)
    if(ieee_is_nan(power)) return
    if(power==0) then
      value=1;return
    end if
    if(ieee_is_nan(x)) return
    if(.not.ieee_is_finite(power)) then
      value=1
      if(x>1) value=ieee_value(0._dp,ieee_positive_inf)
      if(x<1) value=0
    else if(x==0) then
      value=0
    else if(ieee_product(log(x),power)>log(huge(x))) then
      value=ieee_value(0._dp,ieee_positive_inf)
    else
      value=x**power
    end if
  end function
  real(dp) function log_add(a,b) result(value)
    real(dp), intent(in) :: a,b
    real(dp) :: maximum
    ! C fmax suppresses one NaN; retain the source -Inf/NaN boundary case.
    maximum=a
    if(ieee_is_nan(a)) then
      maximum=b
    else if(.not.ieee_is_nan(b)) then
      maximum=max(a,b)
    end if
    value=maximum
    if(ieee_is_nan(maximum)) return
    if(maximum==ieee_value(0._dp,ieee_negative_inf)) return
    value=plus(maximum,quiet_log(exp(ieee_difference(a,maximum))+exp(ieee_difference(b,maximum))))
  end function
  subroutine density_fit(self,x,sample_weight)
    class(kernel_density), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp) :: maximum
    integer :: n,d
    call check_x(x);n=size(x,1);d=size(x,2)
    call require(any(self%kernel==[character(16)::'gaussian','tophat','epanechnikov','exponential','linear','cosine']), &
      'unknown density kernel')
    select case(self%bandwidth_rule)
    case('fixed');self%bandwidth_=self%bandwidth
    case('scott');self%bandwidth_=real(n,dp)**(-1._dp/(d+4))
    case('silverman');self%bandwidth_=(real(n,dp)*(d+2)/4)**(-1._dp/(d+4))
    case default;call require(.false.,'unknown density bandwidth rule')
    end select
    call require(ieee_is_finite(self%bandwidth_),'density bandwidth must be positive and finite')
    call require(self%bandwidth_>0,'density bandwidth must be positive and finite')
    call require(any(self%algorithm==[character(16)::'auto','kd_tree','ball_tree']),'invalid density algorithm')
    call require(self%leaf_size>0,'density leaf_size must be positive')
    call tolerances(self)
    self%algorithm_=self%algorithm
    if(self%algorithm_=='auto') self%algorithm_='kd_tree'
    call require(any(self%metric==[character(16)::'euclidean','l2','minkowski','p','manhattan','cityblock','l1', &
      'chebyshev','infinity']),'unsupported density metric')
    self%metric_fit_=self%metric;self%p_fit_=self%p
    if(allocated(self%metric_weights_fit_)) deallocate(self%metric_weights_fit_)
    if(allocated(self%metric_weights)) then
      call require(self%metric=='minkowski'.or.self%metric=='p','density metric weights require Minkowski')
      call require(size(self%metric_weights)==d,'density metric weight count mismatch')
      call require(all(ieee_is_finite(self%metric_weights)),'density metric weights must be finite')
      call require(all(self%metric_weights>=0),'negative density metric weights')
      self%metric_weights_fit_=self%metric_weights
    end if
    if(self%metric=='minkowski'.or.self%metric=='p') then
      call require(.not.ieee_is_nan(self%p),'density Minkowski p is NaN')
      if(.not.ieee_is_finite(self%p).and..not.allocated(self%metric_weights)) then
        self%metric_fit_='chebyshev'
      else
        call require(self%p>0,'density Minkowski p must be positive')
        call require(ieee_is_finite(self%p),'weighted density Minkowski p must be finite')
        if(.not.allocated(self%metric_weights)) then
          if(self%p==1) self%metric_fit_='manhattan'
          if(self%p==2) self%metric_fit_='euclidean'
        end if
      end if
    end if
    self%sample_weight_=spread(1._dp,1,n);self%weighted_=present(sample_weight)
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'density sample weight size mismatch')
      call require(all(ieee_is_finite(sample_weight)),'nonfinite density sample weights')
      call require(all(sample_weight>=0),'invalid density sample weights')
      self%sample_weight_=sample_weight
    end if
    self%weight_sum_=numpy_sum(self%sample_weight_);self%weights_=self%sample_weight_
    maximum=maxval(self%weights_)
    if(maximum>0) then
      self%weights_=self%weights_/maximum;self%weights_=self%weights_/numpy_sum(self%weights_)
    end if
    self%x_=x;self%n_features_in_=d;self%n_samples_fit_=n
    call build_tree(self)
  end subroutine
  real(dp) function distance(self,x,y) result(value)
    class(kernel_density), intent(in) :: self
    real(dp), intent(in) :: x(:),y(:)
    real(dp) :: delta,term
    integer :: j
    value=0
    do j=1,size(x)
      delta=abs(ieee_difference(x(j),y(j)))
      select case(self%metric_fit_)
      case('euclidean','l2');value=plus(value,ieee_product(delta,delta))
      case('manhattan','cityblock','l1');value=plus(value,delta)
      case('chebyshev','infinity')
        if(.not.ieee_is_nan(delta)) value=max(value,delta)
      case('minkowski','p')
        term=quiet_power(delta,self%p_fit_)
        if(allocated(self%metric_weights_fit_)) term=ieee_product(term,self%metric_weights_fit_(j))
        value=plus(value,term)
      end select
    end do
    select case(self%metric_fit_)
    case('euclidean','l2');value=sqrt(value)
    case('minkowski','p');value=quiet_power(value,ieee_quotient(1._dp,self%p_fit_))
    end select
  end function
  real(dp) function kernel_norm(self) result(value)
    class(kernel_density), intent(in) :: self
    real(dp) :: volume,factor,tmp,pi
    integer :: d,k
    d=self%n_features_in_;pi=acos(-1._dp)
    volume=.5_dp*d*log(pi)-log_gamma(.5_dp*d+1)
    select case(self%kernel)
    case('gaussian');value=-.5_dp*d*log(2*pi)
    case('tophat');value=-volume
    case('epanechnikov');value=-(volume+log(2._dp/(d+2)))
    case('exponential');value=-(log(2*pi)+.5_dp*(d-2)*log(pi)-log_gamma(.5_dp*d)+log_gamma(real(d,dp)))
    case('linear');value=-(volume-log(real(d+1,dp)))
    case('cosine')
      factor=0;tmp=2/pi
      do k=1,d,2
        factor=plus(factor,tmp)
        tmp=ieee_product(tmp,ieee_product(-real(d-k,dp),real(d-k-1,dp))*(2/pi)**2)
      end do
      value=-(quiet_log(factor)+log(2*pi)+.5_dp*(d-2)*log(pi)-log_gamma(.5_dp*d))
    case default;value=0;call require(.false.,'unknown density kernel')
    end select
    value=ieee_difference(value,ieee_product(real(d,dp),log(self%bandwidth_)))
  end function
  real(dp) function log_kernel(self,dist) result(value)
    class(kernel_density), intent(in) :: self
    real(dp), intent(in) :: dist
    real(dp) :: ratio,angle
    value=ieee_value(0._dp,ieee_negative_inf)
    if(ieee_is_nan(dist)) then
      if(self%kernel=='gaussian'.or.self%kernel=='exponential') value=dist
      return
    end if
    select case(self%kernel)
    case('gaussian')
      value=ieee_quotient(ieee_product(-.5_dp,ieee_product(dist,dist)), &
        ieee_product(self%bandwidth_,self%bandwidth_))
    case('exponential');value=-ieee_quotient(dist,self%bandwidth_)
    case default
      if(dist>=self%bandwidth_) return
      select case(self%kernel)
      case('tophat');value=0
      case('epanechnikov')
        ratio=ieee_quotient(ieee_product(dist,dist),ieee_product(self%bandwidth_,self%bandwidth_))
        value=quiet_log(ieee_difference(1._dp,ratio))
      case('linear');value=quiet_log(ieee_difference(1._dp,ieee_quotient(dist,self%bandwidth_)))
      case('cosine')
        angle=ieee_quotient(ieee_product(.5_dp*acos(-1._dp),dist),self%bandwidth_)
        value=ieee_value(0._dp,ieee_quiet_nan)
        if(ieee_is_finite(angle)) value=quiet_log(cos(angle))
      end select
    end select
  end function
  subroutine density_tree_arrays(self,indices,spans,node_bounds,radii,masses)
    class(kernel_density), intent(in) :: self
    integer, allocatable, intent(out) :: indices(:),spans(:,:)
    real(dp), allocatable, intent(out) :: node_bounds(:,:,:),radii(:),masses(:)
    integer :: i
    call require(allocated(self%nodes_),'KernelDensity is not fitted')
    indices=self%order_;allocate(spans(self%n_nodes_,3),node_bounds(2,self%n_nodes_,self%n_features_in_), &
      radii(self%n_nodes_),masses(self%n_nodes_));node_bounds=0
    do i=1,self%n_nodes_
      spans(i,:)=[self%nodes_(i)%first,self%nodes_(i)%last,merge(1,0,2*i>self%n_nodes_)]
      radii(i)=self%nodes_(i)%radius;masses(i)=self%nodes_(i)%mass
      if(self%algorithm_=='ball_tree') then
        node_bounds(1,i,:)=self%nodes_(i)%center
      else
        node_bounds(1,i,:)=self%nodes_(i)%low;node_bounds(2,i,:)=self%nodes_(i)%high
      end if
    end do
  end subroutine
  subroutine tolerances(self)
    class(kernel_density), intent(in) :: self
    call require(ieee_is_finite(self%atol).and.ieee_is_finite(self%rtol),'density tolerances must be finite')
    call require(self%atol>=0.and.self%rtol>=0,'density tolerances must be nonnegative')
  end subroutine
  logical function leq(a,b) result(value)
    real(dp), intent(in) :: a,b
    value=.false.
    if(ieee_is_nan(a).or.ieee_is_nan(b)) return
    value=a<=b
  end function
  real(dp) function log_sub(a,b) result(value)
    real(dp), intent(in) :: a,b
    value=ieee_value(0._dp,ieee_negative_inf)
    if(leq(a,b)) return
    value=plus(a,quiet_log(ieee_difference(1._dp,exp(ieee_difference(b,a)))))
  end function
    subroutine sort_indices(self,first,last,feature)
      class(kernel_density), intent(inout) :: self
      integer, intent(in) :: first,last,feature
      integer, allocatable :: ids(:)
      integer :: root,finish,swap
      ids=self%order_(first:last)
      do root=size(ids)/2,1,-1
        call sift(root,size(ids))
      end do
      do finish=size(ids),2,-1
        swap=ids(1);ids(1)=ids(finish);ids(finish)=swap;call sift(1,finish-1)
      end do
      self%order_(first:last)=ids
    contains
      subroutine sift(start,finish)
        integer, intent(in) :: start,finish
        integer :: at,child,swap
        at=start
        do while(2*at<=finish)
          child=2*at
          if(child<finish) then
            if(greater(ids(child+1),ids(child))) child=child+1
          end if
          if(.not.greater(ids(child),ids(at))) exit
          swap=ids(at);ids(at)=ids(child);ids(child)=swap;at=child
        end do
      end subroutine
      logical function greater(a,b) result(value)
        integer, intent(in) :: a,b
        value=self%x_(a,feature)>self%x_(b,feature)
        if(self%x_(a,feature)==self%x_(b,feature)) value=a>b
      end function
    end subroutine
  subroutine build_tree(self)
    class(kernel_density), intent(inout) :: self
    integer :: levels,ratio,i,t
    ratio=max(1,(self%n_samples_fit_-1)/self%leaf_size);levels=1
    do while(ratio>=2)
      ratio=ratio/2;levels=levels+1
    end do
    call require(levels<30,'density tree exceeds integer capacity')
    self%n_nodes_=2**levels-1
    if(allocated(self%nodes_)) deallocate(self%nodes_)
    allocate(self%nodes_(self%n_nodes_));self%order_=[(i,i=1,self%n_samples_fit_)]
    call grow(1,1,self%n_samples_fit_)
    ! Query node weights use the final partition order, independently of the
    ! pre-partition order used to initialize a BallTree centroid.
    do i=1,self%n_nodes_
      self%nodes_(i)%mass=0
      do t=self%nodes_(i)%first,self%nodes_(i)%last
        self%nodes_(i)%mass=plus(self%nodes_(i)%mass,self%sample_weight_(self%order_(t)))
      end do
    end do
  contains
    recursive subroutine grow(node,first,last)
      integer, intent(in) :: node,first,last
      real(dp), allocatable :: spread(:)
      real(dp) :: term,dist,power
      integer :: j,t,k,feature,key,at,middle
      self%nodes_(node)%first=first;self%nodes_(node)%last=last
      allocate(self%nodes_(node)%low(self%n_features_in_),self%nodes_(node)%high(self%n_features_in_), &
        self%nodes_(node)%center(self%n_features_in_))
      self%nodes_(node)%low=minval(self%x_(self%order_(first:last),:),dim=1)
      self%nodes_(node)%high=maxval(self%x_(self%order_(first:last),:),dim=1)
      self%nodes_(node)%mass=0;self%nodes_(node)%center=0
      do t=first,last
        k=self%order_(t);term=self%sample_weight_(k)
        self%nodes_(node)%mass=plus(self%nodes_(node)%mass,term)
        if(self%algorithm_=='ball_tree') then
          do j=1,self%n_features_in_
            self%nodes_(node)%center(j)=plus(self%nodes_(node)%center(j),ieee_product(self%x_(k,j),term))
          end do
        end if
      end do
      if(self%algorithm_=='ball_tree') then
        do j=1,self%n_features_in_
          self%nodes_(node)%center(j)=ieee_quotient(self%nodes_(node)%center(j),self%nodes_(node)%mass)
        end do
        do t=first,last
          dist=distance(self,self%nodes_(node)%center,self%x_(self%order_(t),:))
          if(.not.ieee_is_nan(dist)) self%nodes_(node)%radius=max(self%nodes_(node)%radius,dist)
        end do
      end if
      if(self%algorithm_=='kd_tree') then
        power=self%p_fit_
        select case(self%metric_fit_)
        case('euclidean','l2');power=2
        case('manhattan','cityblock','l1');power=1
        case('chebyshev','infinity');power=ieee_value(0._dp,ieee_positive_inf)
        end select
        term=0
        do j=1,self%n_features_in_
          dist=.5_dp*ieee_difference(self%nodes_(node)%high(j),self%nodes_(node)%low(j))
          if(ieee_is_finite(power)) then
            term=plus(term,quiet_power(dist,power))
          else
            term=max(term,dist)
          end if
        end do
        ! The source also applies pow(rad, 1/p) for Chebyshev, yielding
        ! the stored diagnostic radius 1 through pow(rad, 0).
        self%nodes_(node)%radius=quiet_power(term,ieee_quotient(1._dp,power))
      end if
      if(2*node>self%n_nodes_.or.last-first<1) return
      allocate(spread(self%n_features_in_))
      do j=1,self%n_features_in_
        spread(j)=ieee_difference(self%nodes_(node)%high(j),self%nodes_(node)%low(j))
      end do
      feature=maxloc(spread,dim=1)
      call sort_indices(self,first,last,feature)
      middle=first+(last-first+1)/2
      call grow(2*node,first,middle-1);call grow(2*node+1,middle,last)
    end subroutine
  end subroutine
  subroutine bounds(self,node,point,low,high)
    class(kernel_density), intent(in) :: self
    integer, intent(in) :: node
    real(dp), intent(in) :: point(:)
    real(dp), intent(out) :: low,high
    real(dp) :: d_lo,d_hi,delta,maximum,power,dist
    integer :: j
    if(self%algorithm_=='ball_tree') then
      dist=distance(self,point,self%nodes_(node)%center)
      low=ieee_difference(dist,self%nodes_(node)%radius)
      if(ieee_is_nan(low)) then
        low=0
      else
        low=max(0._dp,low)
      end if
      high=plus(dist,self%nodes_(node)%radius);return
    end if
    power=self%p_fit_
    select case(self%metric_fit_)
    case('euclidean','l2');power=2
    case('manhattan','cityblock','l1');power=1
    case('chebyshev','infinity');power=ieee_value(0._dp,ieee_positive_inf)
    end select
    low=0;high=0
    do j=1,self%n_features_in_
      d_lo=ieee_difference(self%nodes_(node)%low(j),point(j))
      d_hi=ieee_difference(point(j),self%nodes_(node)%high(j))
      delta=ieee_product(.5_dp,plus(plus(d_lo,abs(d_lo)),plus(d_hi,abs(d_hi))))
      maximum=max(abs(d_lo),abs(d_hi))
      if(.not.ieee_is_finite(power)) then
        low=max(low,delta);high=max(high,maximum)
      else
        low=plus(low,quiet_power(delta,power));high=plus(high,quiet_power(maximum,power))
      end if
    end do
    if(ieee_is_finite(power)) then
      low=quiet_power(low,ieee_quotient(1._dp,power));high=quiet_power(high,ieee_quotient(1._dp,power))
    end if
  end subroutine
  function density_scores(self,x) result(scores)
    class(kernel_density), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: scores(size(x,1)),normalization,log_mass,log_atol,log_rtol
    real(dp) :: global_lower,global_width,local_lower,local_width,near,far
    real(dp), allocatable :: lower(:),width(:),priority(:)
    integer, allocatable :: heap(:)
    integer :: row,node,active
    call require(allocated(self%x_),'KernelDensity is not fitted');call check_x(x);call tolerances(self)
    call require(size(x,2)==self%n_features_in_,'density query feature count mismatch')
    normalization=kernel_norm(self);log_mass=quiet_log(self%weight_sum_)
    log_atol=quiet_log(ieee_product(self%atol,self%weight_sum_));log_rtol=quiet_log(self%rtol)
    allocate(lower(self%n_nodes_),width(self%n_nodes_),priority(self%n_nodes_),heap(self%n_nodes_))
    do row=1,size(x,1)
      ! Root mass uses NumPy's total; child masses follow the indexed C loops.
      call bounds(self,1,x(row,:),near,far)
      global_lower=plus(log_mass,log_kernel(self,far))
      global_width=log_sub(plus(log_mass,log_kernel(self,near)),global_lower)
      if(self%breadth_first) then
        lower(1)=global_lower;width(1)=global_width;active=0;call push(1,near)
        do while(active>0)
          node=pop()
          if(local_done(node,lower(node),width(node))) cycle
          if(global_done()) exit
          if(2*node>self%n_nodes_) then
            call leaf(node,lower(node),width(node))
          else
            call split(node,lower(node),width(node),.true.)
          end if
        end do
      else
        local_lower=global_lower;local_width=global_width
        call depth_first(1,local_lower,local_width)
      end if
      scores(row)=ieee_difference(plus(log_add(global_lower,ieee_difference(global_width,log(2._dp))),normalization),log_mass)
    end do
  contains
    logical function local_done(node,lo,spread) result(done)
      integer, intent(in) :: node
      real(dp), intent(in) :: lo,spread
      real(dp) :: lhs,rhs
      lhs=plus(ieee_difference(plus(normalization,spread),quiet_log(self%nodes_(node)%mass)),log_mass)
      rhs=log_add(log_atol,plus(plus(log_rtol,normalization),lo));done=leq(lhs,rhs)
    end function
    logical function global_done() result(done)
      done=leq(plus(normalization,global_width),log_add(log_atol,plus(plus(log_rtol,normalization),global_lower)))
    end function
    subroutine node_bounds(node,lo,spread,minimum)
      integer, intent(in) :: node
      real(dp), intent(out) :: lo,spread,minimum
      real(dp) :: maximum,lmass
      call bounds(self,node,x(row,:),minimum,maximum);lmass=quiet_log(self%nodes_(node)%mass)
      lo=plus(lmass,log_kernel(self,maximum))
      spread=log_sub(plus(lmass,log_kernel(self,minimum)),lo)
    end subroutine
    subroutine leaf(node,lo,spread)
      integer, intent(in) :: node
      real(dp), intent(in) :: lo,spread
      real(dp) :: term
      integer :: t,k
      global_lower=log_sub(global_lower,lo);global_width=log_sub(global_width,spread)
      do t=self%nodes_(node)%first,self%nodes_(node)%last
        k=self%order_(t);term=log_kernel(self,distance(self,x(row,:),self%x_(k,:)))
        if(self%weighted_) term=plus(term,quiet_log(self%sample_weight_(k)))
        global_lower=log_add(global_lower,term)
      end do
    end subroutine
    subroutine split(node,lo,spread,breadth)
      integer, intent(in) :: node
      real(dp), intent(in) :: lo,spread
      logical, intent(in) :: breadth
      real(dp) :: minimum1,minimum2,maximum
      integer :: left,right
      left=2*node;right=left+1
      call node_bounds(left,lower(left),width(left),minimum1)
      call node_bounds(right,lower(right),width(right),minimum2)
      if(breadth) then
        ! Source breadth-first stores each child's upper bound as its spread.
        call bounds(self,left,x(row,:),minimum1,maximum)
        width(left)=plus(quiet_log(self%nodes_(left)%mass),log_kernel(self,minimum1))
        call bounds(self,right,x(row,:),minimum2,maximum)
        width(right)=plus(quiet_log(self%nodes_(right)%mass),log_kernel(self,minimum2))
      end if
      global_lower=log_sub(global_lower,lo)
      global_lower=log_add(log_add(global_lower,lower(left)),lower(right))
      global_width=log_sub(global_width,spread)
      global_width=log_add(log_add(global_width,width(left)),width(right))
      if(breadth) then
        call push(left,minimum1);call push(right,minimum2)
      end if
    end subroutine
    recursive subroutine depth_first(node,lo,spread)
      integer, intent(in) :: node
      real(dp), intent(in) :: lo,spread
      if(local_done(node,lo,spread)) return
      if(global_done()) return
      if(2*node>self%n_nodes_) then
        call leaf(node,lo,spread)
      else
        call split(node,lo,spread,.false.)
        call depth_first(2*node,lower(2*node),width(2*node))
        call depth_first(2*node+1,lower(2*node+1),width(2*node+1))
      end if
    end subroutine
    subroutine push(node,key)
      integer, intent(in) :: node
      real(dp), intent(in) :: key
      integer :: at,parent,swap
      priority(node)=key;active=active+1;heap(active)=node;at=active
      do while(at>1)
        parent=at/2
        if(leq(priority(heap(parent)),priority(heap(at)))) exit
        swap=heap(parent);heap(parent)=heap(at);heap(at)=swap;at=parent
      end do
    end subroutine
    integer function pop() result(node)
      integer :: at,child,swap
      node=heap(1);heap(1)=heap(active);active=active-1;at=1
      do while(2*at<=active)
        child=2*at
        if(child<active) then
          if(.not.leq(priority(heap(child)),priority(heap(child+1)))) child=child+1
        end if
        if(.not.leq(priority(heap(child)),priority(heap(at)))) exit
        swap=heap(at);heap(at)=heap(child);heap(child)=swap;at=child
      end do
    end function
  end function
  real(dp) function density_score(self,x) result(score)
    class(kernel_density), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    score=numpy_sum(self%score_samples(x))
  end function
  real(dp) function gamma_cdf(a,x) result(value)
    real(dp), intent(in) :: a,x
    real(dp) :: term,total,b,c,d,h,factor,an,delta
    integer :: i
    value=0
    if(x<=0) return
    factor=exp(a*log(x)-x-log_gamma(a))
    if(x<a+1) then
      term=1/a;total=term
      do i=1,10000
        term=term*x/(a+i);total=total+term
        if(abs(term)<=epsilon(1._dp)*abs(total)) exit
      end do
      value=total*factor
    else
      b=x+1-a;c=1.e300_dp;d=1/b;h=d
      do i=1,10000
        an=-real(i,dp)*(i-a);b=b+2;d=an*d+b
        if(abs(d)<1.e-300_dp) d=sign(1.e-300_dp,d)
        c=b+an/c
        if(abs(c)<1.e-300_dp) c=sign(1.e-300_dp,c)
        d=1/d;delta=d*c;h=h*delta
        if(abs(delta-1)<=epsilon(1._dp)) exit
      end do
      value=1-factor*h
    end if
    call require(i<=10000,'density incomplete gamma iteration did not converge')
    value=max(0._dp,min(1._dp,value))
  end function
  subroutine sample_parameters(self,n_samples,n)
    class(kernel_density), intent(in) :: self
    integer, intent(in), optional :: n_samples
    integer, intent(out) :: n
    call require(allocated(self%x_),'KernelDensity is not fitted')
    call require(self%kernel=='gaussian'.or.self%kernel=='tophat','density sampling supports gaussian/tophat only')
    n=1;if(present(n_samples)) n=n_samples
    call require(n>=0,'density sample count must be nonnegative')
  end subroutine
  function density_sample(self,n_samples,random_state) result(x)
    class(kernel_density), intent(in) :: self
    integer, intent(in), optional :: n_samples,random_state
    real(dp), allocatable :: x(:,:)
    type(legacy_random_state) :: rng
    integer(int64) :: count
    integer :: n
    call sample_parameters(self,n_samples,n)
    if(present(random_state)) then
      call require(random_state>=0,'density sample seed must be nonnegative')
      call rng%seed(random_state);x=draw_samples(self,n,rng)
    else
      if(.not.default_initialized) then
        call system_clock(count=count)
        call default_rng%seed(int(modulo(count,2147483647_int64)))
        default_initialized=.true.
      end if
      x=draw_samples(self,n,default_rng)
    end if
  end function
  function density_sample_state(self,n_samples,random_state) result(x)
    class(kernel_density), intent(in) :: self
    integer, intent(in), optional :: n_samples
    type(legacy_random_state), intent(inout) :: random_state
    real(dp), allocatable :: x(:,:)
    integer :: n
    call sample_parameters(self,n_samples,n);x=draw_samples(self,n,random_state)
  end function
  function draw_samples(self,n,rng) result(x)
    class(kernel_density), intent(in) :: self
    integer, intent(in) :: n
    type(legacy_random_state), intent(inout) :: rng
    real(dp), allocatable :: x(:,:),uniforms(:),cumulative(:),direction(:)
    real(dp) :: target,squared,correction,cdf
    integer, allocatable :: selected(:)
    integer :: i,j,left,right,middle,d
    d=self%n_features_in_;allocate(x(n,d),uniforms(n),selected(n),direction(d))
    do i=1,n
      uniforms(i)=rng%uniform()
    end do
    if(self%weighted_) then
      allocate(cumulative(self%n_samples_fit_));target=0
      do i=1,size(cumulative)
        target=plus(target,self%sample_weight_(i));cumulative(i)=target
      end do
      do i=1,n
        target=ieee_product(uniforms(i),cumulative(size(cumulative)))
        call require(.not.ieee_is_nan(target),'density mixture draw lies outside fitted samples')
        left=1;right=size(cumulative)
        do while(left<right)
          middle=left+(right-left)/2
          if(cumulative(middle)>=target) then
            right=middle
          else
            left=middle+1
          end if
        end do
        selected(i)=left
      end do
    else
      selected=int(uniforms*self%n_samples_fit_)+1
    end if
    do i=1,n
      do j=1,d
        direction(j)=rng%normal()
      end do
      if(self%kernel=='gaussian') then
        do j=1,d
          x(i,j)=plus(self%x_(selected(i),j),ieee_product(self%bandwidth_,direction(j)))
        end do
      else
        squared=sum(direction**2);cdf=gamma_cdf(.5_dp*d,.5_dp*squared)
        correction=ieee_quotient(ieee_product(cdf**(1._dp/d),self%bandwidth_),sqrt(squared))
        do j=1,d
          x(i,j)=plus(self%x_(selected(i),j),ieee_product(direction(j),correction))
        end do
      end if
    end do
  end function
end module
