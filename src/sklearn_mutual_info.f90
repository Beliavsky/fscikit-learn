! sklearn 1.6.1 feature_selection/_mutual_info.py and
! _univariate_selection.py, metrics/cluster/_supervised.py (BSD-3-Clause).
! Kraskov, Stogbauer & Grassberger (2004); Ross (2014).
! See THIRD_PARTY_NOTICES.md and retained upstream licenses.
module sklearn_mutual_info
  use sklearn_common, only: dp,require,check_xy,check_x
  use sklearn_legacy_rng, only: legacy_random_state
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value, &
    ieee_quiet_nan,ieee_positive_inf,ieee_next_after
  implicit none
  private
  public :: mutual_info_classif,mutual_info_regression,r_regression,mutual_info_score
  interface mutual_info_classif
    module procedure classif_mask,classif_scalar,classif_indices
  end interface
  interface mutual_info_regression
    module procedure regression_mask,regression_scalar,regression_indices
  end interface
contains
  recursive real(dp) function numpy_sum(values) result(total)
    ! NumPy 2.2.6 _core/src/umath/loops_utils.h.src real pairwise reduction.
    ! BSD notice retained in LICENSES/numpy-COPYING.
    real(dp), intent(in) :: values(:)
    real(dp) :: partial(8)
    integer :: n,i,split,whole
    n=size(values)
    if(n<8) then
      total=-0._dp
      do i=1,n
        total=total+values(i)
      end do
    else if(n<=128) then
      partial=values(:8); whole=n-mod(n,8)
      do i=9,whole,8
        partial=partial+values(i:i+7)
      end do
      total=((partial(1)+partial(2))+(partial(3)+partial(4))) &
        +((partial(5)+partial(6))+(partial(7)+partial(8)))
      do i=whole+1,n
        total=total+values(i)
      end do
    else
      split=n/2; split=split-mod(split,8)
      total=numpy_sum(values(:split))+numpy_sum(values(split+1:))
    end if
  end function
  function classif_mask(x,y,discrete_features,n_neighbors,random_state) result(scores)
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    logical, intent(in), optional :: discrete_features(:)
    integer, intent(in), optional :: n_neighbors,random_state
    real(dp), allocatable :: scores(:)
    scores=estimate_mi(x,real(y,dp),.true.,discrete_features,n_neighbors,random_state)
  end function
  function classif_scalar(x,y,discrete_features,n_neighbors,random_state) result(scores)
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    logical, intent(in) :: discrete_features
    integer, intent(in), optional :: n_neighbors,random_state
    real(dp), allocatable :: scores(:)
    scores=classif_mask(x,y,spread(discrete_features,1,size(x,2)),n_neighbors,random_state)
  end function
  function classif_indices(x,y,discrete_features,n_neighbors,random_state) result(scores)
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:),discrete_features(:)
    integer, intent(in), optional :: n_neighbors,random_state
    real(dp), allocatable :: scores(:)
    scores=classif_mask(x,y,index_mask(size(x,2),discrete_features),n_neighbors,random_state)
  end function
  function regression_mask(x,y,discrete_features,n_neighbors,random_state) result(scores)
    real(dp), intent(in) :: x(:,:),y(:)
    logical, intent(in), optional :: discrete_features(:)
    integer, intent(in), optional :: n_neighbors,random_state
    real(dp), allocatable :: scores(:)
    scores=estimate_mi(x,y,.false.,discrete_features,n_neighbors,random_state)
  end function
  function regression_scalar(x,y,discrete_features,n_neighbors,random_state) result(scores)
    real(dp), intent(in) :: x(:,:),y(:)
    logical, intent(in) :: discrete_features
    integer, intent(in), optional :: n_neighbors,random_state
    real(dp), allocatable :: scores(:)
    scores=regression_mask(x,y,spread(discrete_features,1,size(x,2)),n_neighbors,random_state)
  end function
  function regression_indices(x,y,discrete_features,n_neighbors,random_state) result(scores)
    real(dp), intent(in) :: x(:,:),y(:)
    integer, intent(in) :: discrete_features(:)
    integer, intent(in), optional :: n_neighbors,random_state
    real(dp), allocatable :: scores(:)
    scores=regression_mask(x,y,index_mask(size(x,2),discrete_features),n_neighbors,random_state)
  end function
  function index_mask(p,indices) result(mask)
    integer, intent(in) :: p,indices(:)
    logical :: mask(p)
    integer :: i
    call require(size(indices)>0,'MI discrete feature indices must be nonempty')
    call require(all(indices>=1).and.all(indices<=p),'MI discrete feature index outside [1,P]')
    mask=.false.
    do i=1,size(indices)
      mask(indices(i))=.true.
    end do
  end function
  subroutine scale_column(values,array_scale)
    real(dp), intent(inout) :: values(:)
    logical, intent(in) :: array_scale
    real(dp) :: deviation,mean
    mean=numpy_sum(values)/size(values); deviation=sqrt(numpy_sum((values-mean)**2)/size(values))
    if(array_scale) then
      if(deviation<10*epsilon(1._dp)) deviation=1
    else
      if(deviation==0) deviation=1
    end if
    values=values/deviation
  end subroutine
  function estimate_mi(x,y,discrete_target,discrete_features,n_neighbors,random_state) result(scores)
    real(dp), intent(in) :: x(:,:),y(:)
    logical, intent(in) :: discrete_target
    logical, intent(in), optional :: discrete_features(:)
    integer, intent(in), optional :: n_neighbors,random_state
    real(dp), allocatable :: scores(:),work(:,:),target(:),noise_scale(:),harmonic(:)
    logical :: discrete(size(x,2))
    type(legacy_random_state) :: rng
    integer :: k,seed,n,p,i,j
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    n=size(y); p=size(x,2); k=3; seed=0
    if(present(n_neighbors)) k=n_neighbors
    if(present(random_state)) seed=random_state
    call require(k>0,'MI n_neighbors must be positive')
    call require(seed>=0,'MI seed must be nonnegative')
    discrete=.false.
    if(present(discrete_features)) then
      call require(size(discrete_features)==p,'MI discrete feature mask length mismatch')
      discrete=discrete_features
    end if
    if(.not.discrete_target.and.any(.not.discrete)) call require(k<n,'MI continuous neighbors must be less than N')
    work=x; target=y; allocate(scores(p),noise_scale(p),harmonic(n)); noise_scale=0
    harmonic(1)=0
    do i=2,n
      harmonic(i)=harmonic(i-1)+1._dp/(i-1)
    end do
    call rng%seed(seed)
    do j=1,p
      if(discrete(j)) cycle
      call scale_column(work(:,j),.true.)
      noise_scale(j)=1.e-10_dp*max(1._dp,numpy_sum(abs(work(:,j)))/n)
    end do
    ! NumPy's (N,continuous_features) Gaussian array is drawn in C order.
    do i=1,n
      do j=1,p
        if(.not.discrete(j)) work(i,j)=work(i,j)+noise_scale(j)*rng%normal()
      end do
    end do
    if(.not.discrete_target) then
      call scale_column(target,.false.)
      noise_scale(1)=1.e-10_dp*max(1._dp,numpy_sum(abs(target))/n)
      do i=1,n
        target(i)=target(i)+noise_scale(1)*rng%normal()
      end do
    end if
    call require(all(ieee_is_finite(work)).and.all(ieee_is_finite(target)),'MI scaled data must be finite')
    do j=1,p
      if(discrete(j).and.discrete_target) then
        scores(j)=discrete_mi(work(:,j),target)
      else if(discrete(j)) then
        scores(j)=continuous_discrete(target,work(:,j),k,harmonic)
      else if(discrete_target) then
        scores(j)=continuous_discrete(work(:,j),target,k,harmonic)
      else
        scores(j)=continuous_mi(work(:,j),target,k,harmonic)
      end if
    end do
  end function
  real(dp) function kth_distance(values,k) result(value)
    real(dp), intent(inout) :: values(:)
    integer, intent(in) :: k
    integer :: left,right,i,j
    real(dp) :: pivot,temp
    left=1; right=size(values)
    do while(left<right)
      pivot=values(left+(right-left)/2); i=left; j=right
      do
        do while(values(i)<pivot)
          i=i+1
        end do
        do while(values(j)>pivot)
          j=j-1
        end do
        if(i>j) exit
        temp=values(i); values(i)=values(j); values(j)=temp; i=i+1; j=j-1
        if(i>j) exit
      end do
      if(k<=j) then
        right=j
      else if(k>=i) then
        left=i
      else
        exit
      end if
    end do
    value=values(k)
  end function
  real(dp) function continuous_mi(x,y,k,harmonic) result(value)
    real(dp), intent(in) :: x(:),y(:),harmonic(:)
    integer, intent(in) :: k
    real(dp) :: distances(size(x)-1),radius,total_x,total_y
    integer :: n,i,j,q,nx,ny
    n=size(x); total_x=0; total_y=0
    do i=1,n
      q=0
      do j=1,n
        if(i==j) cycle
        q=q+1; distances(q)=max(abs(x(i)-x(j)),abs(y(i)-y(j)))
      end do
      radius=kth_distance(distances,k); radius=ieee_next_after(radius,0._dp)
      nx=count(abs(x-x(i))<=radius); ny=count(abs(y-y(i))<=radius)
      total_x=total_x+harmonic(nx); total_y=total_y+harmonic(ny)
    end do
    value=max(0._dp,harmonic(n)+harmonic(k)-total_x/n-total_y/n)
  end function
  subroutine category_codes(values,codes,frequencies)
    real(dp), intent(in) :: values(:)
    integer, allocatable, intent(out) :: codes(:),frequencies(:)
    real(dp) :: levels(size(values))
    integer :: i,j,nlevels
    allocate(codes(size(values)),frequencies(size(values))); frequencies=0; nlevels=0
    do i=1,size(values)
      do j=1,nlevels
        if(values(i)==levels(j)) exit
      end do
      if(j>nlevels) then
        nlevels=j; levels(j)=values(i)
      end if
      codes(i)=j; frequencies(j)=frequencies(j)+1
    end do
    frequencies=frequencies(:nlevels)
  end subroutine
  real(dp) function continuous_discrete(c,d,k,harmonic) result(value)
    real(dp), intent(in) :: c(:),d(:),harmonic(:)
    integer, intent(in) :: k
    integer, allocatable :: codes(:),frequencies(:)
    real(dp) :: distances(size(c)-1),radius,total,squared,middle
    integer :: i,j,q,ki,mi,retained
    call category_codes(d,codes,frequencies)
    retained=count(frequencies(codes)>1)
    call require(retained>0,'MI continuous-discrete estimation needs repeated labels')
    total=0
    do i=1,size(c)
      if(frequencies(codes(i))==1) cycle
      ki=min(k,frequencies(codes(i))-1); q=0
      do j=1,size(c)
        if(i==j.or.codes(i)/=codes(j)) cycle
        q=q+1
        if(ki>=frequencies(codes(i))/2) then
          ! Source auto-selects brute force here: ArgKmin's ordered Gram terms.
          squared=c(i)*c(i); middle=-2*c(i)*c(j); squared=squared+middle
          squared=squared+c(j)*c(j); distances(q)=sqrt(max(0._dp,squared))
        else
          distances(q)=sqrt((c(i)-c(j))**2)
        end if
      end do
      radius=kth_distance(distances(:q),ki); radius=ieee_next_after(radius,0._dp)
      mi=count((c-c(i))**2<=radius**2.and.frequencies(codes)>1)
      total=total+harmonic(ki)-harmonic(frequencies(codes(i)))-harmonic(mi)
    end do
    value=max(0._dp,harmonic(retained)+total/retained)
  end function
  real(dp) function discrete_mi(x,y) result(value)
    real(dp), intent(in) :: x(:),y(:)
    integer, allocatable :: cx(:),cy(:),fx(:),fy(:)
    integer :: px(size(x)),py(size(x)),joint(size(x)),i,j,n,pairs
    real(dp) :: mass,term,logn
    value=0; n=size(x)
    if(n==0) return
    call category_codes(x,cx,fx); call category_codes(y,cy,fy)
    if(size(fx)==1.or.size(fy)==1) return
    pairs=0; joint=0; logn=log(real(n,dp))
    do i=1,n
      do j=1,pairs
        if(px(j)==cx(i).and.py(j)==cy(i)) exit
      end do
      if(j>pairs) then
        pairs=j; px(j)=cx(i); py(j)=cy(i)
      end if
      joint(j)=joint(j)+1
    end do
    do j=1,pairs
      mass=real(joint(j),dp)/n
      term=mass*(log(real(joint(j),dp))-logn) &
        +mass*(-log(real(fx(px(j)),dp)*real(fy(py(j)),dp))+2*logn)
      if(abs(term)>=epsilon(1._dp)) value=value+term
    end do
    value=max(0._dp,value)
  end function
  real(dp) function mutual_info_score(labels_true,labels_pred) result(value)
    integer, intent(in) :: labels_true(:),labels_pred(:)
    call require(size(labels_true)==size(labels_pred),'mutual_info_score label count mismatch')
    call require(size(labels_true)>0,'mutual_info_score labels must be nonempty')
    value=discrete_mi(real(labels_true,dp),real(labels_pred,dp))
  end function
  function r_regression(x,y,center,force_finite) result(scores)
    real(dp), intent(in) :: x(:,:),y(:)
    logical, intent(in), optional :: center,force_finite
    real(dp), allocatable :: scores(:)
    real(dp) :: target(size(y)),variance,numerator,norm_y,value
    logical :: centered,finite
    integer :: j,n
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    centered=.true.; finite=.true.
    if(present(center)) centered=center
    if(present(force_finite)) finite=force_finite
    n=size(y); target=y
    if(centered) target=target-numpy_sum(target)/n
    norm_y=sqrt(sum(target**2)); allocate(scores(size(x,2)))
    do j=1,size(x,2)
      variance=sum(x(:,j)**2)
      if(centered) variance=variance-real(n,dp)*(sum(x(:,j))/n)**2
      numerator=dot_product(target,x(:,j)); value=ieee_value(0._dp,ieee_quiet_nan)
      if(variance>=0) then
        if(variance>0) then
          value=numerator/sqrt(variance)
        else if(numerator/=0) then
          value=sign(ieee_value(0._dp,ieee_positive_inf),numerator)
        end if
        if(norm_y>0) then
          if(.not.ieee_is_nan(value)) value=value/norm_y
        else if(ieee_is_finite(value)) then
          if(value==0) then
            value=ieee_value(0._dp,ieee_quiet_nan)
          else
            value=sign(ieee_value(0._dp,ieee_positive_inf),value)
          end if
        end if
      end if
      ! Source force_finite replaces NaNs only; signed infinities survive.
      if(finite.and.ieee_is_nan(value)) value=0
      scores(j)=value
    end do
  end function
end module
