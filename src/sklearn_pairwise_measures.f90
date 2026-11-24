! sklearn 1.6.1 metrics/pairwise.py, _pairwise_fast.pyx and _dist_metrics.pyx.tp (BSD).
! Native interfaces, canonical CSR adapters and quiet IEEE composition are MIT.
module sklearn_pairwise_measures
  use sklearn_common, only: dp,require
  use sklearn_sparse, only: csr_matrix
  use sklearn_ieee_math, only: ieee_difference,ieee_product,ieee_quotient,ieee_total
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value,ieee_quiet_nan
  implicit none
  private
  public :: cosine_distances
  public :: nan_euclidean_distances
  public :: haversine_distances
  public :: paired_euclidean_distances
  public :: paired_manhattan_distances
  public :: paired_cosine_distances
  public :: paired_distances
  public :: laplacian_kernel
  public :: additive_chi2_kernel
  public :: chi2_kernel
  interface cosine_distances
    module procedure cosine_distances_dd
    module procedure cosine_distances_ss
    module procedure cosine_distances_sd
    module procedure cosine_distances_ds
  end interface
  interface haversine_distances
    module procedure haversine_distances_dd
    module procedure haversine_distances_ss
    module procedure haversine_distances_sd
    module procedure haversine_distances_ds
  end interface
  interface laplacian_kernel
    module procedure laplacian_kernel_dd
    module procedure laplacian_kernel_ss
    module procedure laplacian_kernel_sd
    module procedure laplacian_kernel_ds
  end interface
  interface paired_euclidean_distances
    module procedure paired_euclidean_distances_dd
    module procedure paired_euclidean_distances_ss
    module procedure paired_euclidean_distances_sd
    module procedure paired_euclidean_distances_ds
  end interface
  interface paired_manhattan_distances
    module procedure paired_manhattan_distances_dd
    module procedure paired_manhattan_distances_ss
    module procedure paired_manhattan_distances_sd
    module procedure paired_manhattan_distances_ds
  end interface
  interface paired_cosine_distances
    module procedure paired_cosine_distances_dd
    module procedure paired_cosine_distances_ss
    module procedure paired_cosine_distances_sd
    module procedure paired_cosine_distances_ds
  end interface
  interface paired_distances
    module procedure paired_distances_dd
    module procedure paired_callback_dd
  end interface
  abstract interface
    function row_distance(x,y) result(value)
      import dp
      real(dp), intent(in) :: x(:),y(:)
      real(dp) :: value
    end function
  end interface
contains
  real(dp) function nan_value() result(value)
    value=ieee_value(0._dp,ieee_quiet_nan)
  end function
  subroutine check_pair(x,y,paired)
    real(dp), intent(in) :: x(:,:),y(:,:)
    logical, intent(in) :: paired
    call require(size(x,1)>0.and.size(y,1)>0.and.size(x,2)>0.and.size(y,2)>0, &
      'pairwise measures require nonempty samples and features')
    call require(size(x,2)==size(y,2),'pairwise measure feature counts differ')
    if(paired) call require(size(x,1)==size(y,1),'paired measures require equal sample counts')
    call require(all(ieee_is_finite(x)).and.all(ieee_is_finite(y)),'pairwise inputs must be finite')
  end subroutine
  function dense_csr(matrix) result(values)
    type(csr_matrix), intent(in) :: matrix
    real(dp), allocatable :: values(:,:)
    integer :: i,start,stop
    call matrix%validate()
    call require(all(ieee_is_finite(matrix%data)),'pairwise CSR data must be finite')
    do i=1,matrix%n_rows
      start=matrix%indptr(i);stop=matrix%indptr(i+1)-1
      if(stop>start) call require(all(matrix%indices(start+1:stop)>matrix%indices(start:stop-1)), &
        'pairwise CSR indices must be sorted and unique')
    end do
    values=matrix%to_dense()
  end function
  real(dp) function dot_quiet(x,y) result(value)
    real(dp), intent(in) :: x(:),y(:)
    integer :: k
    value=0
    do k=1,size(x)
      value=ieee_difference(value,-ieee_product(x(k),y(k)))
    end do
  end function
  function normalized(x,sparse) result(values)
    real(dp), intent(in) :: x(:,:)
    logical, intent(in) :: sparse
    real(dp) :: values(size(x,1),size(x,2)),norm
    integer :: i,j
    values=x
    do i=1,size(x,1)
      norm=sqrt(dot_quiet(x(i,:),x(i,:)))
      if(sparse) then
        if(norm==0) norm=1
      else
        if(norm<10*epsilon(1._dp)) norm=1
      end if
      do j=1,size(x,2)
        values(i,j)=ieee_quotient(x(i,j),norm)
      end do
    end do
  end function
  function paired_core(x,y,kind,sparse_x,sparse_y) result(values)
    real(dp), intent(in) :: x(:,:),y(:,:)
    integer, intent(in) :: kind
    logical, intent(in) :: sparse_x,sparse_y
    real(dp), allocatable :: values(:),u(:,:),v(:,:)
    real(dp) :: difference,total
    integer :: i,j
    call check_pair(x,y,.true.);allocate(values(size(x,1)))
    if(kind==3) then
      u=normalized(x,sparse_x);v=normalized(y,sparse_y)
    else
      u=x;v=y
    end if
    do i=1,size(x,1)
      total=0
      do j=1,size(x,2)
        difference=ieee_difference(u(i,j),v(i,j))
        if(kind==2) then
          total=ieee_difference(total,-abs(difference))
        else
          total=ieee_difference(total,-ieee_product(difference,difference))
        end if
      end do
      values(i)=total
      if(kind==1) values(i)=sqrt(total)
      if(kind==3) values(i)=ieee_product(.5_dp,total)
    end do
  end function
  integer function paired_kind(metric) result(kind)
    character(*), optional, intent(in) :: metric
    character(:), allocatable :: name
    name='euclidean';if(present(metric)) name=trim(metric)
    kind=0
    select case(name)
    case('euclidean','l2');kind=1
    case('manhattan','cityblock','l1');kind=2
    case('cosine');kind=3
    end select
    call require(kind>0,'unknown paired distance metric')
  end function
  real(dp) function quiet_sin(x) result(value)
    real(dp), intent(in) :: x
    value=nan_value();if(ieee_is_finite(x)) value=sin(x)
  end function
  real(dp) function haversine(x,y) result(value)
    real(dp), intent(in) :: x(:),y(:)
    real(dp) :: first,second,a,part
    first=quiet_sin(ieee_product(.5_dp,ieee_difference(x(1),y(1))))
    second=quiet_sin(ieee_product(.5_dp,ieee_difference(x(2),y(2))))
    part=ieee_product(ieee_product(cos(x(1)),cos(y(1))),second)
    a=ieee_difference(ieee_product(first,first),-ieee_product(part,second))
    value=nan_value()
    if(ieee_is_nan(a)) return
    ! Source asin is not clipped: preserve antipodal roundoff NaNs.
    if(a>=0.and.a<=1) value=2*asin(sqrt(a))
  end function
  function pairwise_core(x,y,kind,gamma,sparse_x,sparse_y) result(values)
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: y(:,:),gamma
    integer, intent(in) :: kind
    logical, intent(in) :: sparse_x,sparse_y
    real(dp), allocatable :: values(:,:),other(:,:),u(:,:),v(:,:)
    real(dp) :: g,value,total
    integer :: i,j,k
    other=x;if(present(y)) other=y
    call check_pair(x,other,.false.)
    if(kind==2) call require(size(x,2)==2,'haversine requires latitude and longitude columns')
    g=1._dp/size(x,2);if(present(gamma)) g=gamma
    if(kind==3) then
      call require(ieee_is_finite(g),'Laplacian gamma must be finite')
      call require(g>0,'Laplacian gamma must be positive')
    end if
    allocate(values(size(x,1),size(other,1)))
    if(kind==1) then
      u=normalized(x,sparse_x);v=normalized(other,sparse_y)
    end if
    do j=1,size(other,1)
      do i=1,size(x,1)
        select case(kind)
        case(1)
          value=ieee_difference(1._dp,dot_quiet(u(i,:),v(j,:)))
          if(.not.ieee_is_nan(value)) value=max(0._dp,min(2._dp,value))
        case(2)
          value=haversine(x(i,:),other(j,:))
        case(3)
          total=0
          do k=1,size(x,2)
            total=ieee_difference(total,-abs(ieee_difference(x(i,k),other(j,k))))
          end do
          value=exp(-ieee_product(g,total))
        end select
        values(i,j)=value
      end do
    end do
    if(.not.present(y).and.kind==1) then
      do i=1,size(x,1)
        values(i,i)=0
      end do
    end if
  end function
  function additive_chi2_kernel(x,y) result(values)
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: y(:,:)
    real(dp), allocatable :: values(:,:),other(:,:)
    real(dp) :: total,difference,denom
    integer :: i,j,k
    other=x;if(present(y)) other=y
    call check_pair(x,other,.false.)
    call require(all(x>=0).and.all(other>=0),'chi-squared kernel inputs must be nonnegative')
    allocate(values(size(x,1),size(other,1)))
    do j=1,size(other,1)
      do i=1,size(x,1)
        total=0
        do k=1,size(x,2)
          difference=ieee_difference(x(i,k),other(j,k));denom=ieee_difference(x(i,k),-other(j,k))
          if(denom/=0) total=ieee_difference(total,-ieee_quotient(ieee_product(difference,difference),denom))
        end do
        values(i,j)=-total
      end do
    end do
  end function
  function chi2_kernel(x,y,gamma) result(values)
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: y(:,:),gamma
    real(dp), allocatable :: values(:,:)
    real(dp) :: g
    integer :: i,j
    g=1;if(present(gamma)) g=gamma
    call require(ieee_is_finite(g),'chi-squared gamma must be finite')
    call require(g>0,'chi-squared gamma must be positive')
    values=additive_chi2_kernel(x,y)
    do j=1,size(values,2)
      do i=1,size(values,1)
        values(i,j)=exp(ieee_product(g,values(i,j)))
      end do
    end do
  end function
  function nan_euclidean_distances(x,y,squared,missing_values) result(values)
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: y(:,:),missing_values
    logical, optional, intent(in) :: squared
    real(dp), allocatable :: values(:,:),u(:,:),v(:,:),xx(:,:),yy(:,:),nx(:),ny(:)
    logical, allocatable :: mx(:,:),my(:,:)
    real(dp) :: marker,value,first,second
    logical :: nan_marker,square
    integer :: i,j,k,p,count_
    marker=nan_value();if(present(missing_values)) marker=missing_values
    nan_marker=ieee_is_nan(marker);square=.false.;if(present(squared)) square=squared
    u=x;v=x;if(present(y)) v=y
    call require(size(u,1)>0.and.size(v,1)>0.and.size(u,2)>0.and.size(v,2)>0, &
      'NaN Euclidean requires nonempty samples and features')
    call require(size(u,2)==size(v,2),'NaN Euclidean feature counts differ')
    if(nan_marker) then
      call require(all(ieee_is_finite(u).or.ieee_is_nan(u)).and.all(ieee_is_finite(v).or.ieee_is_nan(v)), &
        'NaN Euclidean input contains infinity')
      mx=ieee_is_nan(u);my=ieee_is_nan(v)
    else
      call require(all(ieee_is_finite(u)).and.all(ieee_is_finite(v)),'NaN Euclidean non-NaN marker requires finite input')
      mx=u==marker;my=v==marker
    end if
    where(mx) u=0
    where(my) v=0
    p=size(u,2);allocate(xx(size(u,1),p),yy(size(v,1),p),nx(size(u,1)),ny(size(v,1)))
    do i=1,size(u,1)
      nx(i)=dot_quiet(u(i,:),u(i,:))
      do k=1,p
        xx(i,k)=ieee_product(u(i,k),u(i,k))
      end do
    end do
    do j=1,size(v,1)
      ny(j)=dot_quiet(v(j,:),v(j,:))
      do k=1,p
        yy(j,k)=ieee_product(v(j,k),v(j,k))
      end do
    end do
    allocate(values(size(u,1),size(v,1)))
    do j=1,size(v,1)
      do i=1,size(u,1)
        value=ieee_product(-2._dp,dot_quiet(u(i,:),v(j,:)))
        value=ieee_difference(ieee_difference(value,-nx(i)),-ny(j))
        if(.not.ieee_is_nan(value)) value=max(0._dp,value)
        first=dot_quiet(xx(i,:),real(merge(1,0,my(j,:)),dp))
        second=dot_quiet(real(merge(1,0,mx(i,:)),dp),yy(j,:))
        value=ieee_difference(ieee_difference(value,first),second)
        if(.not.ieee_is_nan(value)) value=max(0._dp,value)
        if(.not.present(y).and.i==j) value=0
        count_=count(.not.mx(i,:).and..not.my(j,:))
        if(count_==0) then
          value=nan_value()
        else
          value=ieee_product(ieee_quotient(value,real(count_,dp)),real(p,dp))
        end if
        if(.not.square.and..not.ieee_is_nan(value)) value=sqrt(value)
        values(i,j)=value
      end do
    end do
  end function
  function cosine_distances_dd(x,y) result(values)
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: y(:,:)
    real(dp), allocatable :: values(:,:)
    values=pairwise_core(x,y=y,kind=1,sparse_x=.false.,sparse_y=.false.)
  end function
  function cosine_distances_ss(x,y) result(values)
    type(csr_matrix), intent(in) :: x
    type(csr_matrix), optional, intent(in) :: y
    real(dp), allocatable :: values(:,:)
    real(dp), allocatable :: a(:,:)
    real(dp), allocatable :: b(:,:)
    a=dense_csr(x)
    if(present(y)) b=dense_csr(y)
    values=pairwise_core(a,y=b,kind=1,sparse_x=.true.,sparse_y=.true.)
  end function
  function cosine_distances_sd(x,y) result(values)
    type(csr_matrix), intent(in) :: x
    real(dp), intent(in) :: y(:,:)
    real(dp), allocatable :: values(:,:)
    real(dp), allocatable :: a(:,:)
    a=dense_csr(x)
    values=pairwise_core(a,y=y,kind=1,sparse_x=.true.,sparse_y=.false.)
  end function
  function cosine_distances_ds(x,y) result(values)
    real(dp), intent(in) :: x(:,:)
    type(csr_matrix), intent(in) :: y
    real(dp), allocatable :: values(:,:)
    real(dp), allocatable :: b(:,:)
    b=dense_csr(y)
    values=pairwise_core(x,y=b,kind=1,sparse_x=.false.,sparse_y=.true.)
  end function
  function haversine_distances_dd(x,y) result(values)
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: y(:,:)
    real(dp), allocatable :: values(:,:)
    values=pairwise_core(x,y=y,kind=2,sparse_x=.false.,sparse_y=.false.)
  end function
  function haversine_distances_ss(x,y) result(values)
    type(csr_matrix), intent(in) :: x
    type(csr_matrix), optional, intent(in) :: y
    real(dp), allocatable :: values(:,:)
    real(dp), allocatable :: a(:,:)
    real(dp), allocatable :: b(:,:)
    a=dense_csr(x)
    if(present(y)) b=dense_csr(y)
    values=pairwise_core(a,y=b,kind=2,sparse_x=.true.,sparse_y=.true.)
  end function
  function haversine_distances_sd(x,y) result(values)
    type(csr_matrix), intent(in) :: x
    real(dp), intent(in) :: y(:,:)
    real(dp), allocatable :: values(:,:)
    real(dp), allocatable :: a(:,:)
    a=dense_csr(x)
    values=pairwise_core(a,y=y,kind=2,sparse_x=.true.,sparse_y=.false.)
  end function
  function haversine_distances_ds(x,y) result(values)
    real(dp), intent(in) :: x(:,:)
    type(csr_matrix), intent(in) :: y
    real(dp), allocatable :: values(:,:)
    real(dp), allocatable :: b(:,:)
    b=dense_csr(y)
    values=pairwise_core(x,y=b,kind=2,sparse_x=.false.,sparse_y=.true.)
  end function
  function laplacian_kernel_dd(x,y,gamma) result(values)
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: y(:,:)
    real(dp), optional, intent(in) :: gamma
    real(dp), allocatable :: values(:,:)
    values=pairwise_core(x,y=y,kind=3,sparse_x=.false.,sparse_y=.false.,gamma=gamma)
  end function
  function laplacian_kernel_ss(x,y,gamma) result(values)
    type(csr_matrix), intent(in) :: x
    type(csr_matrix), optional, intent(in) :: y
    real(dp), optional, intent(in) :: gamma
    real(dp), allocatable :: values(:,:)
    real(dp), allocatable :: a(:,:)
    real(dp), allocatable :: b(:,:)
    a=dense_csr(x)
    if(present(y)) b=dense_csr(y)
    values=pairwise_core(a,y=b,kind=3,sparse_x=.true.,sparse_y=.true.,gamma=gamma)
  end function
  function laplacian_kernel_sd(x,y,gamma) result(values)
    type(csr_matrix), intent(in) :: x
    real(dp), intent(in) :: y(:,:)
    real(dp), optional, intent(in) :: gamma
    real(dp), allocatable :: values(:,:)
    real(dp), allocatable :: a(:,:)
    a=dense_csr(x)
    values=pairwise_core(a,y=y,kind=3,sparse_x=.true.,sparse_y=.false.,gamma=gamma)
  end function
  function laplacian_kernel_ds(x,y,gamma) result(values)
    real(dp), intent(in) :: x(:,:)
    type(csr_matrix), intent(in) :: y
    real(dp), optional, intent(in) :: gamma
    real(dp), allocatable :: values(:,:)
    real(dp), allocatable :: b(:,:)
    b=dense_csr(y)
    values=pairwise_core(x,y=b,kind=3,sparse_x=.false.,sparse_y=.true.,gamma=gamma)
  end function
  function paired_euclidean_distances_dd(x,y) result(values)
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(in) :: y(:,:)
    real(dp), allocatable :: values(:)
    values=paired_core(x,y,1,.false.,.false.)
  end function
  function paired_euclidean_distances_ss(x,y) result(values)
    type(csr_matrix), intent(in) :: x
    type(csr_matrix), intent(in) :: y
    real(dp), allocatable :: values(:)
    real(dp), allocatable :: a(:,:)
    real(dp), allocatable :: b(:,:)
    a=dense_csr(x)
    b=dense_csr(y)
    values=paired_core(a,b,1,.true.,.true.)
  end function
  function paired_euclidean_distances_sd(x,y) result(values)
    type(csr_matrix), intent(in) :: x
    real(dp), intent(in) :: y(:,:)
    real(dp), allocatable :: values(:)
    real(dp), allocatable :: a(:,:)
    a=dense_csr(x)
    values=paired_core(a,y,1,.true.,.false.)
  end function
  function paired_euclidean_distances_ds(x,y) result(values)
    real(dp), intent(in) :: x(:,:)
    type(csr_matrix), intent(in) :: y
    real(dp), allocatable :: values(:)
    real(dp), allocatable :: b(:,:)
    b=dense_csr(y)
    values=paired_core(x,b,1,.false.,.true.)
  end function
  function paired_manhattan_distances_dd(x,y) result(values)
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(in) :: y(:,:)
    real(dp), allocatable :: values(:)
    values=paired_core(x,y,2,.false.,.false.)
  end function
  function paired_manhattan_distances_ss(x,y) result(values)
    type(csr_matrix), intent(in) :: x
    type(csr_matrix), intent(in) :: y
    real(dp), allocatable :: values(:)
    real(dp), allocatable :: a(:,:)
    real(dp), allocatable :: b(:,:)
    a=dense_csr(x)
    b=dense_csr(y)
    values=paired_core(a,b,2,.true.,.true.)
  end function
  function paired_manhattan_distances_sd(x,y) result(values)
    type(csr_matrix), intent(in) :: x
    real(dp), intent(in) :: y(:,:)
    real(dp), allocatable :: values(:)
    real(dp), allocatable :: a(:,:)
    a=dense_csr(x)
    values=paired_core(a,y,2,.true.,.false.)
  end function
  function paired_manhattan_distances_ds(x,y) result(values)
    real(dp), intent(in) :: x(:,:)
    type(csr_matrix), intent(in) :: y
    real(dp), allocatable :: values(:)
    real(dp), allocatable :: b(:,:)
    b=dense_csr(y)
    values=paired_core(x,b,2,.false.,.true.)
  end function
  function paired_cosine_distances_dd(x,y) result(values)
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(in) :: y(:,:)
    real(dp), allocatable :: values(:)
    values=paired_core(x,y,3,.false.,.false.)
  end function
  function paired_cosine_distances_ss(x,y) result(values)
    type(csr_matrix), intent(in) :: x
    type(csr_matrix), intent(in) :: y
    real(dp), allocatable :: values(:)
    real(dp), allocatable :: a(:,:)
    real(dp), allocatable :: b(:,:)
    a=dense_csr(x)
    b=dense_csr(y)
    values=paired_core(a,b,3,.true.,.true.)
  end function
  function paired_cosine_distances_sd(x,y) result(values)
    type(csr_matrix), intent(in) :: x
    real(dp), intent(in) :: y(:,:)
    real(dp), allocatable :: values(:)
    real(dp), allocatable :: a(:,:)
    a=dense_csr(x)
    values=paired_core(a,y,3,.true.,.false.)
  end function
  function paired_cosine_distances_ds(x,y) result(values)
    real(dp), intent(in) :: x(:,:)
    type(csr_matrix), intent(in) :: y
    real(dp), allocatable :: values(:)
    real(dp), allocatable :: b(:,:)
    b=dense_csr(y)
    values=paired_core(x,b,3,.false.,.true.)
  end function
  function paired_distances_dd(x,y,metric) result(values)
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(in) :: y(:,:)
    character(*), optional, intent(in) :: metric
    real(dp), allocatable :: values(:)
    values=paired_core(x,y,paired_kind(metric),.false.,.false.)
  end function
  function paired_callback_dd(x,y,metric) result(values)
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(in) :: y(:,:)
    procedure(row_distance) :: metric
    real(dp), allocatable :: values(:),a(:,:),b(:,:)
    integer :: i
    a=x
    b=y
    call check_pair(a,b,.true.);allocate(values(size(a,1)))
    do i=1,size(a,1)
      values(i)=metric(a(i,:),b(i,:))
    end do
  end function
end module
