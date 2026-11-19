! KBinsDiscretizer contracts from sklearn 1.6.1 preprocessing/_discretization.py;
! weighted percentiles from utils/stats.py and Lloyd updates from cluster/.
! Retained BSD-3-Clause notice: LICENSES/scikit-learn-COPYING.
module sklearn_discretization
  use sklearn_common, only: dp,require,check_x,argsort
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_value,ieee_quiet_nan, &
    ieee_negative_inf,ieee_positive_inf
  implicit none
  private
  public :: kbins_discretizer
  type :: bin_edge_column
    real(dp), allocatable :: values(:)
  end type
  type :: kbins_discretizer
    integer :: n_bins=5,n_features_out_=0
    character(16) :: strategy='quantile',encode='onehot-dense'
    integer, allocatable :: n_bins_(:),n_iter_(:)
    type(bin_edge_column), allocatable :: bin_edges_(:)
    logical :: converged_=.false.
    character(16), private :: fitted_encode=''
  contains
    procedure :: fit=>bins_fit
    procedure :: transform=>bins_transform
    procedure :: fit_transform=>bins_fit_transform
    procedure :: inverse_transform=>bins_inverse
  end type
contains
  ! Weighted 1D Lloyd iterations with deterministic uniform-midpoint starts.
  subroutine bin_kmeans(x,w,k,edges,n_iter,converged)
    real(dp), intent(in) :: x(:),w(:)
    integer, intent(in) :: k
    real(dp), intent(out) :: edges(k+1)
    integer, intent(out) :: n_iter
    logical, intent(out) :: converged
    real(dp) :: col(size(x)),centers(k),next(k),mass(k),dist(size(x)),average,variance,shift
    integer :: labels(size(x)),previous(size(x)),order(k),far(size(x)),empty(k)
    integer :: i,j,it,pick,donor,n_empty,largest
    average=sum(x)/size(x); col=x-average; variance=sum(col**2)/size(x)
    do j=1,k
      centers(j)=minval(x)+(maxval(x)-minval(x))*(real(j,dp)-0.5_dp)/k-average
    end do
    previous=-1; converged=.false.
    do it=1,300
      next=0; mass=0
      do i=1,size(x)
        labels(i)=minloc((col(i)-centers)**2,dim=1)
        dist(i)=(col(i)-centers(labels(i)))**2
        next(labels(i))=next(labels(i))+w(i)*col(i)
        mass(labels(i))=mass(labels(i))+w(i)
      end do
      n_empty=0
      do j=1,k
        if(mass(j)>0) cycle
        n_empty=n_empty+1; empty(n_empty)=j
      end do
      if(n_empty>0.and.maxval(dist)>0) then
        far=argsort(dist)
        do j=1,n_empty
          pick=far(size(x)-j+1); donor=labels(pick)
          next(donor)=next(donor)-w(pick)*col(pick); mass(donor)=mass(donor)-w(pick)
          next(empty(j))=w(pick)*col(pick); mass(empty(j))=w(pick)
        end do
      end if
      largest=maxloc(mass,dim=1)
      do j=1,k
        if(mass(j)>0) next(j)=next(j)/mass(j)
      end do
      where(mass<=0) next=next(largest)
      shift=sum((next-centers)**2); centers=next
      if(all(labels==previous).or.shift<=1.e-4_dp*variance) then
        converged=.true.; exit
      end if
      previous=labels
    end do
    n_iter=min(it,300); order=argsort(centers); centers=centers(order)+average
    edges(1)=minval(x); edges(k+1)=maxval(x)
    edges(2:k)=centers(:k-1)*0.5_dp+centers(2:)*0.5_dp
  end subroutine

  subroutine bins_fit(self,x,n_bins,sample_weight)
    class(kbins_discretizer), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in), optional :: n_bins(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: edges(:),sorted(:),weights(:),cdf(:),w(:)
    real(dp) :: low,high,pos,fraction,target
    integer :: j,k,i,n,p,left,order(size(x,1))
    logical, allocatable :: keep(:)
    logical :: converged
    call check_x(x); n=size(x,1); p=size(x,2)
    call require(self%strategy=='uniform'.or.self%strategy=='quantile'.or.self%strategy=='kmeans', &
      'invalid KBinsDiscretizer strategy')
    call require(self%encode=='ordinal'.or.self%encode=='onehot-dense','invalid KBinsDiscretizer encode')
    self%n_bins_=spread(self%n_bins,1,p)
    if(present(n_bins)) then
      call require(size(n_bins)==p,'KBinsDiscretizer bin count shape mismatch')
      self%n_bins_=n_bins
    end if
    call require(all(self%n_bins_>=2),'KBinsDiscretizer requires at least two requested bins')
    w=spread(1._dp,1,n)
    if(present(sample_weight)) then
      call require(self%strategy/='uniform','uniform binning does not support sample weights')
      call require(size(sample_weight)==n,'KBinsDiscretizer weight count mismatch')
      call require(all(ieee_is_finite(sample_weight)),'invalid KBinsDiscretizer sample weights')
      call require(all(sample_weight>=0).and.any(sample_weight>0),'invalid KBinsDiscretizer sample weights')
      w=sample_weight
    end if
    if(allocated(self%bin_edges_)) deallocate(self%bin_edges_)
    allocate(self%bin_edges_(p)); self%n_iter_=spread(0,1,p); self%converged_=.true.
    self%fitted_encode=self%encode
    do j=1,p
      k=self%n_bins_(j); low=minval(x(:,j)); high=maxval(x(:,j))
      if(low==high) then
        self%n_bins_(j)=1
        self%bin_edges_(j)%values=[ieee_value(0._dp,ieee_negative_inf),ieee_value(0._dp,ieee_positive_inf)]
        cycle
      end if
      if(allocated(edges)) deallocate(edges)
      allocate(edges(k+1))
      select case(self%strategy)
      case('uniform')
        do i=0,k
          edges(i+1)=low+(high-low)*real(i,dp)/k
        end do
      case('quantile')
        order=argsort(x(:,j)); sorted=x(order,j)
        if(present(sample_weight)) then
          weights=w(order); cdf=weights
          do i=2,n
            cdf(i)=cdf(i-1)+weights(i)
          end do
          do i=0,k
            target=real(i,dp)/k*cdf(n); left=1
            do while(left<n)
              if(target==0) then
                if(cdf(left)>0) exit
              else
                if(cdf(left)>=target) exit
              end if
              left=left+1
            end do
            edges(i+1)=sorted(left)
          end do
        else
          do i=0,k
            pos=real(i,dp)/k*(n-1); left=min(n,1+int(pos)); fraction=pos-int(pos)
            edges(i+1)=sorted(left)*(1-fraction)+sorted(min(n,left+1))*fraction
          end do
        end if
      case('kmeans')
        call require(k<=n,'kmeans bin count exceeds sample count')
        call bin_kmeans(x(:,j),w,k,edges,self%n_iter_(j),converged)
        self%converged_=self%converged_.and.converged
      end select
      if(self%strategy/='uniform') then
        keep=spread(.true.,1,k+1); keep(2:)=edges(2:)-edges(:k)>1.e-8_dp
        edges=pack(edges,keep)
      end if
      self%n_bins_(j)=size(edges)-1
      call require(self%n_bins_(j)>0,'KBinsDiscretizer has no bins after width pruning')
      self%bin_edges_(j)%values=edges
    end do
    self%n_features_out_=p
    if(self%fitted_encode=='onehot-dense') self%n_features_out_=sum(self%n_bins_)
  end subroutine

  function bins_transform(self,x) result(z)
    class(kbins_discretizer), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    integer :: i,j,b,offset,k
    call require(allocated(self%bin_edges_),'KBinsDiscretizer is not fitted')
    call check_x(x)
    call require(size(x,2)==size(self%n_bins_),'KBinsDiscretizer feature count mismatch')
    allocate(z(size(x,1),self%n_features_out_)); z=0; offset=0
    do j=1,size(x,2)
      k=self%n_bins_(j)
      do i=1,size(x,1)
        ! searchsorted(side='right'): interior edges belong to the upper bin.
        b=count(x(i,j)>=self%bin_edges_(j)%values(2:k))
        if(self%fitted_encode=='ordinal') then
          z(i,j)=real(b,dp)
        else
          z(i,offset+b+1)=1
        end if
      end do
      offset=offset+k
    end do
  end function

  function bins_fit_transform(self,x,n_bins,sample_weight) result(z)
    class(kbins_discretizer), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in), optional :: n_bins(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x,n_bins,sample_weight); z=self%transform(x)
  end function

  function bins_inverse(self,x) result(z)
    class(kbins_discretizer), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    integer :: i,j,b,k,offset
    call require(allocated(self%bin_edges_),'KBinsDiscretizer is not fitted')
    call check_x(x)
    call require(size(x,2)==self%n_features_out_,'KBinsDiscretizer encoded width mismatch')
    allocate(z(size(x,1),size(self%n_bins_))); offset=0
    do j=1,size(self%n_bins_)
      k=self%n_bins_(j)
      do i=1,size(x,1)
        if(self%fitted_encode=='ordinal') then
          call require(x(i,j)>=0.and.x(i,j)<k,'KBinsDiscretizer inverse bin out of range')
          b=int(x(i,j))+1
          call require(x(i,j)==real(b-1,dp),'KBinsDiscretizer inverse needs integer bin codes')
        else
          call require(all(x(i,offset+1:offset+k)==0.or.x(i,offset+1:offset+k)==1), &
            'KBinsDiscretizer inverse needs one-hot values')
          call require(count(x(i,offset+1:offset+k)==1)==1,'KBinsDiscretizer inverse needs one active bin')
          b=maxloc(x(i,offset+1:offset+k),dim=1)
        end if
        if(.not.ieee_is_finite(self%bin_edges_(j)%values(1))) then
          ! sklearn's midpoint of (-inf,+inf) is NaN; construct it explicitly.
          z(i,j)=ieee_value(0._dp,ieee_quiet_nan)
        else
          z(i,j)=0.5_dp*self%bin_edges_(j)%values(b)+0.5_dp*self%bin_edges_(j)%values(b+1)
        end if
      end do
      offset=offset+k
    end do
  end function
end module
