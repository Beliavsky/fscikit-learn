! Dense OPTICS and Xi extraction, following scikit-learn 1.6.1 cluster/_optics.py.
! See THIRD_PARTY_NOTICES.md for the retained BSD attribution.
module sklearn_optics
  use sklearn_common, only: dp,require,check_x,argsort
  use sklearn_metrics, only: pairwise_distances
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value,ieee_positive_inf
  implicit none
  private
  public :: optics,cluster_optics_dbscan,cluster_optics_xi
  type :: optics
    integer :: min_samples=5,min_cluster_size=0,min_samples_=0,min_cluster_size_=0
    real(dp) :: min_samples_fraction=-1,min_cluster_fraction=-1
    real(dp) :: max_eps=huge(1._dp),max_eps_=0,eps=-1,xi=.05_dp,p=2
    character(16) :: metric='minkowski',cluster_method='xi'
    logical :: predecessor_correction=.true.
    integer, allocatable :: ordering_(:),predecessor_(:),labels_(:),cluster_hierarchy_(:,:)
    real(dp), allocatable :: reachability_(:),core_distances_(:)
  contains
    procedure :: fit=>optics_fit
    procedure :: fit_predict=>optics_fit_predict
  end type
contains
  integer function resolved_size(count,fraction,n,automatic) result(k)
    integer, intent(in) :: count,n,automatic
    real(dp), intent(in) :: fraction
    call require(ieee_is_finite(fraction),'nonfinite OPTICS size fraction')
    call require(fraction==-1 .or. (fraction>=0 .and. fraction<=1),'invalid OPTICS size fraction')
    k=count
    if(k==0) k=automatic
    if(fraction>=0) k=max(2,int(fraction*n))
    call require(k>=2 .and. k<=n,'invalid OPTICS sample or cluster size')
  end function

  real(dp) function rounded(x) result(y)
    real(dp), intent(in) :: x
    real(dp) :: z,a,b
    y=x
    if(.not.ieee_is_finite(x)) return
    if(x>huge(x)/1.e15_dp) return
    z=x*1.e15_dp; a=aint(z); b=z-a
    if(b>.5_dp) a=a+1
    if(b==.5_dp) then
      if(modulo(a,2._dp)/=0) a=a+1
    end if
    y=a/1.e15_dp
  end function

  subroutine optics_fit(self,x)
    class(optics), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: d(:,:),row(:)
    integer, allocatable :: order(:)
    logical, allocatable :: processed(:)
    integer :: n,i,j,t,point
    real(dp) :: inf,rd,radius
    call check_x(x)
    n=size(x,1); inf=ieee_value(0._dp,ieee_positive_inf)
    self%min_samples_=resolved_size(self%min_samples,self%min_samples_fraction,n,self%min_samples)
    self%min_cluster_size_=resolved_size(self%min_cluster_size,self%min_cluster_fraction,n,self%min_samples_)
    call require(.not.ieee_is_nan(self%max_eps),'invalid OPTICS max_eps')
    call require(self%max_eps>=0,'invalid OPTICS max_eps')
    call require(.not.ieee_is_nan(self%eps),'invalid OPTICS eps')
    call require(self%eps==-1 .or. self%eps>=0,'invalid OPTICS eps')
    call require(ieee_is_finite(self%p),'invalid OPTICS Minkowski p')
    call require(self%p>=1,'invalid OPTICS Minkowski p')
    call require(trim(self%cluster_method)=='xi' .or. trim(self%cluster_method)=='dbscan', &
      'invalid OPTICS cluster_method')
    call require(ieee_is_finite(self%xi),'invalid OPTICS xi')
    call require(self%xi>0 .and. self%xi<1,'invalid OPTICS xi')
    self%max_eps_=self%max_eps
    if(self%max_eps==huge(1._dp)) self%max_eps_=inf
    if(trim(self%metric)=='precomputed') then
      call require(size(x,2)==n,'OPTICS distance matrix must be square')
      call require(all(x>=0),'OPTICS distances must be nonnegative')
      call require(all(abs(x-transpose(x))<=1.e-10_dp*max(1._dp,abs(x))), &
        'OPTICS distance matrix must be symmetric')
      d=x
      do i=1,n
        call require(abs(d(i,i))<=1.e-12_dp,'OPTICS distance diagonal must be zero')
      end do
    else
      d=pairwise_distances(x,x,self%metric,self%p)
    end if
    do i=1,n
      d(i,i)=0
    end do
    if(allocated(self%ordering_)) deallocate(self%ordering_,self%predecessor_,self%reachability_,self%core_distances_)
    allocate(self%ordering_(n),self%predecessor_(n),self%reachability_(n),self%core_distances_(n),processed(n))
    self%reachability_=inf; self%predecessor_=-1; processed=.false.
    do i=1,n
      row=d(i,:); order=argsort(row)
      rd=row(order(self%min_samples_))
      if(rd>self%max_eps_) rd=inf
      self%core_distances_(i)=rounded(rd)
    end do
    do t=1,n
      point=0
      do i=1,n
        if(processed(i)) cycle
        if(point==0) then
          point=i
        else if(self%reachability_(i)<self%reachability_(point)) then
          point=i
        end if
      end do
      self%ordering_(t)=point; processed(point)=.true.
      if(.not.ieee_is_finite(self%core_distances_(point))) cycle
      do j=1,n
        if(processed(j) .or. d(point,j)>self%max_eps_) cycle
        rd=rounded(max(self%core_distances_(point),d(point,j)))
        if(rd<self%reachability_(j)) then
          self%reachability_(j)=rd; self%predecessor_(j)=point
        end if
      end do
    end do
    if(trim(self%cluster_method)=='xi') then
      call cluster_optics_xi(self%reachability_,self%predecessor_,self%ordering_,self%min_samples_, &
        self%labels_,self%cluster_hierarchy_,self%xi,self%min_cluster_size_,self%predecessor_correction)
    else
      radius=self%eps
      if(radius==-1) radius=self%max_eps_
      call require(radius<=self%max_eps_,'OPTICS eps exceeds max_eps')
      self%labels_=cluster_optics_dbscan(self%reachability_,self%core_distances_,self%ordering_,radius)
      if(allocated(self%cluster_hierarchy_)) deallocate(self%cluster_hierarchy_)
      allocate(self%cluster_hierarchy_(0,2))
    end if
  end subroutine

  function optics_fit_predict(self,x) result(labels)
    class(optics), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: labels(:)
    call self%fit(x)
    labels=self%labels_
  end function

  subroutine validate_graph(reachability,ordering)
    real(dp), intent(in) :: reachability(:)
    integer, intent(in) :: ordering(:)
    integer :: i,n
    n=size(reachability)
    call require(n>=2 .and. size(ordering)==n,'invalid OPTICS graph shape')
    call require(.not.any(ieee_is_nan(reachability)),'invalid OPTICS reachability')
    call require(all(reachability>=0),'invalid OPTICS reachability')
    call require(all(ordering>=1) .and. all(ordering<=n),'invalid OPTICS ordering')
    do i=1,n
      call require(count(ordering==i)==1,'invalid OPTICS ordering')
    end do
  end subroutine

  function cluster_optics_dbscan(reachability,core_distances,ordering,eps) result(labels)
    real(dp), intent(in) :: reachability(:),core_distances(:),eps
    integer, intent(in) :: ordering(:)
    integer, allocatable :: labels(:)
    integer :: t,i,k
    logical :: far,near
    call validate_graph(reachability,ordering)
    call require(size(core_distances)==size(ordering),'invalid OPTICS core distance shape')
    call require(.not.any(ieee_is_nan(core_distances)),'invalid OPTICS core distances')
    call require(all(core_distances>=0),'invalid OPTICS core distances')
    call require(.not.ieee_is_nan(eps),'invalid OPTICS extraction eps')
    call require(eps>=0,'invalid OPTICS extraction eps')
    allocate(labels(size(ordering))); k=-1
    do t=1,size(ordering)
      i=ordering(t); far=reachability(i)>eps; near=core_distances(i)<=eps
      if(far .and. near) k=k+1
      labels(i)=k
      if(far .and. .not.near) labels(i)=-1
    end do
  end function

  integer function extend_region(steep,xward,start,min_samples) result(last)
    logical, intent(in) :: steep(:),xward(:)
    integer, intent(in) :: start,min_samples
    integer :: i,nonsteep
    last=start; nonsteep=0
    do i=start,size(steep)
      if(steep(i)) then
        nonsteep=0; last=i
      else if(.not.xward(i)) then
        nonsteep=nonsteep+1
        if(nonsteep>min_samples) return
      else
        return
      end if
    end do
  end function

  subroutine cluster_optics_xi(reachability,predecessor,ordering,min_samples,labels,clusters, &
      xi,min_cluster_size,predecessor_correction)
    real(dp), intent(in) :: reachability(:)
    integer, intent(in) :: predecessor(:),ordering(:),min_samples
    integer, allocatable, intent(out) :: labels(:),clusters(:,:)
    real(dp), intent(in), optional :: xi
    integer, intent(in), optional :: min_cluster_size
    logical, intent(in), optional :: predecessor_correction
    real(dp), allocatable :: r(:),dm(:)
    integer, allocatable :: ds(:),de(:),buffer(:,:),grown(:,:),ordered_labels(:),pred(:)
    logical, allocatable :: su(:),sd(:),up(:),down(:)
    integer :: n,csize,i,j,index,ns,ne,first,last,cs,ce,nc,begin_block,k,a,b,tmp(2),label
    real(dp) :: xc,mib,dmax,xival
    logical :: correction,valid
    call validate_graph(reachability,ordering)
    n=size(ordering); csize=min_samples; xival=.05_dp; correction=.true.
    if(present(xi)) xival=xi
    if(present(min_cluster_size)) csize=min_cluster_size
    if(present(predecessor_correction)) correction=predecessor_correction
    call require(ieee_is_finite(xival),'invalid OPTICS xi')
    call require(xival>0 .and. xival<1,'invalid OPTICS xi')
    call require(min_samples>=2 .and. min_samples<=n,'invalid OPTICS sample or cluster size')
    call require(csize>=2 .and. csize<=n,'invalid OPTICS sample or cluster size')
    call require(size(predecessor)==n,'invalid OPTICS predecessor shape')
    call require(all(predecessor==-1 .or. (predecessor>=1 .and. predecessor<=n)), &
      'invalid OPTICS predecessor')
    allocate(r(n+1),dm(n),ds(n),de(n),buffer(n,2),su(n),sd(n),up(n),down(n))
    r(:n)=reachability(ordering); r(n+1)=ieee_value(0._dp,ieee_positive_inf)
    pred=predecessor(ordering); xc=1-xival
    su=.false.; sd=.false.; up=.false.; down=.false.
    do i=1,n
      if(.not.ieee_is_finite(r(i)) .and. .not.ieee_is_finite(r(i+1))) cycle
      if(r(i)==0 .and. r(i+1)==0) cycle
      up(i)=r(i)<r(i+1); down(i)=r(i)>r(i+1)
      su(i)=r(i)<=xc*r(i+1); sd(i)=xc*r(i)>=r(i+1)
    end do
    index=1; mib=0; ns=0; nc=0
    do i=1,n
      if(.not.(su(i) .or. sd(i)) .or. i<index) cycle
      mib=max(mib,maxval(r(index:i)))
      ne=0
      if(ieee_is_finite(mib)) then
        do j=1,ns
          if(mib>r(ds(j))*xc) cycle
          ne=ne+1; ds(ne)=ds(j); de(ne)=de(j); dm(ne)=max(dm(j),mib)
        end do
      end if
      ns=ne
      if(sd(i)) then
        last=extend_region(sd,up,i,min_samples)
        ns=ns+1; ds(ns)=i; de(ns)=last; dm(ns)=0
        index=last+1; mib=r(index)
      else
        first=i; last=extend_region(su,down,i,min_samples)
        index=last+1; mib=r(index); begin_block=nc+1
        do j=1,ns
          cs=ds(j); ce=last
          if(r(ce+1)*xc<dm(j)) cycle
          dmax=r(ds(j))
          if(dmax*xc>=r(ce+1)) then
            do while(cs<de(j))
              if(r(cs+1)<=r(ce+1)) exit
              cs=cs+1
            end do
          else if(r(ce+1)*xc>=dmax) then
            do while(ce>first)
              if(r(ce-1)<=dmax) exit
              ce=ce-1
            end do
          end if
          valid=.true.
          if(correction) then
            valid=.false.
            do while(cs<ce)
              if(r(cs)>r(ce)) then
                valid=.true.; exit
              end if
              if(any(pred(ce)==ordering(cs:ce-1))) then
                valid=.true.; exit
              end if
              ce=ce-1
            end do
          end if
          if(.not.valid) cycle
          if(ce-cs+1<csize .or. cs>de(j) .or. ce<first) cycle
          if(nc==size(buffer,1)) then
            allocate(grown(2*size(buffer,1),2)); grown(:nc,:)=buffer
            call move_alloc(grown,buffer)
          end if
          nc=nc+1; buffer(nc,:)=[cs,ce]
        end do
        a=begin_block; b=nc
        do while(a<b)
          tmp=buffer(a,:); buffer(a,:)=buffer(b,:); buffer(b,:)=tmp; a=a+1; b=b-1
        end do
      end if
    end do
    clusters=buffer(:nc,:)
    allocate(ordered_labels(n),labels(n)); ordered_labels=-1; label=0
    do k=1,nc
      cs=clusters(k,1); ce=clusters(k,2)
      if(any(ordered_labels(cs:ce)/=-1)) cycle
      ordered_labels(cs:ce)=label; label=label+1
    end do
    labels(ordering)=ordered_labels
  end subroutine
end module
