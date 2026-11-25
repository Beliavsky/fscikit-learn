! Dense OPTICS and Xi extraction, following scikit-learn 1.6.1 cluster/_optics.py.
! See THIRD_PARTY_NOTICES.md for the retained BSD attribution.
module sklearn_optics
  use sklearn_common, only: dp,require,check_x
  use sklearn_metrics, only: pairwise_distances
  use sklearn_kernel_dispatch,only:kernel_blas_dot,kernel_blas_inner_product
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value,ieee_positive_inf
  implicit none
  private
  public :: optics,cluster_optics_dbscan,cluster_optics_xi,compute_optics_graph,optics_graph_result
  type :: optics_graph_result
    integer :: min_samples=0
    logical :: all_unreachable=.false.
    integer,allocatable :: ordering(:),predecessor(:)
    real(dp),allocatable :: core_distances(:),reachability(:)
  end type
  abstract interface
    function graph_distance(x,y)result(value)
      import dp
      real(dp),intent(in)::x(:),y(:)
      real(dp)::value
    end function
    function graph_row(point,update,eligible)result(row)
      import dp
      integer,intent(in)::point
      logical,intent(in)::update
      logical,optional,intent(in)::eligible(:)
      real(dp),allocatable::row(:)
    end function
  end interface
  interface compute_optics_graph
    module procedure compute_graph_integer,compute_graph_fraction
  end interface
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
  function compute_graph_integer(x,min_samples,max_eps,metric,p,distance)result(graph)
    real(dp),intent(in)::x(:,:)
    integer,intent(in)::min_samples
    real(dp),optional,intent(in)::max_eps,p
    character(*),optional,intent(in)::metric
    procedure(graph_distance),optional::distance
    type(optics_graph_result)::graph
    character(32)::kind
    real(dp)::radius,power
    integer::n
    call check_x(x);n=size(x,1)
    call require(min_samples>=2.and.min_samples<=n,'invalid OPTICS graph min_samples')
    radius=ieee_value(0._dp,ieee_positive_inf);if(present(max_eps))radius=max_eps
    call require(.not.ieee_is_nan(radius),'OPTICS graph max_eps must not be NaN')
    call require(radius>=0,'OPTICS graph max_eps must be nonnegative')
    kind='minkowski';if(present(metric))kind=metric
    power=2;if(present(p))power=p
    if(.not.present(distance))then
      call require(any(kind==[character(32)::'minkowski','euclidean','l2','sqeuclidean','manhattan', &
        'cityblock','l1','chebyshev','cosine','precomputed']),'unsupported OPTICS graph metric')
      if(kind=='minkowski')then
        call require(.not.ieee_is_nan(power),'OPTICS graph p must not be NaN')
        call require(power>0,'OPTICS graph p must be positive')
      end if
      if(kind=='precomputed')then
        call require(size(x,2)==n,'OPTICS graph precomputed matrix must be square')
        call require(all(x>=0),'OPTICS graph distances must be nonnegative')
      end if
    end if
    graph=build_graph(n,min_samples,radius,rows)
  contains
    function rows(point,update,eligible)result(row)
      integer,intent(in)::point
      logical,intent(in)::update
      logical,optional,intent(in)::eligible(:)
      real(dp),allocatable::row(:)
      real(dp)::nx,ny,product
      integer::j,k,count_pairs,pair_index
      allocate(row(n));row=0;count_pairs=n;pair_index=0
      if(present(eligible))count_pairs=count(eligible)
      if(present(distance))then
        do j=1,n
          row(j)=distance(x(point,:),x(j,:))
        end do
      else if(kind=='precomputed')then
        row=x(point,:)
      else
        do j=1,n
          if(present(eligible))then
            if(.not.eligible(j))cycle
          end if
          pair_index=pair_index+1
          select case(trim(kind))
          case('euclidean','l2')
            if(update)then
              nx=sum(x(point,:)**2);ny=sum(x(j,:)**2)
              product=kernel_blas_dot(x(point,:),x(j,:),1,count_pairs,1,pair_index)
            else
              nx=kernel_blas_inner_product(x(point,:),x(point,:))
              ny=kernel_blas_inner_product(x(j,:),x(j,:))
              product=kernel_blas_dot(x(point,:),x(j,:),n,n,point,j)
            end if
            row(j)=sqrt(max((nx-2*product)+ny,0._dp))
          case('sqeuclidean')
            if(update)then
              row(j)=sum((x(point,:)-x(j,:))**2)
            else
              nx=kernel_blas_inner_product(x(point,:),x(point,:))
              ny=kernel_blas_inner_product(x(j,:),x(j,:))
              product=kernel_blas_dot(x(point,:),x(j,:),n,n,point,j)
              row(j)=max((nx-2*product)+ny,0._dp)
            end if
          case('manhattan','cityblock','l1')
            row(j)=sum(abs(x(point,:)-x(j,:)))
          case('chebyshev')
            row(j)=maxval(abs(x(point,:)-x(j,:)))
          case('minkowski')
            if(.not.update.and.power==2)then
              nx=kernel_blas_inner_product(x(point,:),x(point,:))
              ny=kernel_blas_inner_product(x(j,:),x(j,:))
              product=kernel_blas_dot(x(point,:),x(j,:),n,n,point,j)
              row(j)=sqrt(max((-2*product+nx)+ny,0._dp))
            else if(ieee_is_finite(power))then
              row(j)=sum(abs(x(point,:)-x(j,:))**power)**(1/power)
            else
              row(j)=maxval(abs(x(point,:)-x(j,:)))
            end if
          case('cosine')
            nx=sqrt(sum(x(point,:)**2));ny=sqrt(sum(x(j,:)**2));product=0
            if(nx<10*epsilon(nx))nx=1
            if(ny<10*epsilon(ny))ny=1
            if(nx>0.and.ny>0)then
              if(update)then
                product=kernel_blas_dot(x(point,:)/nx,x(j,:)/ny,1,count_pairs,1,pair_index)
              else
                product=kernel_blas_dot(x(point,:)/nx,x(j,:)/ny,n,n,point,j)
              end if
            end if
            row(j)=min(2._dp,max(0._dp,1-product))
          end select
        end do
        if(.not.update)row(point)=0
      end if
      call require(all(ieee_is_finite(row)),'OPTICS graph metric returned nonfinite distances')
      call require(all(row>=0),'OPTICS graph metric returned negative distances')
    end function
  end function

  function compute_graph_fraction(x,min_samples,max_eps,metric,p,distance)result(graph)
    real(dp),intent(in)::x(:,:),min_samples
    real(dp),optional,intent(in)::max_eps,p
    character(*),optional,intent(in)::metric
    procedure(graph_distance),optional::distance
    type(optics_graph_result)::graph
    call require(ieee_is_finite(min_samples),'OPTICS graph sample fraction must be finite')
    call require(min_samples>0.and.min_samples<=1,'OPTICS graph sample fraction must be in (0,1]')
    graph=compute_graph_integer(x,max(2,int(min_samples*size(x,1))),max_eps,metric,p,distance)
  end function

  real(dp)function kth_distance(row,k)result(value)
    real(dp),intent(in)::row(:)
    integer,intent(in)::k
    real(dp)::heap(k)
    integer::i
    heap=row(:k)
    do i=k/2,1,-1
      call sift_max(heap,i)
    end do
    do i=k+1,size(row)
      if(row(i)<heap(1))then
        heap(1)=row(i);call sift_max(heap,1)
      end if
    end do
    value=heap(1)
  end function
  subroutine sift_max(heap,start)
    real(dp),intent(inout)::heap(:)
    integer,intent(in)::start
    integer::parent,child
    real(dp)::value
    parent=start;value=heap(parent)
    do while(parent<=size(heap)/2)
      child=2*parent
      if(child<size(heap))then
        if(heap(child+1)>heap(child))child=child+1
      end if
      if(value>=heap(child))exit
      heap(parent)=heap(child);parent=child
    end do
    heap(parent)=value
  end subroutine
  function build_graph(n,min_samples,max_eps,rows)result(graph)
    integer,intent(in)::n,min_samples
    real(dp),intent(in)::max_eps
    procedure(graph_row)::rows
    type(optics_graph_result)::graph
    real(dp),allocatable::row(:),radius_row(:)
    logical::processed(n),eligible(n)
    integer::i,j,t,point
    real(dp)::rd,inf
    inf=ieee_value(0._dp,ieee_positive_inf)
    graph%min_samples=min_samples
    allocate(graph%ordering(n),graph%predecessor(n),graph%reachability(n),graph%core_distances(n))
    graph%reachability=inf;graph%predecessor=-1;processed=.false.
    do i=1,n
      row=rows(i,.false.);rd=kth_distance(row,min_samples)
      if(rd>max_eps)rd=inf
      graph%core_distances(i)=rounded(rd)
    end do
    do t=1,n
      point=0
      do i=1,n
        if(processed(i))cycle
        if(point==0)then
          point=i
        else if(graph%reachability(i)<graph%reachability(point))then
          point=i
        end if
      end do
      graph%ordering(t)=point;processed(point)=.true.
      if(.not.ieee_is_finite(graph%core_distances(point)))cycle
      eligible=.not.processed
      if(ieee_is_finite(max_eps))then
        radius_row=rows(point,.false.);eligible=eligible.and.radius_row<=max_eps
      end if
      if(.not.any(eligible))cycle
      row=rows(point,.true.,eligible)
      do j=1,n
        if(.not.eligible(j))cycle
        rd=rounded(max(graph%core_distances(point),row(j)))
        if(rd<graph%reachability(j))then
          graph%reachability(j)=rd;graph%predecessor(j)=point
        end if
      end do
    end do
    graph%all_unreachable=all(.not.ieee_is_finite(graph%reachability))
  end function

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
    if(x>huge(x)/1.e15_dp)then
      y=ieee_value(0._dp,ieee_positive_inf);return
    end if
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
    type(optics_graph_result)::graph
    integer :: n,i
    real(dp) :: inf,radius
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
      do i=1,n
        call require(all(abs(x(i,:)-x(:,i))<=1.e-10_dp*max(1._dp,abs(x(i,:)))), &
          'OPTICS distance matrix must be symmetric')
        call require(abs(x(i,i))<=1.e-12_dp,'OPTICS distance diagonal must be zero')
      end do
    end if
    graph=build_graph(n,self%min_samples_,self%max_eps_,legacy_rows)
    self%ordering_=graph%ordering;self%predecessor_=graph%predecessor
    self%reachability_=graph%reachability;self%core_distances_=graph%core_distances
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
  contains
    function legacy_rows(point,update,eligible)result(row)
      integer,intent(in)::point
      logical,intent(in)::update
      logical,optional,intent(in)::eligible(:)
      real(dp),allocatable::row(:),distances(:,:)
      if(trim(self%metric)=='precomputed')then
        row=x(point,:)
      else
        distances=pairwise_distances(x(point:point,:),x,self%metric,self%p);row=distances(1,:)
      end if
      row(point)=0
    end function
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
