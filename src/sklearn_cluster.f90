module sklearn_cluster
  use sklearn_common
  use sklearn_metrics, only: pairwise_distances
  implicit none
  private
  public :: kmeans,dbscan,agglomerative_clustering
  type :: kmeans
    integer :: n_clusters=8,n_init=10,max_iter=300,random_state=0,n_iter_=0
    real(dp) :: tol=1.e-4_dp,inertia_=0
    logical :: converged_=.false.
    real(dp), allocatable :: cluster_centers_(:,:)
    integer, allocatable :: labels_(:)
  contains
    procedure :: fit=>kmeans_fit
    procedure :: predict=>kmeans_predict
    procedure :: transform=>kmeans_transform
    procedure :: fit_predict=>kmeans_fit_predict
    procedure :: score=>kmeans_score
  end type
  type :: dbscan
    real(dp) :: eps=0.5_dp
    integer :: min_samples=5
    character(16) :: metric='euclidean'
    integer, allocatable :: labels_(:),core_sample_indices_(:)
    real(dp), allocatable :: components_(:,:)
  contains
    procedure :: fit=>dbscan_fit
    procedure :: fit_predict=>dbscan_fit_predict
  end type
  type :: agglomerative_clustering
    integer :: n_clusters=2
    character(16) :: linkage='ward'
    integer, allocatable :: labels_(:),children_(:,:)
    real(dp), allocatable :: distances_(:)
  contains
    procedure :: fit=>agg_fit
    procedure :: fit_predict=>agg_fit_predict
  end type
contains
  subroutine kmeans_fit(self,x,init_centers)
    class(kmeans), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(in), optional :: init_centers(:,:)
    real(dp), allocatable :: centers(:,:),next(:,:),d(:,:),closest(:),weights(:)
    integer, allocatable :: labels(:),previous(:),counts(:)
    real(dp) :: best,inertia,r,cumulative,shift,variance
    integer :: run,runs,it,n,p,k,i,j,chosen
    call check_x(x); n=size(x,1); p=size(x,2); k=self%n_clusters
    call require(k>0.and.k<=n,'invalid number of kmeans clusters')
    call require(self%n_init>0.and.self%max_iter>0.and.self%tol>=0,'invalid kmeans controls')
    runs=self%n_init
    if(present(init_centers)) then
      call require(all(shape(init_centers)==[k,p]),'kmeans initial centers shape mismatch')
      call check_x(init_centers); runs=1
    end if
    allocate(centers(k,p),next(k,p),labels(n),previous(n),counts(k),closest(n),weights(n))
    best=huge(1._dp); call seed_rng(self%random_state)
    variance=0
    do j=1,p
      variance=variance+sum((x(:,j)-sum(x(:,j))/n)**2)/n
    end do
    variance=variance/p
    do run=1,runs
      if(present(init_centers)) then
        centers=init_centers
      else
        call random_number(r); chosen=min(n,1+int(r*n)); centers(1,:)=x(chosen,:)
        closest=sum((x-spread(centers(1,:),1,n))**2,dim=2)
        do j=2,k
          call random_number(r); cumulative=0; chosen=n
          if(sum(closest)==0) then
            chosen=1+mod(j-1,n)
          else
            r=r*sum(closest)
            do i=1,n
              cumulative=cumulative+closest(i)
              if(cumulative>r) then
                chosen=i; exit
              end if
            end do
          end if
          centers(j,:)=x(chosen,:)
          closest=min(closest,sum((x-spread(centers(j,:),1,n))**2,dim=2))
        end do
      end if
      previous=-1
      do it=1,self%max_iter
        d=pairwise_distances(x,centers,'sqeuclidean'); labels=minloc(d,dim=2)
        next=0; counts=0
        do i=1,n
          next(labels(i),:)=next(labels(i),:)+x(i,:); counts(labels(i))=counts(labels(i))+1
        end do
        closest=minval(d,dim=2)
        do j=1,k
          if(counts(j)>0) then
            next(j,:)=next(j,:)/counts(j)
          else
            chosen=maxloc(closest,dim=1); next(j,:)=x(chosen,:); closest(chosen)=-1
          end if
        end do
        shift=sum((next-centers)**2); centers=next
        if(all(labels==previous).or.shift<=self%tol*variance) exit
        previous=labels
      end do
      d=pairwise_distances(x,centers,'sqeuclidean'); labels=minloc(d,dim=2)
      inertia=sum(minval(d,dim=2))
      if(inertia<best) then
        best=inertia; self%cluster_centers_=centers; self%labels_=labels-1
        self%inertia_=inertia; self%n_iter_=min(it,self%max_iter); self%converged_=it<=self%max_iter
      end if
    end do
  end subroutine
  function kmeans_transform(self,x) result(d)
    class(kmeans), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: d(:,:)
    call require(allocated(self%cluster_centers_),'kmeans is not fitted')
    d=pairwise_distances(x,self%cluster_centers_)
  end function
  function kmeans_predict(self,x) result(labels)
    class(kmeans), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: labels(size(x,1))
    real(dp), allocatable :: d(:,:)
    d=self%transform(x); labels=minloc(d,dim=2)-1
  end function
  function kmeans_fit_predict(self,x) result(labels)
    class(kmeans), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: labels(size(x,1))
    call self%fit(x); labels=self%labels_
  end function
  real(dp) function kmeans_score(self,x) result(v)
    class(kmeans), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: d(:,:)
    d=self%transform(x); v=-sum(minval(d,dim=2)**2)
  end function
  subroutine dbscan_fit(self,x)
    class(dbscan), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    logical, allocatable :: adjacent(:,:),core(:),queued(:)
    integer :: n,i,j,c,head,tail,item
    integer, allocatable :: queue(:)
    call check_x(x); n=size(x,1)
    call require(self%eps>0.and.self%min_samples>0,'invalid DBSCAN controls')
    adjacent=pairwise_distances(x,x,self%metric)<=self%eps
    core=count(adjacent,dim=2)>=self%min_samples
    self%labels_=spread(-1,1,n); allocate(queue(n),queued(n)); c=0
    do i=1,n
      if(self%labels_(i)/=-1.or..not.core(i)) cycle
      head=1; tail=1; queue(1)=i; queued=.false.; queued(i)=.true.
      do while(head<=tail)
        item=queue(head); head=head+1
        if(self%labels_(item)/=-1) cycle
        self%labels_(item)=c
        if(.not.core(item)) cycle
        do j=1,n
          if(adjacent(item,j).and..not.queued(j).and.self%labels_(j)==-1) then
            tail=tail+1; queue(tail)=j; queued(j)=.true.
          end if
        end do
      end do
      c=c+1
    end do
    self%core_sample_indices_=pack([(i,i=1,n)],core)
    self%components_=x(self%core_sample_indices_,:)
  end subroutine
  function dbscan_fit_predict(self,x) result(labels)
    class(dbscan), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: labels(size(x,1))
    call self%fit(x); labels=self%labels_
  end function
  subroutine agg_fit(self,x)
    class(agglomerative_clustering), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: d(:,:),centers(:,:),sizes(:)
    integer, allocatable :: membership(:),node(:)
    logical, allocatable :: active(:)
    real(dp) :: best,value
    integer :: n,p,step,i,j,a,b,c,total
    call check_x(x); n=size(x,1); p=size(x,2)
    call require(self%n_clusters>0.and.self%n_clusters<=n,'invalid agglomerative cluster count')
    call require(any(self%linkage==['ward    ','single  ','complete','average ']),'unknown linkage')
    d=pairwise_distances(x,x); allocate(centers(n,p),sizes(n),membership(n),node(n),active(n))
    centers=x; sizes=1; membership=[(i,i=1,n)]; node=[(i-1,i=1,n)]; active=.true.
    if(allocated(self%children_)) deallocate(self%children_,self%distances_)
    allocate(self%children_(n-1,2),self%distances_(n-1))
    self%labels_=membership-1
    do step=1,n-1
      best=huge(1._dp); a=0; b=0
      do j=2,n
        if(.not.active(j)) cycle
        do i=1,j-1
          if(.not.active(i)) cycle
          value=d(i,j)
          if(trim(self%linkage)=='ward') value=sqrt(2*sizes(i)*sizes(j)/(sizes(i)+sizes(j))* &
            sum((centers(i,:)-centers(j,:))**2))
          if(value<best) then
            best=value; a=i; b=j
          end if
        end do
      end do
      self%children_(step,:)=[node(a),node(b)]; self%distances_(step)=best
      centers(a,:)=(sizes(a)*centers(a,:)+sizes(b)*centers(b,:))/(sizes(a)+sizes(b))
      do c=1,n
        if(c==a.or.c==b.or..not.active(c)) cycle
        select case(trim(self%linkage))
        case('single'); value=min(d(a,c),d(b,c))
        case('complete'); value=max(d(a,c),d(b,c))
        case('average'); value=(sizes(a)*d(a,c)+sizes(b)*d(b,c))/(sizes(a)+sizes(b))
        case('ward'); value=0
        end select
        d(a,c)=value; d(c,a)=value
      end do
      sizes(a)=sizes(a)+sizes(b); active(b)=.false.; node(a)=n+step-1
      where(membership==b) membership=a
      if(n-step==self%n_clusters) then
        total=0
        do c=1,n
          if(.not.active(c)) cycle
          where(membership==c) self%labels_=total
          total=total+1
        end do
      end if
    end do
  end subroutine
  function agg_fit_predict(self,x) result(labels)
    class(agglomerative_clustering), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: labels(size(x,1))
    call self%fit(x); labels=self%labels_
  end function
end module
