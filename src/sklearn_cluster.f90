module sklearn_cluster
  use sklearn_common
  use sklearn_metrics, only: pairwise_distances
  use sklearn_spectral, only: spectral_affinity_matrix,spectral_eigenvectors
  use sklearn_optics, only: optics,cluster_optics_dbscan,cluster_optics_xi
  implicit none
  private
  public :: optics,cluster_optics_dbscan,cluster_optics_xi
  public :: kmeans,dbscan,agglomerative_clustering
  public :: mean_shift,estimate_bandwidth,affinity_propagation,spectral_clustering
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
  type :: mean_shift
    real(dp) :: bandwidth=-1,bandwidth_=0
    integer :: max_iter=300,n_iter_=0
    logical :: cluster_all=.true.,converged_=.false.
    real(dp), allocatable :: cluster_centers_(:,:)
    integer, allocatable :: labels_(:)
  contains
    procedure :: fit=>mean_shift_fit
    procedure :: predict=>mean_shift_predict
    procedure :: fit_predict=>mean_shift_fit_predict
  end type
  type :: affinity_propagation
    real(dp) :: damping=0.5_dp,preference=huge(1._dp)
    integer :: max_iter=200,convergence_iter=15,random_state=0,n_iter_=0
    character(16) :: affinity='euclidean'
    logical :: converged_=.false.
    real(dp), allocatable :: affinity_matrix_(:,:),cluster_centers_(:,:)
    integer, allocatable :: cluster_centers_indices_(:),labels_(:)
  contains
    procedure :: fit=>affinity_fit
    procedure :: fit_predict=>affinity_fit_predict
    procedure :: predict=>affinity_predict
  end type
  type :: spectral_clustering
    integer :: n_clusters=8,n_neighbors=10,n_init=10,random_state=0,n_connected_components_=0
    real(dp) :: gamma=1
    character(20) :: affinity='rbf',assign_labels='kmeans'
    integer, allocatable :: labels_(:)
    real(dp), allocatable :: affinity_matrix_(:,:),embedding_(:,:)
  contains
    procedure :: fit=>spectral_fit
    procedure :: fit_predict=>spectral_fit_predict
  end type
contains
  ! sklearn 1.6.1 cluster/_mean_shift.py, _affinity_propagation.py and
  ! _spectral.py. See THIRD_PARTY_NOTICES.md for retained upstream attribution.
  real(dp) function estimate_bandwidth(x,quantile) result(bandwidth)
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(in), optional :: quantile
    real(dp) :: q
    real(dp), allocatable :: d(:,:)
    integer :: n,k,i,order(size(x,1))
    call check_x(x); n=size(x,1); q=0.3_dp; if(present(quantile)) q=quantile
    call require(q>=0.and.q<=1,'bandwidth quantile must be in [0,1]')
    k=max(1,int(n*q)); d=pairwise_distances(x,x); bandwidth=0
    do i=1,n
      order=argsort(d(i,:)); bandwidth=bandwidth+d(i,order(k))
    end do
    bandwidth=bandwidth/n
  end function
  logical function mode_before(a,count_a,b,count_b) result(before)
    real(dp), intent(in) :: a(:),b(:)
    integer, intent(in) :: count_a,count_b
    integer :: j
    before=count_a>count_b
    if(count_a/=count_b) return
    do j=1,size(a)
      if(a(j)==b(j)) cycle
      before=a(j)>b(j); return
    end do
  end function
  subroutine mean_shift_fit(self,x,seeds)
    class(mean_shift), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(in), optional :: seeds(:,:)
    real(dp), allocatable :: starts(:,:),centers(:,:),d(:,:)
    real(dp) :: center(size(x,2)),next(size(x,2))
    integer, allocatable :: counts(:),order(:)
    logical, allocatable :: keep(:)
    logical :: nearby(size(x,1)),done
    integer :: n,p,s,i,j,k,completed,valid_count,idx
    call check_x(x); n=size(x,1); p=size(x,2)
    call require(self%max_iter>=0,'MeanShift max_iter must be nonnegative')
    call require(self%bandwidth>0.or.self%bandwidth==-1,'MeanShift bandwidth must be positive or -1')
    self%bandwidth_=self%bandwidth
    if(self%bandwidth==-1) self%bandwidth_=estimate_bandwidth(x)
    call check_x(reshape([self%bandwidth_],[1,1]))
    starts=x
    if(present(seeds)) then
      call check_xy(transpose(seeds),p); starts=seeds
    end if
    s=size(starts,1); allocate(centers(s,p),counts(s),order(s)); valid_count=0
    self%n_iter_=0; self%converged_=.true.
    do i=1,s
      center=starts(i,:); completed=0; done=.false.
      do
        nearby=sum((x-spread(center,1,n))**2,dim=2)<=self%bandwidth_**2
        k=count(nearby); if(k==0) exit
        next=0
        do j=1,n
          if(nearby(j)) next=next+x(j,:)
        end do
        next=next/k; done=sqrt(sum((next-center)**2))<=1.e-3_dp*self%bandwidth_; center=next
        if(done.or.completed==self%max_iter) exit
        completed=completed+1
      end do
      self%n_iter_=max(self%n_iter_,completed)
      if(k==0) cycle
      self%converged_=self%converged_.and.done
      idx=0
      do j=1,valid_count
        if(all(centers(j,:)==center)) then
          idx=j; exit
        end if
      end do
      if(idx==0) then
        valid_count=valid_count+1; idx=valid_count
      end if
      centers(idx,:)=center; counts(idx)=k
    end do
    call require(valid_count>0,'no point within MeanShift bandwidth of any seed')
    order(:valid_count)=[(i,i=1,valid_count)]
    do i=2,valid_count
      idx=order(i); j=i-1
      do while(j>=1)
        if(.not.mode_before(centers(idx,:),counts(idx),centers(order(j),:),counts(order(j)))) exit
        order(j+1)=order(j); j=j-1
      end do
      order(j+1)=idx
    end do
    centers(:valid_count,:)=centers(order(:valid_count),:)
    allocate(keep(valid_count)); keep=.true.
    do i=1,valid_count
      if(.not.keep(i)) cycle
      do j=i+1,valid_count
        if(sum((centers(i,:)-centers(j,:))**2)<=self%bandwidth_**2) keep(j)=.false.
      end do
    end do
    order=pack([(i,i=1,valid_count)],keep); self%cluster_centers_=centers(order,:)
    self%labels_=self%predict(x)
    if(.not.self%cluster_all) then
      d=pairwise_distances(x,self%cluster_centers_)
      where(minval(d,dim=2)>self%bandwidth_) self%labels_=-1
    end if
  end subroutine
  function mean_shift_predict(self,x) result(labels)
    class(mean_shift), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: labels(size(x,1))
    real(dp), allocatable :: d(:,:)
    call require(allocated(self%cluster_centers_),'MeanShift is not fitted')
    d=pairwise_distances(x,self%cluster_centers_); labels=minloc(d,dim=2)-1
  end function
  function mean_shift_fit_predict(self,x) result(labels)
    class(mean_shift), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: labels(size(x,1))
    call self%fit(x); labels=self%labels_
  end function
  subroutine affinity_fit(self,x)
    class(affinity_propagation), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: s(:,:),a(:,:),r(:,:),tmp(:,:),col(:),flat(:)
    integer, allocatable :: exemplars(:),assignment(:),members(:),sorted(:)
    logical, allocatable :: history(:,:),chosen(:)
    real(dp) :: pref,first,second
    integer :: n,i,j,k,it,idx
    logical :: equal
    call check_x(x); n=size(x,1)
    call check_x(reshape([self%preference],[1,1]))
    call require(self%damping>=0.5_dp.and.self%damping<1.and.self%max_iter>0.and.self%convergence_iter>0, &
      'invalid AffinityPropagation controls')
    select case(self%affinity)
    case('euclidean'); self%affinity_matrix_=-pairwise_distances(x,x,'sqeuclidean')
    case('precomputed')
      call require(size(x,2)==n,'affinity similarities must be square'); self%affinity_matrix_=x
    case default; call require(.false.,'unknown AffinityPropagation affinity')
    end select
    s=self%affinity_matrix_; pref=self%preference
    if(pref==huge(1._dp)) then
      flat=reshape(s,[n*n]); sorted=argsort(flat)
      pref=0.5_dp*(flat(sorted((n*n+1)/2))+flat(sorted(n*n/2+1)))
    end if
    equal=.true.
    do j=1,n
      do i=1,n
        if(i==j) cycle
        if(s(i,j)/=s(1,n)) equal=.false.
      end do
    end do
    self%n_iter_=0; self%converged_=.false.
    if(n==1.or.equal) then
      if(pref>s(1,n)) then
        self%cluster_centers_indices_=[(i,i=1,n)]; self%labels_=[(i-1,i=1,n)]
      else
        self%cluster_centers_indices_=[1]; self%labels_=spread(0,1,n)
      end if
      self%converged_=.true.
    else
      allocate(a(n,n),r(n,n),tmp(n,n),col(n),history(n,self%convergence_iter),chosen(n),assignment(n))
      do i=1,n
        s(i,i)=pref
      end do
      call seed_rng(self%random_state)
      s=s+(epsilon(1._dp)*s+100*tiny(1._dp))*normal_random(n,n)
      a=0; r=0; history=.false.
      do it=1,self%max_iter
        tmp=a+s
        do i=1,n
          idx=maxloc(tmp(i,:),dim=1); first=tmp(i,idx); tmp(i,idx)=-huge(1._dp); second=maxval(tmp(i,:))
          tmp(i,:)=s(i,:)-first; tmp(i,idx)=s(i,idx)-second
        end do
        r=self%damping*r+(1-self%damping)*tmp; tmp=max(r,0._dp)
        do i=1,n
          tmp(i,i)=r(i,i)
        end do
        col=sum(tmp,dim=1)
        do j=1,n
          do i=1,n
            if(i==j) then
              tmp(i,j)=col(j)-r(j,j)
            else
              tmp(i,j)=min(0._dp,col(j)-tmp(i,j))
            end if
          end do
        end do
        a=self%damping*a+(1-self%damping)*tmp
        chosen=[(a(i,i)+r(i,i)>0,i=1,n)]; history(:,mod(it-1,self%convergence_iter)+1)=chosen
        self%n_iter_=it
        if(it>self%convergence_iter.and.any(chosen)) then
          if(all(all(history,dim=2).or..not.any(history,dim=2))) then
            self%converged_=.true.; exit
          end if
        end if
      end do
      exemplars=pack([(i,i=1,n)],chosen); k=size(exemplars)
      if(k>0) then
        assignment=maxloc(s(:,exemplars),dim=2)
        do i=1,k
          assignment(exemplars(i))=i
        end do
        do i=1,k
          members=pack([(j,j=1,n)],assignment==i)
          idx=maxloc(sum(s(members,members),dim=1),dim=1); exemplars(i)=members(idx)
        end do
        assignment=maxloc(s(:,exemplars),dim=2)
        do i=1,k
          assignment(exemplars(i))=i
        end do
        self%cluster_centers_indices_=unique_labels(exemplars(assignment))
        self%labels_=[(label_index(self%cluster_centers_indices_,exemplars(assignment(i)))-1,i=1,n)]
      else
        self%cluster_centers_indices_=pack([(i,i=1,n)],chosen); self%labels_=spread(-1,1,n)
      end if
    end if
    self%affinity_matrix_=s
    if(allocated(self%cluster_centers_)) deallocate(self%cluster_centers_)
    if(self%affinity=='euclidean') self%cluster_centers_=x(self%cluster_centers_indices_,:)
  end subroutine
  function affinity_fit_predict(self,x) result(labels)
    class(affinity_propagation), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: labels(size(x,1))
    call self%fit(x); labels=self%labels_
  end function
  function affinity_predict(self,x) result(labels)
    class(affinity_propagation), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: labels(size(x,1))
    real(dp), allocatable :: d(:,:)
    call require(allocated(self%labels_),'AffinityPropagation is not fitted')
    call require(allocated(self%cluster_centers_),'affinity prediction requires feature-input fitting')
    call check_xy(transpose(x),size(self%cluster_centers_,2))
    labels=-1
    if(size(self%cluster_centers_,1)>0) then
      d=pairwise_distances(x,self%cluster_centers_); labels=minloc(d,dim=2)-1
    end if
  end function
  subroutine spectral_fit(self,x)
    class(spectral_clustering), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    type(kmeans) :: model
    real(dp), allocatable :: work(:,:),rotation(:,:),u(:,:),v(:,:),s(:),direction(:)
    integer, allocatable :: pivots(:)
    integer :: n,k,i,j,idx
    call check_x(x); n=size(x,1); k=self%n_clusters
    call require(k>0.and.k<=n,'invalid spectral n_clusters')
    call require(self%assign_labels=='kmeans'.or.self%assign_labels=='cluster_qr','unknown spectral assignment')
    self%affinity_matrix_=spectral_affinity_matrix(x,self%affinity,self%gamma,self%n_neighbors)
    self%embedding_=spectral_eigenvectors(self%affinity_matrix_,k,.false.,self%n_connected_components_)
    if(self%assign_labels=='kmeans') then
      model%n_clusters=k; model%n_init=self%n_init; model%random_state=self%random_state
      call model%fit(self%embedding_); self%labels_=model%labels_
    else
      allocate(pivots(k),direction(k),u(k,k),s(k),v(k,k)); work=transpose(self%embedding_)
      do i=1,k
        idx=maxloc(sum(work**2,dim=1),dim=1); pivots(i)=idx; direction=work(:,idx)
        call require(sum(direction**2)>epsilon(1._dp)**2,'spectral QR embedding is rank deficient')
        direction=direction/sqrt(sum(direction**2))
        do j=1,n
          work(:,j)=work(:,j)-direction*dot_product(direction,work(:,j))
          work(:,j)=work(:,j)-direction*dot_product(direction,work(:,j))
        end do
      end do
      call dense_svd(transpose(self%embedding_(pivots,:)),s,u,v)
      rotation=matmul(u,transpose(v)); work=abs(matmul(self%embedding_,rotation))
      self%labels_=maxloc(work,dim=2)-1
    end if
  end subroutine
  function spectral_fit_predict(self,x) result(labels)
    class(spectral_clustering), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: labels(size(x,1))
    call self%fit(x); labels=self%labels_
  end function
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
