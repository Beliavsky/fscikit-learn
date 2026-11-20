! Dense sklearn 1.6.1 MiniBatchKMeans; upstream BSD provenance in THIRD_PARTY_NOTICES.md.
module sklearn_minibatch_kmeans
  use sklearn_common, only: dp,require,check_x
  use, intrinsic :: iso_fortran_env, only: int64
  use sklearn_legacy_rng, only: legacy_random_state
  use ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: mini_batch_kmeans
  type :: mini_batch_kmeans
    integer :: n_clusters=8,max_iter=100,batch_size=1024,n_init=0,init_size=0,random_state=0
    integer :: max_no_improvement=10,n_iter_=0,n_features_in_=0
    integer(int64) :: n_steps_=0
    character(16) :: init='k-means++'
    real(dp) :: tol=0,reassignment_ratio=.01_dp,inertia_=0
    logical :: compute_labels=.true.,converged_=.false.,has_inertia_=.false.
    real(dp), allocatable :: initial_centers(:,:),cluster_centers_(:,:),counts_(:)
    integer, allocatable :: labels_(:)
    type(legacy_random_state), private :: rng
    logical, private :: rng_ready=.false.
    integer, private :: effective_batch=0,effective_init=0,effective_n_init=0
    integer(int64), private :: since_reassign=0
    real(dp), private :: scaled_tol=0
  contains
    procedure, public :: fit
    procedure, public :: partial_fit
    procedure, public :: predict
    procedure, public :: transform
    procedure, public :: score
    procedure, public :: fit_predict
    procedure, public :: fit_transform
  end type
contains
  function weights_for(n,sample_weight) result(w)
    integer, intent(in) :: n
    real(dp), optional, intent(in) :: sample_weight(:)
    real(dp) :: w(n)
    w=1
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'MiniBatchKMeans sample weight shape mismatch')
      call require(all(ieee_is_finite(sample_weight)),'MiniBatchKMeans weights must be finite and nonnegative')
      call require(all(sample_weight>=0),'MiniBatchKMeans weights must be finite and nonnegative')
      w=sample_weight
    end if
    call require(ieee_is_finite(sum(w)),'MiniBatchKMeans total sample weight overflow')
  end function
  subroutine parameters(self,x)
    class(mini_batch_kmeans), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: mean(size(x,2))
    integer :: n
    integer(int64) :: initial_size
    call check_x(x); n=size(x,1)
    call require(self%n_clusters>=1.and.self%n_clusters<=n,'invalid MiniBatchKMeans n_clusters')
    call require(self%max_iter>0.and.self%batch_size>0,'invalid MiniBatchKMeans iteration or batch size')
    call require(self%n_init>=0.and.self%init_size>=0,'invalid MiniBatchKMeans initialization controls')
    call require(self%max_no_improvement>=-1,'invalid MiniBatchKMeans no-improvement limit')
    call require(self%random_state>=0,'MiniBatchKMeans seed must be nonnegative')
    call require(ieee_is_finite(self%tol),'MiniBatchKMeans tolerance must be finite and nonnegative')
    call require(self%tol>=0,'MiniBatchKMeans tolerance must be finite and nonnegative')
    call require(ieee_is_finite(self%reassignment_ratio),'MiniBatchKMeans reassignment ratio must be finite and nonnegative')
    call require(self%reassignment_ratio>=0,'MiniBatchKMeans reassignment ratio must be finite and nonnegative')
    call require(self%init=='random'.or.self%init=='k-means++','unknown MiniBatchKMeans initialization')
    self%effective_n_init=self%n_init
    if(self%n_init==0) then
      self%effective_n_init=1
      if(self%init=='random') self%effective_n_init=3
    end if
    if(allocated(self%initial_centers)) then
      call require(all(shape(self%initial_centers)==[self%n_clusters,size(x,2)]), &
        'MiniBatchKMeans initial center shape mismatch')
      call require(all(ieee_is_finite(self%initial_centers)),'MiniBatchKMeans initial centers must be finite')
      self%effective_n_init=1
    end if
    self%effective_batch=min(self%batch_size,n); initial_size=self%init_size
    if(self%init_size==0) then
      initial_size=3_int64*self%effective_batch
      if(initial_size<self%n_clusters) initial_size=3_int64*self%n_clusters
    else if(self%init_size<self%n_clusters) then
      initial_size=3_int64*self%n_clusters
    end if
    self%effective_init=int(min(initial_size,int(n,int64)))
    self%scaled_tol=0
    if(self%tol>0) then
      mean=sum(x,dim=1)/n
      self%scaled_tol=self%tol*sum(sum((x-spread(mean,1,n))**2,dim=1)/n)/size(x,2)
    end if
    self%n_features_in_=size(x,2)
  end subroutine
  function draw_indices(rng,n,k) result(ids)
    type(legacy_random_state), intent(inout) :: rng
    integer, intent(in) :: n,k
    integer :: ids(k),i
    do i=1,k
      ids(i)=rng%integer_index(n)
    end do
  end function
  function permutation(rng,n) result(ids)
    type(legacy_random_state), intent(inout) :: rng
    integer, intent(in) :: n
    integer :: ids(n),i,j,t
    ids=[(i,i=1,n)]
    do i=n,2,-1
      j=rng%integer_index(i); t=ids(i); ids(i)=ids(j); ids(j)=t
    end do
  end function
  function weighted_draw(rng,w,k) result(ids)
    type(legacy_random_state), intent(inout) :: rng
    real(dp), intent(in) :: w(:)
    integer, intent(in) :: k
    integer :: ids(k),batch(k),found,i,j,needed,index
    real(dp) :: remaining(size(w)),cdf(size(w)),u
    call require(count(w>0)>=k,'MiniBatchKMeans random init needs n_clusters positive-weight rows in subset')
    remaining=w/sum(w); found=0
    do while(found<k)
      if(found>0) remaining(ids(:found))=0
      cdf=remaining
      do i=2,size(w)
        cdf(i)=cdf(i-1)+cdf(i)
      end do
      cdf=cdf/cdf(size(w)); needed=k-found
      do j=1,needed
        u=rng%uniform(); index=size(w)
        do i=1,size(w)
          if(u<cdf(i)) then
            index=i; exit
          end if
        end do
        batch(j)=index
      end do
      ! np.unique(...,return_index=True), reordered by first occurrence.
      do j=1,needed
        if(any(batch(:j-1)==batch(j))) cycle
        found=found+1; ids(found)=batch(j)
      end do
    end do
  end function
  function initialize(self,x,w,rng) result(centers)
    class(mini_batch_kmeans), intent(in) :: self
    real(dp), intent(in) :: x(:,:),w(:)
    type(legacy_random_state), intent(inout) :: rng
    real(dp) :: centers(self%n_clusters,size(x,2))
    real(dp), allocatable :: z(:,:),v(:),dist(:),candidate(:),best_dist(:),cdf(:),u(:)
    integer, allocatable :: subset(:),ids(:)
    integer :: n,k,i,j,c,trials,chosen,best_id
    real(dp) :: cost,best,potential
    n=size(w); k=self%n_clusters
    if(self%effective_init<n) then
      subset=draw_indices(rng,n,self%effective_init); z=x(subset,:); v=w(subset)
    else
      z=x; v=w
    end if
    ! Source subsampling consumes RNG even when explicit centers were supplied.
    if(allocated(self%initial_centers)) then
      centers=self%initial_centers; return
    end if
    call require(sum(v)>0,'MiniBatchKMeans initialization subset needs positive total weight')
    if(self%init=='random') then
      ids=weighted_draw(rng,v,k); centers=z(ids,:); return
    end if
    ids=weighted_draw(rng,v,1); centers(1,:)=z(ids(1),:)
    n=size(v); allocate(dist(n),candidate(n),best_dist(n),cdf(n))
    do i=1,n
      dist(i)=max(0._dp,sum(centers(1,:)**2)-2*dot_product(centers(1,:),z(i,:))+sum(z(i,:)**2))
    end do
    potential=sum(v*dist); trials=2+int(log(real(k,dp))); allocate(u(trials))
    do c=2,k
      cdf=v*dist
      do i=2,n
        cdf(i)=cdf(i-1)+cdf(i)
      end do
      do j=1,trials
        u(j)=rng%uniform()*potential
      end do
      best=huge(1._dp); best_id=1
      do j=1,trials
        chosen=n
        do i=1,n
          if(cdf(i)>=u(j)) then
            chosen=i; exit
          end if
        end do
        do i=1,n
          candidate(i)=min(dist(i),max(0._dp,sum(z(chosen,:)**2)-2*dot_product(z(chosen,:),z(i,:))+sum(z(i,:)**2)))
        end do
        cost=sum(v*candidate)
        if(cost<best) then
          best=cost; best_id=chosen; best_dist=candidate
        end if
      end do
      centers(c,:)=z(best_id,:); potential=best; dist=best_dist
    end do
  end function
  function assignments(x,centers) result(labels)
    real(dp), intent(in) :: x(:,:),centers(:,:)
    integer :: labels(size(x,1)),i,j
    real(dp) :: norms(size(centers,1)),best,value
    norms=sum(centers**2,dim=2)
    do i=1,size(x,1)
      labels(i)=1; best=norms(1)-2*dot_product(x(i,:),centers(1,:))
      do j=2,size(centers,1)
        value=norms(j)-2*dot_product(x(i,:),centers(j,:))
        if(value<best) then
          best=value; labels(i)=j
        end if
      end do
    end do
  end function
  real(dp) function inertia(x,w,centers) result(value)
    real(dp), intent(in) :: x(:,:),w(:),centers(:,:)
    integer :: labels(size(w)),i
    labels=assignments(x,centers); value=0
    do i=1,size(w)
      value=value+w(i)*sum((x(i,:)-centers(labels(i),:))**2)
    end do
  end function
  subroutine update(self,x,w,rng,batch_inertia,shift)
    class(mini_batch_kmeans), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),w(:)
    type(legacy_random_state), intent(inout) :: rng
    real(dp), intent(out) :: batch_inertia,shift
    real(dp) :: old(size(self%cluster_centers_,1),size(x,2)),mass,min_count
    logical :: reassign(self%n_clusters),try_reassign
    integer :: labels(size(w)),order(self%n_clusters),i,j,k,t,number
    integer, allocatable :: pool(:)
    old=self%cluster_centers_; labels=assignments(x,old); batch_inertia=0
    self%since_reassign=self%since_reassign+self%effective_batch
    try_reassign=any(self%counts_==0).or.self%since_reassign>=10_int64*self%n_clusters
    if(try_reassign) self%since_reassign=0
    do i=1,size(w)
      batch_inertia=batch_inertia+w(i)*sum((x(i,:)-old(labels(i),:))**2)
    end do
    do j=1,self%n_clusters
      mass=sum(w,mask=labels==j)
      if(mass<=0) cycle
      self%cluster_centers_(j,:)=old(j,:)*self%counts_(j)
      do i=1,size(w)
        if(labels(i)==j) self%cluster_centers_(j,:)=self%cluster_centers_(j,:)+w(i)*x(i,:)
      end do
      self%counts_(j)=self%counts_(j)+mass
      self%cluster_centers_(j,:)=self%cluster_centers_(j,:)*(1/self%counts_(j))
    end do
    if(try_reassign.and.self%reassignment_ratio>0) then
      reassign=self%counts_<self%reassignment_ratio*maxval(self%counts_)
      if(count(reassign)>size(w)/2) then
        order=[(i,i=1,self%n_clusters)]
        do i=2,self%n_clusters
          t=order(i); j=i-1
          do while(j>=1)
            if(self%counts_(order(j))<=self%counts_(t)) exit
            order(j+1)=order(j); j=j-1
          end do
          order(j+1)=t
        end do
        reassign(order(size(w)/2+1:))=.false.
      end if
      number=count(reassign)
      if(number>0) then
        call require(number<self%n_clusters,'MiniBatchKMeans reassignment would reset every cluster')
        pool=permutation(rng,size(w)); k=0; min_count=minval(self%counts_,mask=.not.reassign)
        do j=1,self%n_clusters
          if(.not.reassign(j)) cycle
          k=k+1; self%cluster_centers_(j,:)=x(pool(k),:); self%counts_(j)=min_count
        end do
      end if
    end if
    shift=sum((old-self%cluster_centers_)**2)
  end subroutine
  subroutine fit(self,x,sample_weight)
    class(mini_batch_kmeans), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    type(legacy_random_state) :: rng
    real(dp) :: w(size(x,1)),best,value,batch_inertia,shift,ewa,minimum,alpha
    real(dp), allocatable :: centers(:,:)
    integer, allocatable :: ids(:),valid(:)
    integer :: n,trial,no_improvement
    integer(int64) :: step,steps
    logical :: have_minimum,have_ewa
    call parameters(self,x); n=size(x,1); w=weights_for(n,sample_weight)
    call rng%seed(self%random_state); valid=draw_indices(rng,n,self%effective_init); best=huge(1._dp)
    do trial=1,self%effective_n_init
      centers=initialize(self,x,w,rng); value=inertia(x(valid,:),w(valid),centers)
      if(trial==1.or.value<best) then
        best=value; self%cluster_centers_=centers
      end if
    end do
    self%counts_=spread(0._dp,1,self%n_clusters); self%since_reassign=0
    steps=int(self%max_iter,int64)*n/self%effective_batch; have_ewa=.false.; have_minimum=.false.
    no_improvement=0; ewa=0; minimum=0; self%converged_=.false.
    do step=1,steps
      ids=draw_indices(rng,n,self%effective_batch)
      call update(self,x(ids,:),w(ids),rng,batch_inertia,shift)
      if(step==1) cycle
      batch_inertia=batch_inertia/self%effective_batch
      if(.not.have_ewa) then
        ewa=batch_inertia; have_ewa=.true.
      else
        alpha=min(1._dp,2._dp*self%effective_batch/(n+1))
        ewa=ewa*(1-alpha)+batch_inertia*alpha
      end if
      if(self%scaled_tol>0.and.shift<=self%scaled_tol) then
        self%converged_=.true.; exit
      end if
      if(.not.have_minimum.or.ewa<minimum) then
        minimum=ewa; have_minimum=.true.; no_improvement=0
      else
        no_improvement=no_improvement+1
      end if
      if(self%max_no_improvement>=0.and.no_improvement>=self%max_no_improvement) then
        self%converged_=.true.; exit
      end if
    end do
    self%n_steps_=min(step,steps); self%n_iter_=int((self%n_steps_*self%effective_batch+n-1)/n)
    self%has_inertia_=self%compute_labels.or.have_ewa
    if(self%compute_labels) then
      self%labels_=assignments(x,self%cluster_centers_)-1; self%inertia_=inertia(x,w,self%cluster_centers_)
    else
      if(allocated(self%labels_)) deallocate(self%labels_)
      self%inertia_=ewa*n
    end if
  end subroutine
  subroutine partial_fit(self,x,sample_weight)
    class(mini_batch_kmeans), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    real(dp) :: w(size(x,1)),value,shift
    ! Use a local RNG copy to avoid aliasing self and its RNG component in update.
    type(legacy_random_state) :: rng
    call check_x(x); w=weights_for(size(x,1),sample_weight)
    if(.not.allocated(self%cluster_centers_)) call parameters(self,x)
    if(.not.self%rng_ready) then
      call self%rng%seed(self%random_state); self%rng_ready=.true.
    end if
    rng=self%rng
    if(.not.allocated(self%cluster_centers_)) then
      self%cluster_centers_=initialize(self,x,w,rng)
      self%counts_=spread(0._dp,1,self%n_clusters); self%since_reassign=0; self%n_steps_=0
    else
      call require(size(x,2)==self%n_features_in_,'MiniBatchKMeans query feature count mismatch')
      call require(size(self%cluster_centers_,1)==self%n_clusters,'MiniBatchKMeans n_clusters changed after fitting')
    end if
    call update(self,x,w,rng,value,shift); self%rng=rng
    if(self%compute_labels) then
      self%labels_=assignments(x,self%cluster_centers_)-1; self%inertia_=inertia(x,w,self%cluster_centers_)
      self%has_inertia_=.true.
    end if
    self%n_steps_=self%n_steps_+1
  end subroutine
  subroutine query_check(self,x)
    class(mini_batch_kmeans), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    call require(allocated(self%cluster_centers_),'MiniBatchKMeans is not fitted'); call check_x(x)
    call require(size(x,2)==self%n_features_in_,'MiniBatchKMeans query feature count mismatch')
  end subroutine
  function predict(self,x) result(labels)
    class(mini_batch_kmeans), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: labels(size(x,1))
    call query_check(self,x); labels=assignments(x,self%cluster_centers_)-1
  end function
  function transform(self,x) result(values)
    class(mini_batch_kmeans), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:,:)
    integer :: i,j
    call query_check(self,x); allocate(values(size(x,1),size(self%cluster_centers_,1)))
    do j=1,size(values,2)
      do i=1,size(x,1)
        values(i,j)=sqrt(sum((x(i,:)-self%cluster_centers_(j,:))**2))
      end do
    end do
  end function
  real(dp) function score(self,x,sample_weight) result(value)
    class(mini_batch_kmeans), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    call query_check(self,x); value=-inertia(x,weights_for(size(x,1),sample_weight),self%cluster_centers_)
  end function
  function fit_predict(self,x,sample_weight) result(labels)
    class(mini_batch_kmeans), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer :: labels(size(x,1))
    call require(self%compute_labels,'MiniBatchKMeans fit_predict requires compute_labels')
    call self%fit(x,sample_weight); labels=self%labels_
  end function
  function fit_transform(self,x,sample_weight) result(values)
    class(mini_batch_kmeans), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    real(dp), allocatable :: values(:,:)
    call self%fit(x,sample_weight); values=self%transform(x)
  end function
end module
