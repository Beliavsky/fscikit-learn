! sklearn 1.6.1 BisectingKMeans, weighted two-center seeding and Lloyd/Elkan updates.
! Independent dense implementation; BSD provenance retained in THIRD_PARTY_NOTICES.md.
module sklearn_bisecting_kmeans
  use sklearn_common, only: dp,require,check_x
  use sklearn_legacy_rng, only: legacy_random_state
  use ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: bisecting_kmeans
  type :: cluster_node
    real(dp), allocatable :: center(:)
    integer, allocatable :: indices(:)
    real(dp) :: score=0
    integer :: left=0,right=0,label=-1
  end type
  type :: bisecting_kmeans
    integer :: n_clusters=8,n_init=1,max_iter=300,random_state=0,n_features_in_=0,node_count_=0
    character(24) :: init='random',bisecting_strategy='biggest_inertia',algorithm='lloyd'
    real(dp) :: tol=1.e-4_dp,inertia_=0
    logical :: converged_=.false.
    real(dp), allocatable :: cluster_centers_(:,:),tree_centers_(:,:),tree_scores_(:)
    integer, allocatable :: labels_(:),children_(:,:),tree_leaf_labels_(:),tree_sample_counts_(:),n_iter_per_split_(:)
    real(dp), allocatable :: split_inertia_(:)
    logical, allocatable :: converged_per_split_(:)
  contains
    procedure :: fit
    procedure :: predict
    procedure :: transform
    procedure :: score
    procedure :: fit_predict
    procedure :: fit_transform
  end type
contains
  function checked_weights(n,sample_weight) result(w)
    integer, intent(in) :: n
    real(dp), optional, intent(in) :: sample_weight(:)
    real(dp) :: w(n)
    w=1
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'BisectingKMeans sample weight shape mismatch')
      call require(all(ieee_is_finite(sample_weight)),'BisectingKMeans weights must be finite and nonnegative')
      call require(all(sample_weight>=0),'BisectingKMeans weights must be finite and nonnegative')
      w=sample_weight
    end if
    call require(ieee_is_finite(sum(w)),'BisectingKMeans total sample weight overflow')
  end function
  integer function weighted_choice(rng,weights) result(index)
    type(legacy_random_state), intent(inout) :: rng
    real(dp), intent(in) :: weights(:)
    real(dp) :: cdf(size(weights)),u
    integer :: i
    call require(sum(weights)>0,'BisectingKMeans selected cluster needs positive total weight')
    cdf=weights/sum(weights)
    do i=2,size(weights)
      cdf(i)=cdf(i-1)+cdf(i)
    end do
    cdf=cdf/cdf(size(cdf)); u=rng%uniform(); index=size(cdf)
    do i=1,size(cdf)
      if(u<cdf(i)) then
        index=i
        exit
      end if
    end do
  end function
  function seed_centers(x,w,rng,method) result(centers)
    real(dp), intent(in) :: x(:,:),w(:)
    type(legacy_random_state), intent(inout) :: rng
    character(*), intent(in) :: method
    real(dp) :: centers(2,size(x,2)),remaining(size(w)),dist(size(w)),candidate_dist(size(w)),cdf(size(w))
    real(dp) :: potential,u(2),cost,best,norm
    integer :: first,second,j,i,chosen
    call require(size(w)>0,'BisectingKMeans selected cluster has no rows')
    if(method=='random') then
      call require(count(w>0)>=2,'BisectingKMeans random init needs two positive-weight rows in selected cluster')
      ! NumPy choice(replace=False,p=...) first draws a batch of two from
      ! the original CDF, then resamples only if those two indices coincide.
      first=weighted_choice(rng,w); second=weighted_choice(rng,w)
      if(first==second) then
        remaining=w; remaining(first)=0; second=weighted_choice(rng,remaining)
      end if
    else
      first=weighted_choice(rng,w)
      norm=sum(x(first,:)**2)
      do i=1,size(w)
        dist(i)=max(0._dp,norm-2*dot_product(x(first,:),x(i,:))+sum(x(i,:)**2))
      end do
      potential=sum(w*dist); cdf(1)=w(1)*dist(1)
      do i=2,size(w)
        cdf(i)=cdf(i-1)+w(i)*dist(i)
      end do
      u(1)=rng%uniform()*potential; u(2)=rng%uniform()*potential
      best=huge(1._dp); second=1
      do j=1,2
        chosen=size(w)
        do i=1,size(w)
          if(cdf(i)>=u(j)) then
            chosen=i
            exit
          end if
        end do
        norm=sum(x(chosen,:)**2)
        do i=1,size(w)
          candidate_dist(i)=min(dist(i),max(0._dp,norm-2*dot_product(x(chosen,:),x(i,:))+sum(x(i,:)**2)))
        end do
        cost=sum(candidate_dist*w)
        if(cost<best) then
          best=cost; second=chosen
        end if
      end do
    end if
    centers(1,:)=x(first,:); centers(2,:)=x(second,:)
  end function
  function assign(x,centers) result(labels)
    real(dp), intent(in) :: x(:,:),centers(:,:)
    integer :: labels(size(x,1)),i,j
    real(dp) :: norm(size(centers,1)),value,best
    norm=sum(centers**2,dim=2)
    do i=1,size(x,1)
      labels(i)=1; best=norm(1)-2*dot_product(x(i,:),centers(1,:))
      do j=2,size(centers,1)
        value=norm(j)-2*dot_product(x(i,:),centers(j,:))
        if(value<best) then
          best=value; labels(i)=j
        end if
      end do
    end do
  end function
  subroutine elkan_assign(x,centers,labels,upper,lower)
    real(dp), intent(in) :: x(:,:),centers(:,:)
    integer, intent(inout) :: labels(:)
    real(dp), intent(inout) :: upper(:),lower(:,:)
    real(dp) :: half,distance
    integer :: i,other,current
    half=sqrt(sum((centers(1,:)-centers(2,:))**2))/2
    do i=1,size(x,1)
      current=labels(i); other=3-current
      if(half>=upper(i)) cycle
      if(upper(i)<=lower(i,other).or.upper(i)<=half) cycle
      upper(i)=sqrt(sum((x(i,:)-centers(current,:))**2)); lower(i,current)=upper(i)
      if(upper(i)>lower(i,other).or.upper(i)>half) then
        distance=sqrt(sum((x(i,:)-centers(other,:))**2)); lower(i,other)=distance
        if(distance<upper(i)) then
          labels(i)=other; upper(i)=distance
        end if
      end if
    end do
  end subroutine
  subroutine two_means(x,w,centers,tolerance,max_iter,algorithm,labels,inertia,iterations,converged)
    real(dp), intent(in) :: x(:,:),w(:),tolerance
    real(dp), intent(inout) :: centers(:,:)
    integer, intent(in) :: max_iter
    character(*), intent(in) :: algorithm
    integer, intent(out) :: labels(:),iterations
    real(dp), intent(out) :: inertia
    logical, intent(out) :: converged
    real(dp) :: updated(2,size(x,2)),mass(2),shift(2),distance(size(w)),upper(size(w)),lower(size(w),2),half
    integer :: old(size(w)),i,j,c,far,largest,it
    logical :: strict
    old=0; strict=.false.; converged=.false.
    if(algorithm=='elkan') then
      lower=0; half=sqrt(sum((centers(1,:)-centers(2,:))**2))/2
      do i=1,size(w)
        labels(i)=1; upper(i)=sqrt(sum((x(i,:)-centers(1,:))**2)); lower(i,1)=upper(i)
        if(upper(i)>half) then
          lower(i,2)=sqrt(sum((x(i,:)-centers(2,:))**2))
          if(lower(i,2)<upper(i)) then
            labels(i)=2; upper(i)=lower(i,2)
          end if
        end if
      end do
    end if
    do it=1,max_iter
      if(algorithm=='elkan') then
        call elkan_assign(x,centers,labels,upper,lower)
      else
        labels=assign(x,centers)
      end if
      updated=0; mass=0
      do i=1,size(w)
        c=labels(i); mass(c)=mass(c)+w(i); updated(c,:)=updated(c,:)+w(i)*x(i,:)
      end do
      if(any(mass==0)) then
        do i=1,size(w)
          distance(i)=sum((x(i,:)-centers(labels(i),:))**2)
        end do
        if(maxval(distance)>0) then
          do j=1,2
            if(mass(j)/=0) cycle
            far=maxloc(distance,dim=1); c=labels(far)
            updated(c,:)=updated(c,:)-w(far)*x(far,:); updated(j,:)=w(far)*x(far,:)
            mass(j)=w(far); mass(c)=mass(c)-w(far); distance(far)=-1
          end do
        end if
      end if
      largest=maxloc(mass,dim=1)
      do j=1,2
        if(mass(j)>0) then
          updated(j,:)=updated(j,:)*(1/mass(j))
        else
          updated(j,:)=updated(largest,:)
        end if
        shift(j)=sqrt(sum((centers(j,:)-updated(j,:))**2))
      end do
      centers=updated
      if(algorithm=='elkan') then
        do i=1,size(w)
          upper(i)=upper(i)+shift(labels(i)); lower(i,:)=max(0._dp,lower(i,:)-shift)
        end do
      end if
      if(all(labels==old)) then
        strict=.true.; converged=.true.; exit
      end if
      if(sum(shift**2)<=tolerance) then
        converged=.true.; exit
      end if
      old=labels
    end do
    iterations=min(it,max_iter)
    if(.not.strict) then
      if(algorithm=='elkan') then
        call elkan_assign(x,centers,labels,upper,lower)
      else
        labels=assign(x,centers)
      end if
    end if
    inertia=0
    do i=1,size(w)
      inertia=inertia+w(i)*sum((x(i,:)-centers(labels(i),:))**2)
    end do
  end subroutine
  subroutine fit(self,x,sample_weight)
    class(bisecting_kmeans), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    type(legacy_random_state) :: rng
    type(cluster_node), allocatable :: nodes(:)
    real(dp) :: z(size(x,1),size(x,2)),mean(size(x,2)),weights(size(x,1)),best_inertia,inertia,scores(2)
    real(dp), allocatable :: centers(:,:),best_centers(:,:)
    integer, allocatable :: leaves(:),indices(:),labels(:),best_labels(:)
    integer :: n,p,node,split,trial,j,k,i,next,label_count,selected,best_iterations
    logical :: stopped,best_stopped
    call check_x(x); n=size(x,1); p=size(x,2)
    call require(self%n_clusters>=1.and.self%n_clusters<=n,'invalid BisectingKMeans n_clusters')
    call require(self%n_init>=1.and.self%max_iter>=1,'invalid BisectingKMeans iteration controls')
    call require(ieee_is_finite(self%tol),'BisectingKMeans tolerance must be finite and nonnegative')
    call require(self%tol>=0,'BisectingKMeans tolerance must be finite and nonnegative')
    call require(self%random_state>=0,'BisectingKMeans seed must be nonnegative')
    call require(self%init=='random'.or.self%init=='k-means++','unknown BisectingKMeans initialization')
    call require(self%algorithm=='lloyd'.or.self%algorithm=='elkan','unknown BisectingKMeans algorithm')
    call require(self%bisecting_strategy=='biggest_inertia'.or.self%bisecting_strategy=='largest_cluster', &
      'unknown BisectingKMeans strategy')
    weights=checked_weights(n,sample_weight); mean=sum(x,dim=1)/n; z=x-spread(mean,1,n)
    allocate(nodes(2*self%n_clusters-1)); nodes(1)%center=sum(z,dim=1)/n; nodes(1)%indices=[(i,i=1,n)]
    call rng%seed(self%random_state)
    self%n_iter_per_split_=spread(0,1,self%n_clusters-1); self%split_inertia_=spread(0._dp,1,self%n_clusters-1)
    self%converged_per_split_=spread(.false.,1,self%n_clusters-1); next=1
    do split=1,self%n_clusters-1
      leaves=leaf_order(1); selected=leaves(1)
      do j=2,size(leaves)
        if(nodes(leaves(j))%score>nodes(selected)%score) selected=leaves(j)
      end do
      indices=nodes(selected)%indices
      if(allocated(labels)) deallocate(labels)
      allocate(labels(size(indices))); best_inertia=huge(1._dp)
      do trial=1,self%n_init
        centers=seed_centers(z(indices,:),weights(indices),rng,self%init)
        call two_means(z(indices,:),weights(indices),centers,self%tol,self%max_iter,self%algorithm, &
          labels,inertia,k,stopped)
        if(trial==1.or.inertia<best_inertia*(1-1.e-6_dp)) then
          best_inertia=inertia; best_labels=labels; best_centers=centers; best_iterations=k; best_stopped=stopped
        end if
      end do
      scores=0
      if(self%bisecting_strategy=='largest_cluster') then
        scores=[real(count(best_labels==1),dp),real(count(best_labels==2),dp)]
      else
        do i=1,size(indices)
          j=best_labels(i)
          scores(j)=scores(j)+weights(indices(i))*sum((z(indices(i),:)-best_centers(j,:))**2)
        end do
      end if
      nodes(selected)%left=next+1; nodes(selected)%right=next+2
      do j=1,2
        node=next+j; nodes(node)%center=best_centers(j,:)
        nodes(node)%indices=pack(indices,best_labels==j); nodes(node)%score=scores(j)
      end do
      deallocate(nodes(selected)%indices); next=next+2
      self%n_iter_per_split_(split)=best_iterations; self%converged_per_split_(split)=best_stopped
      self%split_inertia_(split)=best_inertia
    end do
    self%node_count_=next; self%n_features_in_=p
    self%tree_centers_=spread(spread(0._dp,1,next),2,p); self%children_=spread(spread(0,1,next),2,2)
    self%tree_leaf_labels_=spread(-1,1,next); self%tree_scores_=spread(0._dp,1,next)
    self%tree_sample_counts_=spread(0,1,next)
    self%cluster_centers_=spread(spread(0._dp,1,self%n_clusters),2,p); self%labels_=spread(-1,1,n)
    leaves=leaf_order(1); label_count=0
    do j=1,size(leaves)
      node=leaves(j); self%cluster_centers_(j,:)=nodes(node)%center+mean
      self%labels_(nodes(node)%indices)=label_count; nodes(node)%label=label_count; label_count=label_count+1
    end do
    do node=next,1,-1
      self%tree_centers_(node,:)=nodes(node)%center+mean
      self%children_(node,:)=[nodes(node)%left,nodes(node)%right]
      self%tree_leaf_labels_(node)=nodes(node)%label; self%tree_scores_(node)=nodes(node)%score
      if(nodes(node)%left==0) then
        self%tree_sample_counts_(node)=size(nodes(node)%indices)
      else
        self%tree_sample_counts_(node)=self%tree_sample_counts_(nodes(node)%left)+ &
          self%tree_sample_counts_(nodes(node)%right)
      end if
    end do
    self%inertia_=0
    do i=1,n
      self%inertia_=self%inertia_+weights(i)*sum((z(i,:)+mean-self%cluster_centers_(self%labels_(i)+1,:))**2)
    end do
    self%converged_=all(self%converged_per_split_)
  contains
    recursive function leaf_order(node) result(order)
      integer, intent(in) :: node
      integer, allocatable :: order(:),left(:),right(:)
      if(nodes(node)%left==0) then
        order=[node]
      else
        left=leaf_order(nodes(node)%left); right=leaf_order(nodes(node)%right); order=[left,right]
      end if
    end function
  end subroutine
  subroutine check_query(self,x)
    class(bisecting_kmeans), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    call require(allocated(self%cluster_centers_),'BisectingKMeans is not fitted'); call check_x(x)
    call require(size(x,2)==self%n_features_in_,'BisectingKMeans query feature count mismatch')
  end subroutine
  function predict(self,x) result(labels)
    class(bisecting_kmeans), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: labels(size(x,1)),i,node,pair(2),assignment(1)
    call check_query(self,x)
    do i=1,size(x,1)
      node=1
      do while(self%children_(node,1)>0)
        pair=self%children_(node,:); assignment=assign(x(i:i,:),self%tree_centers_(pair,:))
        node=pair(assignment(1))
      end do
      labels(i)=self%tree_leaf_labels_(node)
    end do
  end function
  function transform(self,x) result(values)
    class(bisecting_kmeans), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:,:)
    integer :: i,j
    call check_query(self,x); allocate(values(size(x,1),size(self%cluster_centers_,1)))
    do j=1,size(values,2)
      do i=1,size(values,1)
        values(i,j)=sqrt(sum((x(i,:)-self%cluster_centers_(j,:))**2))
      end do
    end do
  end function
  real(dp) function score(self,x,sample_weight) result(value)
    class(bisecting_kmeans), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    real(dp) :: weights(size(x,1))
    integer :: labels(size(x,1)),i
    call check_query(self,x); weights=checked_weights(size(x,1),sample_weight)
    labels=assign(x,self%cluster_centers_); value=0
    do i=1,size(x,1)
      value=value-weights(i)*sum((x(i,:)-self%cluster_centers_(labels(i),:))**2)
    end do
  end function
  function fit_predict(self,x,sample_weight) result(labels)
    class(bisecting_kmeans), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer :: labels(size(x,1))
    call self%fit(x,sample_weight); labels=self%labels_
  end function
  function fit_transform(self,x,sample_weight) result(values)
    class(bisecting_kmeans), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    real(dp), allocatable :: values(:,:)
    call self%fit(x,sample_weight); values=self%transform(x)
  end function
end module
