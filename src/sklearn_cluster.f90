module sklearn_cluster
  use sklearn_dbscan,only:dbscan
  use sklearn_kmeans_core,only:k_means,kmeans_result,kmeans_labels,kmeans_distances,kmeans_inertia
  use sklearn_hierarchy,only:ward_tree,linkage_tree,hierarchy_result
  use sklearn_sparse,only:csr_matrix
  use sklearn_kmeans_seeding, only: kmeans_plusplus
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use,intrinsic::ieee_exceptions,only:ieee_all,ieee_get_flag,ieee_set_flag, &
    ieee_get_halting_mode,ieee_set_halting_mode
  use sklearn_bisecting_kmeans, only: bisecting_kmeans
  use sklearn_minibatch_kmeans, only: mini_batch_kmeans
  use sklearn_bicluster, only: spectral_coclustering,spectral_biclustering
  use sklearn_hdbscan, only: hdbscan
  use sklearn_common
  use sklearn_legacy_rng, only: legacy_random_state
  use, intrinsic :: iso_fortran_env, only: real32,int64
  use sklearn_metrics, only: pairwise_distances
  use sklearn_spectral, only: spectral_affinity_matrix,spectral_eigenvectors,spectral_discretize
  use sklearn_optics, only: optics,cluster_optics_dbscan,cluster_optics_xi,compute_optics_graph,optics_graph_result
  implicit none
  private
  public :: k_means,kmeans_result
  public :: kmeans_plusplus
  public :: bisecting_kmeans,mini_batch_kmeans
  public :: spectral_coclustering,spectral_biclustering
  public :: hdbscan
  public :: optics,cluster_optics_dbscan,cluster_optics_xi,compute_optics_graph,optics_graph_result
  public :: ward_tree,linkage_tree,hierarchy_result
  public :: kmeans,dbscan,agglomerative_clustering
  public :: mean_shift,estimate_bandwidth,get_bin_seeds,affinity_propagation,spectral_clustering
  public :: feature_agglomeration
  public :: birch
  ! CF tree contracts follow sklearn 1.6.1 cluster/_birch.py (BSD notices retained).
  type :: cf_summary
    integer :: count=0,child=0
    real(dp) :: squared_sum=0,norm=0
    real(dp), allocatable :: linear_sum(:),centroid(:)
  end type
  type :: cf_node
    integer :: used=0,previous=0,next=0
    logical :: leaf=.true.
    type(cf_summary), allocatable :: entries(:)
  end type
  type :: birch
    real(dp) :: threshold=.5_dp
    integer :: branching_factor=50,n_clusters=3,n_features_in_=0,n_samples_seen_=0
    logical :: compute_labels=.true.,insufficient_subclusters_=.false.
    real(dp), allocatable :: subcluster_centers_(:,:)
    integer, allocatable :: subcluster_labels_(:),labels_(:)
    type(cf_node), allocatable, private :: nodes(:)
    integer, private :: root=0,first_leaf=0,node_count=0,fitted_branching=0
    real(dp), private :: fitted_threshold=0
  contains
    procedure :: fit=>birch_fit
    procedure :: partial_fit=>birch_partial_fit
    procedure :: predict=>birch_predict
    procedure :: transform=>birch_transform
    procedure :: fit_predict=>birch_fit_predict
    procedure :: fit_transform=>birch_fit_transform
  end type
  type :: kmeans
    integer :: n_clusters=8,n_init=10,max_iter=300,random_state=0,n_iter_=0,n_features_in_=0,n_init_effective_=0
    real(dp) :: tol=1.e-4_dp,inertia_=0
    character(16)::init='k-means++',algorithm='lloyd'
    logical :: converged_=.false.
    real(dp), allocatable :: initial_centers(:,:),cluster_centers_(:,:)
    integer, allocatable :: labels_(:)
  contains
    procedure,private::fit_dense=>kmeans_fit,fit_sparse=>kmeans_fit_sparse
    generic::fit=>fit_dense,fit_sparse
    procedure,private::predict_dense=>kmeans_predict,predict_sparse=>kmeans_predict_sparse
    generic::predict=>predict_dense,predict_sparse
    procedure,private::transform_dense=>kmeans_transform,transform_sparse=>kmeans_transform_sparse
    generic::transform=>transform_dense,transform_sparse
    procedure,private::fit_predict_dense=>kmeans_fit_predict,fit_predict_sparse=>kmeans_fit_predict_sparse
    generic::fit_predict=>fit_predict_dense,fit_predict_sparse
    procedure,private::fit_transform_dense=>kmeans_fit_transform,fit_transform_sparse=>kmeans_fit_transform_sparse
    generic::fit_transform=>fit_transform_dense,fit_transform_sparse
    procedure,private::score_dense=>kmeans_score,score_sparse=>kmeans_score_sparse
    generic::score=>score_dense,score_sparse
  end type
  type :: agglomerative_clustering
    integer :: n_clusters=2,n_clusters_=0,n_features_in_=0,n_leaves_=0,n_connected_components_=0
    character(16) :: linkage='ward',compute_full_tree='auto',metric='euclidean'
    logical :: compute_distances=.true.,compute_full_tree_=.true.,connectivity_completed_=.false.
    real(dp),allocatable :: distance_threshold,connectivity(:,:)
    logical,allocatable :: connectivity_mask(:,:)
    type(csr_matrix),allocatable :: connectivity_csr
    integer, allocatable :: labels_(:),children_(:,:)
    real(dp), allocatable :: distances_(:)
  contains
    procedure :: fit=>agg_fit
    procedure :: fit_predict=>agg_fit_predict
  end type
  ! sklearn 1.6.1 cluster/_agglomerative.py and _feature_agglomeration.py (BSD).
  type :: feature_agglomeration
    integer :: n_clusters=2,n_clusters_=0,n_features_in_=0,n_leaves_=0,n_connected_components_=0
    character(16) :: linkage='ward',pooling_func='mean',compute_full_tree='auto',metric='euclidean'
    logical :: compute_distances=.true.,compute_full_tree_=.true.,connectivity_completed_=.false.
    real(dp),allocatable :: distance_threshold,connectivity(:,:)
    logical,allocatable :: connectivity_mask(:,:)
    type(csr_matrix),allocatable :: connectivity_csr
    integer, allocatable :: labels_(:),children_(:,:)
    real(dp), allocatable :: distances_(:)
  contains
    procedure :: fit=>feature_agg_fit
    procedure :: transform=>feature_agg_transform
    procedure :: inverse_transform=>feature_agg_inverse
    procedure :: fit_transform=>feature_agg_fit_transform
  end type
  type :: mean_shift
    real(dp) :: bandwidth=-1,bandwidth_=0
    integer :: max_iter=300,n_iter_=0,min_bin_freq=1
    logical :: cluster_all=.true.,converged_=.false.,bin_seeding=.false.,binning_failed_=.false.
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
    character(32) :: affinity='rbf'
    character(20) :: assign_labels='kmeans'
    integer, allocatable :: labels_(:)
    real(dp), allocatable :: affinity_matrix_(:,:),embedding_(:,:)
  contains
    procedure :: fit_dense=>spectral_fit,fit_csr=>spectral_fit_csr
    generic :: fit=>fit_dense,fit_csr
    procedure :: fit_predict_dense=>spectral_fit_predict,fit_predict_csr=>spectral_fit_predict_csr
    generic :: fit_predict=>fit_predict_dense,fit_predict_csr
  end type
contains
  ! sklearn 1.6.1 cluster/_mean_shift.py, _affinity_propagation.py and
  ! _spectral.py. See THIRD_PARTY_NOTICES.md for retained upstream attribution.
  ! sklearn 1.6.1 cluster/_mean_shift.py: insertion-ordered grid bins,
  ! nearest-even rounding and float32 seed storage; native queries stay real64.
  function get_bin_seeds(x,bin_size,min_bin_freq,binning_failed,float64_width) result(seeds)
    real(dp), intent(in) :: x(:,:),bin_size
    integer, optional, intent(in) :: min_bin_freq
    logical, optional, intent(out) :: binning_failed
    logical, optional, intent(in) :: float64_width
    logical :: wide
    real(dp), allocatable :: seeds(:,:),grid(:,:)
    real(dp) :: point(size(x,2)),v,t
    real(real32) :: width32,coordinate32
    integer, allocatable :: slots(:),frequency(:)
    integer :: n,p,threshold,capacity,i,j,pos,idx,used,selected,k
    integer(int64) :: hash,bits
    wide=.false.;if(present(float64_width)) wide=float64_width
    call check_x(x);n=size(x,1);p=size(x,2);threshold=1
    if(present(min_bin_freq)) threshold=min_bin_freq
    call require(threshold>=1,'min_bin_freq must be positive')
    call require(ieee_is_finite(bin_size),'bin_size must be finite')
    call require(bin_size>=0,'bin_size must be nonnegative')
    if(present(binning_failed)) binning_failed=.false.
    if(bin_size==0) then
      seeds=x;return
    end if
    if(bin_size<1) call require(maxval(abs(x))<=huge(1._dp)*bin_size,'bin coordinates overflow')
    call require(n<=huge(n)/4,'too many grid bin samples')
    capacity=1
    do while(capacity<2*n)
      capacity=2*capacity
    end do
    allocate(grid(n,p),frequency(n),slots(capacity));slots=0;frequency=0;used=0
    do i=1,n
      hash=0
      do j=1,p
        v=x(i,j)/bin_size;t=aint(v)
        if(abs(v-t)>.5_dp) t=t+sign(1._dp,v)
        if(abs(v-t)==.5_dp.and.mod(abs(t),2._dp)==1) t=t+sign(1._dp,v)
        if(t==0) t=0._dp
        point(j)=t;bits=transfer(t,bits)
        bits=ieor(bits,ishft(bits,-32));bits=ieor(bits,ishft(bits,-16))
        bits=ieor(bits,ishft(bits,-8));bits=ieor(bits,ishft(bits,-4))
        bits=ieor(bits,ishft(bits,-2));bits=ieor(bits,ishft(bits,-1))
        hash=ieor(ieor(ishft(hash,5),ishft(hash,-2)),bits)
      end do
      pos=int(iand(hash,int(capacity-1,int64)))+1
      do
        idx=slots(pos)
        if(idx==0) then
          used=used+1;idx=used;slots(pos)=idx;grid(idx,:)=point
          exit
        end if
        if(all(grid(idx,:)==point)) exit
        pos=mod(pos,capacity)+1
      end do
      frequency(idx)=frequency(idx)+1
    end do
    selected=count(frequency(:used)>=threshold)
    if(selected==n) then
      if(present(binning_failed)) binning_failed=.true.
      seeds=x;return
    end if
    allocate(seeds(selected,p));k=0
    if(selected==0) return
    if(.not.wide) then
      call require(bin_size<=real(huge(width32),dp),'bin_size exceeds float32 seed range')
      width32=real(bin_size,real32)
    end if
    do i=1,used
      if(frequency(i)<threshold) cycle
      k=k+1
      do j=1,p
        call require(abs(grid(i,j))<=real(huge(coordinate32),dp),'bin coordinate exceeds float32 seed range')
        coordinate32=real(grid(i,j),real32)
        if(wide) then
          if(bin_size>1) call require(abs(real(coordinate32,dp))<=huge(1._dp)/bin_size,'bin seed overflow')
          seeds(k,j)=real(coordinate32,dp)*bin_size
        else
          call require(abs(real(coordinate32,dp)*real(width32,dp))<=real(huge(width32),dp), &
            'bin seed exceeds float32 range')
          seeds(k,j)=real(coordinate32*width32,dp)
        end if
      end do
    end do
  end function
  real(dp) function estimate_bandwidth(x,quantile,n_samples,random_state) result(bandwidth)
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(in), optional :: quantile
    integer, optional, intent(in) :: n_samples,random_state
    real(dp) :: q
    real(dp), allocatable :: d(:,:),sample(:,:)
    integer, allocatable :: permutation(:),order(:)
    integer :: n,k,i,j,temp,seed
    type(legacy_random_state) :: rng
    call check_x(x);q=0.3_dp;if(present(quantile)) q=quantile
    call require(ieee_is_finite(q),'bandwidth quantile must be finite')
    call require(q>=0.and.q<=1,'bandwidth quantile must be in [0,1]')
    seed=0;if(present(random_state)) seed=random_state
    call rng%seed(seed)
    sample=x;n=size(x,1)
    if(present(n_samples)) then
      call require(n_samples>=1,'bandwidth n_samples must be positive')
      permutation=[(i,i=1,n)]
      do i=n,2,-1
        j=rng%integer_index(i);temp=permutation(i);permutation(i)=permutation(j);permutation(j)=temp
      end do
      sample=x(permutation(:min(n,n_samples)),:);n=size(sample,1)
    end if
    allocate(order(n));k=max(1,int(n*q));d=pairwise_distances(sample,sample);bandwidth=0
    do i=1,n
      order=argsort(d(i,:));bandwidth=bandwidth+d(i,order(k))
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
    call require(self%min_bin_freq>=1,'MeanShift min_bin_freq must be positive')
    call require(ieee_is_finite(self%bandwidth),'MeanShift bandwidth must be finite')
    call require(self%bandwidth>0.or.self%bandwidth==-1,'MeanShift bandwidth must be positive or -1')
    self%bandwidth_=self%bandwidth
    if(self%bandwidth==-1) self%bandwidth_=estimate_bandwidth(x)
    call check_x(reshape([self%bandwidth_],[1,1]))
    starts=x;self%binning_failed_=.false.
    if(present(seeds)) then
      call check_xy(transpose(seeds),p);starts=seeds
    else if(self%bin_seeding) then
      starts=get_bin_seeds(x,self%bandwidth_,self%min_bin_freq,self%binning_failed_,float64_width=self%bandwidth==-1)
    end if
    call require(size(starts,1)>0,'no MeanShift seeds satisfy min_bin_freq')
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
    call check_x(x)
    call require(size(x,1)>=2,'spectral clustering requires at least two samples')
    self%affinity_matrix_=spectral_affinity_matrix(x,self%affinity,self%gamma,self%n_neighbors)
    call spectral_finish(self)
  end subroutine
  subroutine spectral_fit_csr(self,x)
    class(spectral_clustering),intent(inout)::self
    type(csr_matrix),intent(in)::x
    self%affinity_matrix_=spectral_affinity_matrix(x,self%affinity,self%gamma,self%n_neighbors)
    call spectral_finish(self)
  end subroutine
  subroutine spectral_finish(self)
    class(spectral_clustering),intent(inout)::self
    type(kmeans) :: model
    type(legacy_random_state)::discrete_rng
    real(dp)::draw
    real(dp), allocatable :: work(:,:),rotation(:,:),u(:,:),v(:,:),s(:),direction(:)
    integer, allocatable :: pivots(:)
    integer :: n,k,i,j,idx
    n=size(self%affinity_matrix_,1); k=self%n_clusters
    call require(k>0.and.k<=n,'invalid spectral n_clusters')
    call require(any(self%assign_labels==[character(20)::'kmeans','cluster_qr','discretize']), &
      'unknown spectral assignment')
    self%embedding_=spectral_eigenvectors(self%affinity_matrix_,k,.false.,self%n_connected_components_)
    if(self%assign_labels=='kmeans') then
      model%n_clusters=k; model%n_init=self%n_init; model%random_state=self%random_state
      call model%fit(self%embedding_); self%labels_=model%labels_
    else if(self%assign_labels=='discretize')then
      call discrete_rng%seed(self%random_state)
      ! Default Source ARPACK initialization consumes one uniform per sample
      ! before handing the same RandomState to discretize.
      do i=1,n
        draw=discrete_rng%uniform()
      end do
      self%labels_=spectral_discretize(self%embedding_,rng=discrete_rng)
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
  function spectral_fit_predict_csr(self,x)result(labels)
    class(spectral_clustering),intent(inout)::self
    type(csr_matrix),intent(in)::x
    integer::labels(x%n_rows)
    call self%fit(x);labels=self%labels_
  end function
  subroutine kmeans_store(self,result)
    class(kmeans),intent(inout)::self
    type(kmeans_result),intent(in)::result
    self%cluster_centers_=result%centers;self%labels_=result%labels
    self%inertia_=result%inertia;self%n_iter_=result%n_iter;self%converged_=result%converged
    self%n_features_in_=size(result%centers,2);self%n_init_effective_=result%n_init
  end subroutine
  subroutine kmeans_fit(self,x,init_centers,sample_weight)
    class(kmeans),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    real(dp),optional,intent(in)::init_centers(:,:),sample_weight(:)
    type(kmeans_result)::result
    if(present(init_centers))then
      result=k_means(x,self%n_clusters,sample_weight=sample_weight,init=self%init,n_init=self%n_init, &
        max_iter=self%max_iter,tol=self%tol,random_state=self%random_state,init_centers=init_centers,algorithm=self%algorithm)
    else
      result=k_means(x,self%n_clusters,sample_weight=sample_weight,init=self%init,n_init=self%n_init, &
        max_iter=self%max_iter,tol=self%tol,random_state=self%random_state,init_centers=self%initial_centers, &
        algorithm=self%algorithm)
    end if
    call kmeans_store(self,result)
  end subroutine
  function kmeans_transform(self,x)result(d)
    class(kmeans),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),allocatable::d(:,:)
    call require(allocated(self%cluster_centers_),'kmeans is not fitted')
    d=kmeans_distances(x,self%cluster_centers_)
  end function
  function kmeans_predict(self,x)result(labels)
    class(kmeans),intent(in)::self
    real(dp),intent(in)::x(:,:)
    integer::labels(size(x,1))
    call require(allocated(self%cluster_centers_),'kmeans is not fitted')
    labels=kmeans_labels(x,self%cluster_centers_)
  end function
  function kmeans_fit_predict(self,x,sample_weight)result(labels)
    class(kmeans),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    integer::labels(size(x,1))
    call self%fit(x,sample_weight=sample_weight);labels=self%labels_
  end function
  function kmeans_fit_transform(self,x,sample_weight)result(d)
    class(kmeans),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    real(dp),allocatable::d(:,:)
    call self%fit(x,sample_weight=sample_weight);d=self%transform(x)
  end function
  real(dp)function kmeans_score(self,x,sample_weight)result(value)
    class(kmeans),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),optional,intent(in)::sample_weight(:)
    call require(allocated(self%cluster_centers_),'kmeans is not fitted')
    value=-kmeans_inertia(x,self%cluster_centers_,sample_weight)
  end function
  subroutine kmeans_fit_sparse(self,x,init_centers,sample_weight)
    class(kmeans),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),optional,intent(in)::init_centers(:,:),sample_weight(:)
    type(kmeans_result)::result
    if(present(init_centers))then
      result=k_means(x,self%n_clusters,sample_weight=sample_weight,init=self%init,n_init=self%n_init, &
        max_iter=self%max_iter,tol=self%tol,random_state=self%random_state,init_centers=init_centers,algorithm=self%algorithm)
    else
      result=k_means(x,self%n_clusters,sample_weight=sample_weight,init=self%init,n_init=self%n_init, &
        max_iter=self%max_iter,tol=self%tol,random_state=self%random_state,init_centers=self%initial_centers, &
        algorithm=self%algorithm)
    end if
    call kmeans_store(self,result)
  end subroutine
  function kmeans_transform_sparse(self,x)result(d)
    class(kmeans),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::d(:,:)
    call require(allocated(self%cluster_centers_),'kmeans is not fitted')
    d=kmeans_distances(x,self%cluster_centers_)
  end function
  function kmeans_predict_sparse(self,x)result(labels)
    class(kmeans),intent(in)::self
    type(csr_matrix),intent(in)::x
    integer::labels(x%n_rows)
    call require(allocated(self%cluster_centers_),'kmeans is not fitted')
    labels=kmeans_labels(x,self%cluster_centers_)
  end function
  function kmeans_fit_predict_sparse(self,x,sample_weight)result(labels)
    class(kmeans),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),optional,intent(in)::sample_weight(:)
    integer::labels(x%n_rows)
    call self%fit(x,sample_weight=sample_weight);labels=self%labels_
  end function
  function kmeans_fit_transform_sparse(self,x,sample_weight)result(d)
    class(kmeans),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),optional,intent(in)::sample_weight(:)
    real(dp),allocatable::d(:,:)
    call self%fit(x,sample_weight=sample_weight);d=self%transform(x)
  end function
  real(dp)function kmeans_score_sparse(self,x,sample_weight)result(value)
    class(kmeans),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),optional,intent(in)::sample_weight(:)
    call require(allocated(self%cluster_centers_),'kmeans is not fitted')
    value=-kmeans_inertia(x,self%cluster_centers_,sample_weight)
  end function
  subroutine agg_fit(self,x)
    class(agglomerative_clustering),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    call check_x(x)
    call require(size(x,1)>=2,'agglomerative fitting needs at least two samples')
    call require(any(self%compute_full_tree==[character(16)::'auto','true','false']),'invalid compute_full_tree')
    call require(any(self%linkage==[character(16)::'ward','single','complete','average']),'unknown linkage')
    call require(self%metric/='p'.and.self%metric/='infinity','p and infinity aliases are not valid estimator metrics')
    call agg_tree_fit(self,x)
  end subroutine
  ! sklearn 1.6.1 AgglomerativeClustering._fit: auto full-tree heuristic,
  ! threshold cluster counts and structured partial trees. Native labels retain
  ! first-sample order; estimator children retain their historical zero base.
  subroutine agg_tree_fit(self,x)
    class(agglomerative_clustering),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    type(hierarchy_result)::tree
    integer,allocatable::parent(:),root_labels(:)
    integer::n,clusters,requested,connections,i,k,root,node,next,total
    logical::full,distances,structured,threshold
    logical::modes(size(ieee_all)),flags(size(ieee_all))
    n=size(x,1);threshold=allocated(self%distance_threshold)
    if(self%linkage=='ward')call require(self%metric=='euclidean','Ward requires Euclidean metric')
    call require((self%n_clusters==0).eqv.threshold,'set exactly one of n_clusters or distance_threshold')
    if(threshold)then
      call require(ieee_is_finite(self%distance_threshold),'distance_threshold must be finite')
      call require(self%distance_threshold>=0,'distance_threshold must be nonnegative')
      call require(self%compute_full_tree/='false','distance_threshold requires a full tree')
    else
      call require(self%n_clusters>0.and.self%n_clusters<=n,'invalid agglomerative cluster count')
    end if
    connections=0
    if(allocated(self%connectivity))connections=connections+1
    if(allocated(self%connectivity_mask))connections=connections+1
    if(allocated(self%connectivity_csr))connections=connections+1
    call require(connections<=1,'choose one agglomerative connectivity representation')
    call ieee_get_flag(ieee_all,flags);call ieee_get_halting_mode(ieee_all,modes)
    call ieee_set_halting_mode(ieee_all,.false.)
    structured=connections==1;full=.true.
    if(structured)then
      select case(self%compute_full_tree)
      case('true');full=.true.
      case('false');full=.false.
      case('auto');full=threshold.or.self%n_clusters<max(100._dp,.02_dp*n)
      end select
    end if
    requested=0;if(.not.full)requested=self%n_clusters
    distances=threshold.or.self%compute_distances
    if(self%linkage=='ward')then
      if(allocated(self%connectivity))then
        tree=ward_tree(x,self%connectivity,requested,distances)
      else if(allocated(self%connectivity_mask))then
        tree=ward_tree(x,self%connectivity_mask,requested,distances)
      else if(allocated(self%connectivity_csr))then
        tree=ward_tree(x,self%connectivity_csr,requested,distances)
      else
        tree=ward_tree(x,n_clusters=requested,return_distance=distances)
      end if
    else
      if(allocated(self%connectivity))then
        tree=linkage_tree(x,self%connectivity,requested,distances,self%linkage,self%metric)
      else if(allocated(self%connectivity_mask))then
        tree=linkage_tree(x,self%connectivity_mask,requested,distances,self%linkage,self%metric)
      else if(allocated(self%connectivity_csr))then
        tree=linkage_tree(x,self%connectivity_csr,requested,distances,self%linkage,self%metric)
      else
        tree=linkage_tree(x,n_clusters=requested,return_distance=distances,linkage=self%linkage,affinity=self%metric)
      end if
    end if
    clusters=self%n_clusters
    if(threshold)clusters=count(tree%distances>=self%distance_threshold)+1
    ! Cutting a full tree by splitting its newest roots retains its first n-k
    ! merges. A partial tree already contains exactly that prefix. Path
    ! compression yields O(n + number_of_merges) storage without leaf matrices.
    allocate(parent(n+size(tree%children,1)),root_labels(n+size(tree%children,1)))
    parent=[(i,i=1,size(parent))];root_labels=-1
    do k=1,n-clusters
      parent(tree%children(k,:))=n+k
    end do
    if(allocated(self%labels_))deallocate(self%labels_)
    allocate(self%labels_(n));total=0
    do i=1,n
      root=i
      do while(parent(root)/=root)
        root=parent(root)
      end do
      node=i
      do while(parent(node)/=node)
        next=parent(node);parent(node)=root;node=next
      end do
      if(root_labels(root)<0)then
        root_labels(root)=total;total=total+1
      end if
      self%labels_(i)=root_labels(root)
    end do
    call require(total==clusters,'agglomerative tree cut produced an inconsistent forest')
    self%children_=tree%children-1;self%distances_=tree%distances
    self%n_clusters_=clusters;self%n_leaves_=tree%n_leaves;self%n_features_in_=size(x,2)
    self%n_connected_components_=tree%n_connected_components
    self%compute_full_tree_=full;self%connectivity_completed_=tree%connectivity_completed
    call ieee_set_halting_mode(ieee_all,modes);call ieee_set_flag(ieee_all,flags)
  end subroutine
  function agg_fit_predict(self,x) result(labels)
    class(agglomerative_clustering), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: labels(size(x,1))
    call self%fit(x); labels=self%labels_
  end function
  subroutine feature_agg_fit(self,x)
    class(feature_agglomeration), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    type(agglomerative_clustering) :: tree
    call check_x(x)
    call require(size(x,2)>=2,'FeatureAgglomeration needs at least two features')
    call require(any(self%pooling_func==['mean  ','median','max   ']),'unknown feature pooling function')
    tree%n_clusters=self%n_clusters; tree%linkage=self%linkage;tree%metric=self%metric
    tree%compute_full_tree=self%compute_full_tree;tree%compute_distances=self%compute_distances
    if(allocated(self%distance_threshold))tree%distance_threshold=self%distance_threshold
    if(allocated(self%connectivity))tree%connectivity=self%connectivity
    if(allocated(self%connectivity_mask))tree%connectivity_mask=self%connectivity_mask
    if(allocated(self%connectivity_csr))tree%connectivity_csr=self%connectivity_csr
    call tree%fit(transpose(x))
    self%labels_=tree%labels_; self%children_=tree%children_; self%distances_=tree%distances_
    self%n_clusters_=tree%n_clusters_; self%n_features_in_=size(x,2)
    self%n_leaves_=tree%n_leaves_; self%n_connected_components_=tree%n_connected_components_
    self%compute_full_tree_=tree%compute_full_tree_;self%connectivity_completed_=tree%connectivity_completed_
  end subroutine

  function feature_agg_transform(self,x) result(z)
    class(feature_agglomeration), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:),values(:)
    integer :: c,i,n
    call require(allocated(self%labels_),'FeatureAgglomeration is not fitted'); call check_x(x)
    call require(size(x,2)==self%n_features_in_,'FeatureAgglomeration feature count mismatch')
    call require(any(self%pooling_func==['mean  ','median','max   ']),'unknown feature pooling function')
    allocate(z(size(x,1),self%n_clusters_))
    do c=0,self%n_clusters_-1
      n=count(self%labels_==c)
      do i=1,size(x,1)
        values=pack(x(i,:),self%labels_==c)
        select case(self%pooling_func)
        case('mean'); z(i,c+1)=sum(values)/n
        case('max'); z(i,c+1)=maxval(values)
        case('median')
          values=values(argsort(values))
          z(i,c+1)=.5_dp*values((n+1)/2)+.5_dp*values(n/2+1)
        end select
      end do
    end do
  end function

  function feature_agg_inverse(self,z) result(x)
    class(feature_agglomeration), intent(in) :: self
    real(dp), intent(in) :: z(:,:)
    real(dp), allocatable :: x(:,:)
    call require(allocated(self%labels_),'FeatureAgglomeration is not fitted'); call check_x(z)
    call require(size(z,2)==self%n_clusters_,'FeatureAgglomeration pooled width mismatch')
    x=z(:,self%labels_+1)
  end function

  function feature_agg_fit_transform(self,x) result(z)
    class(feature_agglomeration), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x); z=self%transform(x)
  end function

  subroutine birch_controls(self)
    class(birch), intent(in) :: self
    call require(ieee_is_finite(self%threshold),'nonfinite BIRCH threshold')
    call require(self%threshold>0,'BIRCH threshold must be positive')
    call require(self%branching_factor>=2,'BIRCH branching_factor must be at least two')
    call require(self%n_clusters>=0,'BIRCH n_clusters must be nonnegative')
  end subroutine

  integer function birch_new_node(self,leaf) result(id)
    class(birch), intent(inout) :: self
    logical, intent(in) :: leaf
    type(cf_node), allocatable :: grown(:)
    if(self%node_count==size(self%nodes)) then
      allocate(grown(2*size(self%nodes)))
      grown(:self%node_count)=self%nodes
      call move_alloc(grown,self%nodes)
    end if
    self%node_count=self%node_count+1; id=self%node_count
    self%nodes(id)%leaf=leaf
    allocate(self%nodes(id)%entries(self%fitted_branching+1))
  end function

  subroutine cf_add(target,source)
    type(cf_summary), intent(inout) :: target
    type(cf_summary), intent(in) :: source
    if(target%count==0) then
      target%linear_sum=source%linear_sum
    else
      target%linear_sum=target%linear_sum+source%linear_sum
    end if
    target%count=target%count+source%count
    target%squared_sum=target%squared_sum+source%squared_sum
    target%centroid=target%linear_sum/real(target%count,dp)
    target%norm=sum(target%centroid**2)
  end subroutine

  subroutine cf_append(self,id,entry)
    class(birch), intent(inout) :: self
    integer, intent(in) :: id
    type(cf_summary), intent(in) :: entry
    integer :: k
    k=self%nodes(id)%used+1
    self%nodes(id)%entries(k)=entry; self%nodes(id)%used=k
  end subroutine

  subroutine birch_split(self,id,left,right)
    class(birch), intent(inout) :: self
    integer, intent(in) :: id
    type(cf_summary), intent(out) :: left,right
    type(cf_node) :: old
    real(dp) :: distance,best,d1,d2
    integer :: i,j,a,b,other,n
    old=self%nodes(id); n=old%used
    best=-1; a=1; b=1
    ! Row-major argmax order matches numpy's first farthest pair.
    do i=1,n
      do j=1,n
        distance=max(0._dp,old%entries(i)%norm+old%entries(j)%norm- &
          2*dot_product(old%entries(i)%centroid,old%entries(j)%centroid))
        if(i==j) distance=0
        if(distance>best) then
          best=distance; a=i; b=j
        end if
      end do
    end do
    other=birch_new_node(self,old%leaf)
    self%nodes(id)%used=0
    if(old%leaf) then
      self%nodes(id)%next=other
      self%nodes(other)%previous=id; self%nodes(other)%next=old%next
      if(old%next/=0) self%nodes(old%next)%previous=other
    end if
    left%child=id; right%child=other
    do i=1,n
      d1=max(0._dp,old%entries(a)%norm+old%entries(i)%norm- &
        2*dot_product(old%entries(a)%centroid,old%entries(i)%centroid))
      d2=max(0._dp,old%entries(b)%norm+old%entries(i)%norm- &
        2*dot_product(old%entries(b)%centroid,old%entries(i)%centroid))
      if(i==a) d1=0
      if(i==b) d2=0
      if(d1<d2 .or. i==a) then
        call cf_append(self,id,old%entries(i)); call cf_add(left,old%entries(i))
      else
        call cf_append(self,other,old%entries(i)); call cf_add(right,old%entries(i))
      end if
    end do
  end subroutine

  recursive logical function birch_insert(self,id,entry) result(split)
    class(birch), intent(inout) :: self
    integer, intent(in) :: id
    type(cf_summary), intent(in) :: entry
    type(cf_summary) :: merged,left,right
    real(dp) :: best,distance,radius
    integer :: i,closest,child,n
    n=self%nodes(id)%used; split=.false.
    if(n==0) then
      call cf_append(self,id,entry); return
    end if
    best=huge(1._dp); closest=1
    do i=1,n
      distance=self%nodes(id)%entries(i)%norm- &
        2*dot_product(self%nodes(id)%entries(i)%centroid,entry%centroid)
      if(distance<best) then
        best=distance; closest=i
      end if
    end do
    child=self%nodes(id)%entries(closest)%child
    if(child/=0) then
      if(birch_insert(self,child,entry)) then
        call birch_split(self,child,left,right)
        self%nodes(id)%entries(closest)=left; call cf_append(self,id,right)
      else
        call cf_add(self%nodes(id)%entries(closest),entry)
      end if
    else
      merged=self%nodes(id)%entries(closest); call cf_add(merged,entry)
      radius=merged%squared_sum/real(merged%count,dp)-merged%norm
      if(radius<=self%fitted_threshold**2) then
        self%nodes(id)%entries(closest)=merged
      else
        call cf_append(self,id,entry)
      end if
    end if
    split=self%nodes(id)%used>self%fitted_branching
  end function

  recursive subroutine birch_fit(self,x)
    class(birch), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    call birch_controls(self); call check_x(x)
    if(allocated(self%nodes)) deallocate(self%nodes)
    if(allocated(self%labels_)) deallocate(self%labels_)
    allocate(self%nodes(8)); self%node_count=0
    self%fitted_branching=self%branching_factor; self%fitted_threshold=self%threshold
    self%root=birch_new_node(self,.true.); self%first_leaf=self%root
    self%n_features_in_=size(x,2); self%n_samples_seen_=0
    call self%partial_fit(x)
  end subroutine

  recursive subroutine birch_partial_fit(self,x)
    class(birch), intent(inout) :: self
    real(dp), intent(in), optional :: x(:,:)
    type(cf_summary) :: entry,left,right
    integer :: i,new_root
    call birch_controls(self)
    if(.not.allocated(self%nodes)) then
      call require(present(x),'BIRCH is not fitted')
      call self%fit(x); return
    end if
    call require(self%branching_factor==self%fitted_branching .and. &
      self%threshold==self%fitted_threshold,'BIRCH tree controls changed: call fit to reset')
    if(present(x)) then
      call check_x(x)
      call require(size(x,2)==self%n_features_in_,'BIRCH feature count mismatch')
      do i=1,size(x,1)
        entry%count=1; entry%child=0; entry%linear_sum=x(i,:); entry%centroid=x(i,:)
        entry%squared_sum=sum(x(i,:)**2); entry%norm=entry%squared_sum
        if(birch_insert(self,self%root,entry)) then
          call birch_split(self,self%root,left,right)
          new_root=birch_new_node(self,.false.)
          call cf_append(self,new_root,left); call cf_append(self,new_root,right)
          self%root=new_root
        end if
      end do
      self%n_samples_seen_=self%n_samples_seen_+size(x,1)
    end if
    call birch_global(self)
    if(present(x)) then
      if(self%compute_labels) self%labels_=self%predict(x)
    end if
  end subroutine

  subroutine birch_global(self)
    class(birch), intent(inout) :: self
    type(agglomerative_clustering) :: ward
    integer :: leaf,n,i,k
    n=0; leaf=self%first_leaf
    do while(leaf/=0)
      n=n+self%nodes(leaf)%used; leaf=self%nodes(leaf)%next
    end do
    if(allocated(self%subcluster_centers_)) deallocate(self%subcluster_centers_)
    allocate(self%subcluster_centers_(n,self%n_features_in_))
    leaf=self%first_leaf; k=0
    do while(leaf/=0)
      do i=1,self%nodes(leaf)%used
        k=k+1; self%subcluster_centers_(k,:)=self%nodes(leaf)%entries(i)%centroid
      end do
      leaf=self%nodes(leaf)%next
    end do
    self%insufficient_subclusters_=n<self%n_clusters
    if(self%n_clusters==0 .or. self%insufficient_subclusters_) then
      self%subcluster_labels_=[(i-1,i=1,n)]
    else
      ward%n_clusters=self%n_clusters; call ward%fit(self%subcluster_centers_)
      self%subcluster_labels_=ward%labels_
    end if
  end subroutine

  function birch_predict(self,x) result(labels)
    class(birch), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: labels(:)
    real(dp), allocatable :: distance(:,:)
    call require(allocated(self%subcluster_centers_),'BIRCH is not fitted')
    call check_x(x); call require(size(x,2)==self%n_features_in_,'BIRCH feature count mismatch')
    distance=pairwise_distances(x,self%subcluster_centers_)
    labels=self%subcluster_labels_(minloc(distance,dim=2))
  end function

  function birch_transform(self,x) result(distance)
    class(birch), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: distance(:,:)
    call require(allocated(self%subcluster_centers_),'BIRCH is not fitted')
    call check_x(x); call require(size(x,2)==self%n_features_in_,'BIRCH feature count mismatch')
    distance=pairwise_distances(x,self%subcluster_centers_)
  end function

  function birch_fit_predict(self,x) result(labels)
    class(birch), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: labels(:)
    call require(self%compute_labels,'BIRCH fit_predict requires compute_labels')
    call self%fit(x); labels=self%labels_
  end function

  function birch_fit_transform(self,x) result(distance)
    class(birch), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: distance(:,:)
    call self%fit(x); distance=self%transform(x)
  end function
end module
