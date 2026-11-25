! Ward, single, complete and average trees: sklearn 1.6.1 cluster/_agglomerative.py
! and _hierarchical_fast.pyx.
! Unstructured chain/relabeling: SciPy 1.15.3 cluster/_hierarchy.pyx and
! _hierarchy_distance_update.pxi. Retain BSD notices in LICENSES/ and
! THIRD_PARTY_NOTICES.md. Native typed storage/adapters are MIT.
module sklearn_hierarchy
  use sklearn_common,only:dp,require,check_x,argsort
  use sklearn_sparse,only:csr_matrix
  use sklearn_ranking,only:stable_order
  use sklearn_kernel_dispatch,only:kernel_blas_dot
  use,intrinsic::iso_fortran_env,only:int64
  use,intrinsic::ieee_arithmetic,only:ieee_is_finite,ieee_value,ieee_positive_inf
  use,intrinsic::ieee_exceptions
  implicit none
  private
  public::ward_tree,linkage_tree,hierarchy_result
  type::hierarchy_result
    integer::n_leaves=0,n_connected_components=1
    logical::structured=.false.,connectivity_completed=.false.
    integer,allocatable::children(:,:),parents(:)
    real(dp),allocatable::distances(:)
  end type
  type::neighbor_list
    integer::used=0
    integer,allocatable::id(:)
  end type
  type::weighted_neighbors
    integer::used=0
    integer,allocatable::id(:)
    real(dp),allocatable::cost(:)
  end type
  type::edge
    real(dp)::cost=0
    integer::a=0,b=0
  end type
  type::edge_heap
    type(edge),allocatable::data(:)
    integer::used=0
  end type
  interface ward_tree
    module procedure ward_dense,ward_vector,ward_real,ward_logical,ward_csr
    module procedure ward_real_vector,ward_logical_vector,ward_csr_vector
  end interface
  interface linkage_tree
    module procedure linkage_dense,linkage_vector,linkage_real,linkage_logical,linkage_csr
    module procedure linkage_real_vector,linkage_logical_vector,linkage_csr_vector
  end interface
contains
  function ward_dense(x,n_clusters,return_distance)result(tree)
    real(dp),intent(in)::x(:,:)
    integer,optional,intent(in)::n_clusters
    logical,optional,intent(in)::return_distance
    type(hierarchy_result)::tree
    logical::modes(size(ieee_all)),flags(size(ieee_all))
    logical::distances
    call check_x(x);call capacity(size(x,1))
    call require(size(x,1)>=2,'unstructured Ward needs at least two samples')
    distances=.false.;if(present(return_distance))distances=return_distance
    call fp_begin(modes,flags)
    tree=chain_tree(x,distances)
    call fp_end(modes,flags)
  end function
  function ward_vector(x,n_clusters,return_distance)result(tree)
    real(dp),intent(in)::x(:)
    integer,optional,intent(in)::n_clusters
    logical,optional,intent(in)::return_distance
    type(hierarchy_result)::tree
    tree=ward_dense(reshape(x,[size(x),1]),n_clusters,return_distance)
  end function
  function ward_real(x,connectivity,n_clusters,return_distance)result(tree)
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::connectivity(:,:)
    integer,optional,intent(in)::n_clusters
    logical,optional,intent(in)::return_distance
    type(hierarchy_result)::tree
    type(neighbor_list),allocatable::adjacency(:)
    logical::modes(size(ieee_all)),flags(size(ieee_all))
    integer::n,i,j
    real(dp)::row(size(x,1))
    call check_x(x);n=size(x,1);call capacity(n)
    call require(all(shape(connectivity)==[n,n]),'Ward connectivity must match sample count')
    call require(all(ieee_is_finite(connectivity)),'Ward connectivity must be finite')
    allocate(adjacency(n+(n-1)))
    call fp_begin(modes,flags)
    do i=1,n
      row=connectivity(i,:)+connectivity(:,i)
      adjacency(i)%id=pack([(j,j=1,n)],row/=0);adjacency(i)%used=size(adjacency(i)%id)
    end do
    tree=structured_tree(x,adjacency,n_clusters,return_distance)
    call fp_end(modes,flags)
  end function
  function ward_real_vector(x,connectivity,n_clusters,return_distance)result(tree)
    real(dp),intent(in)::x(:)
    real(dp),intent(in)::connectivity(:,:)
    integer,optional,intent(in)::n_clusters
    logical,optional,intent(in)::return_distance
    type(hierarchy_result)::tree
    tree=ward_real(reshape(x,[size(x),1]),connectivity,n_clusters,return_distance)
  end function
  function ward_logical(x,connectivity,n_clusters,return_distance)result(tree)
    real(dp),intent(in)::x(:,:)
    logical,intent(in)::connectivity(:,:)
    integer,optional,intent(in)::n_clusters
    logical,optional,intent(in)::return_distance
    type(hierarchy_result)::tree
    type(neighbor_list),allocatable::adjacency(:)
    logical::modes(size(ieee_all)),flags(size(ieee_all))
    integer::n,i,j
    logical::row(size(x,1))
    call check_x(x);n=size(x,1);call capacity(n)
    call require(all(shape(connectivity)==[n,n]),'Ward connectivity must match sample count')
    allocate(adjacency(n+(n-1)))
    call fp_begin(modes,flags)
    do i=1,n
      row=connectivity(i,:).or.connectivity(:,i)
      adjacency(i)%id=pack([(j,j=1,n)],row);adjacency(i)%used=size(adjacency(i)%id)
    end do
    tree=structured_tree(x,adjacency,n_clusters,return_distance)
    call fp_end(modes,flags)
  end function
  function ward_logical_vector(x,connectivity,n_clusters,return_distance)result(tree)
    real(dp),intent(in)::x(:)
    logical,intent(in)::connectivity(:,:)
    integer,optional,intent(in)::n_clusters
    logical,optional,intent(in)::return_distance
    type(hierarchy_result)::tree
    tree=ward_logical(reshape(x,[size(x),1]),connectivity,n_clusters,return_distance)
  end function
  function ward_csr(x,connectivity,n_clusters,return_distance)result(tree)
    real(dp),intent(in)::x(:,:)
    type(csr_matrix),intent(in)::connectivity
    integer,optional,intent(in)::n_clusters
    logical,optional,intent(in)::return_distance
    type(hierarchy_result)::tree
    type(neighbor_list),allocatable::adjacency(:)
    logical::modes(size(ieee_all)),flags(size(ieee_all))
    integer::n,i,j,k,position,count
    integer,allocatable::column_ptr(:),column_rows(:),offset(:),marks(:),touched(:)
    real(dp),allocatable::column_values(:),accumulated(:)
    call check_x(x);n=size(x,1);call capacity(n)
    call connectivity%validate()
    call require(connectivity%n_rows==n.and.connectivity%n_cols==n,'Ward connectivity must match sample count')
    call require(all(ieee_is_finite(connectivity%data)),'Ward connectivity must be finite')
    allocate(adjacency(n+(n-1)))
    call fp_begin(modes,flags)
    allocate(column_ptr(n+1),column_rows(size(connectivity%data)),column_values(size(connectivity%data)))
    allocate(marks(n),touched(n),accumulated(n));marks=0;column_ptr=0
    do k=1,size(connectivity%indices)
      j=connectivity%indices(k);column_ptr(j+1)=column_ptr(j+1)+1
    end do
    column_ptr(1)=1
    do j=1,n
      column_ptr(j+1)=column_ptr(j+1)+column_ptr(j)
    end do
    offset=column_ptr(:n)
    do i=1,n
      do k=connectivity%indptr(i),connectivity%indptr(i+1)-1
        j=connectivity%indices(k);position=offset(j)
        column_rows(position)=i;column_values(position)=connectivity%data(k);offset(j)=position+1
      end do
    end do
    do i=1,n
      count=0
      do k=connectivity%indptr(i),connectivity%indptr(i+1)-1
        call accumulate(connectivity%indices(k),connectivity%data(k))
      end do
      do k=column_ptr(i),column_ptr(i+1)-1
        call accumulate(column_rows(k),column_values(k))
      end do
      adjacency(i)%id=pack(touched(:count),accumulated(touched(:count))/=0)
      adjacency(i)%used=size(adjacency(i)%id)
    end do
    tree=structured_tree(x,adjacency,n_clusters,return_distance)
    call fp_end(modes,flags)
  contains
    subroutine accumulate(node,value)
      integer,intent(in)::node
      real(dp),intent(in)::value
      if(marks(node)/=i)then
        marks(node)=i;accumulated(node)=0;count=count+1;touched(count)=node
      end if
      accumulated(node)=accumulated(node)+value
    end subroutine
  end function
  function ward_csr_vector(x,connectivity,n_clusters,return_distance)result(tree)
    real(dp),intent(in)::x(:)
    type(csr_matrix),intent(in)::connectivity
    integer,optional,intent(in)::n_clusters
    logical,optional,intent(in)::return_distance
    type(hierarchy_result)::tree
    tree=ward_csr(reshape(x,[size(x),1]),connectivity,n_clusters,return_distance)
  end function
  function linkage_dense(x,n_clusters,return_distance,linkage,affinity)result(tree)
    real(dp),intent(in)::x(:,:)
    integer,optional,intent(in)::n_clusters
    logical,optional,intent(in)::return_distance
    character(*),optional,intent(in)::linkage,affinity
    character(16)::method,metric
    type(hierarchy_result)::tree
    logical::modes(size(ieee_all)),flags(size(ieee_all))
    logical::distances
    call linkage_settings(x,linkage,affinity,method,metric);call capacity(size(x,1))
    call require(size(x,1)>=2,'unstructured linkage needs at least two samples')
    distances=.false.;if(present(return_distance))distances=return_distance
    call fp_begin(modes,flags)
    if(method=='single')then
      tree=single_stream_tree(x,distances,metric)
    else
      tree=linkage_chain(x,distances,method,metric)
    end if
    call fp_end(modes,flags)
  end function
  function linkage_vector(x,n_clusters,return_distance,linkage,affinity)result(tree)
    real(dp),intent(in)::x(:)
    integer,optional,intent(in)::n_clusters
    logical,optional,intent(in)::return_distance
    character(*),optional,intent(in)::linkage,affinity
    character(16)::method,metric
    type(hierarchy_result)::tree
    tree=linkage_dense(reshape(x,[size(x),1]),n_clusters,return_distance,linkage,affinity)
  end function
  function linkage_real(x,connectivity,n_clusters,return_distance,linkage,affinity)result(tree)
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::connectivity(:,:)
    integer,optional,intent(in)::n_clusters
    logical,optional,intent(in)::return_distance
    character(*),optional,intent(in)::linkage,affinity
    character(16)::method,metric
    type(hierarchy_result)::tree
    type(neighbor_list),allocatable::adjacency(:)
    logical::modes(size(ieee_all)),flags(size(ieee_all))
    integer::n,i,j
    real(dp)::row(size(x,1))
    call linkage_settings(x,linkage,affinity,method,metric,.true.);n=size(x,1);call capacity(n)
    call require(all(shape(connectivity)==[n,n]),'linkage connectivity must match sample count')
    call require(all(ieee_is_finite(connectivity)),'linkage connectivity must be finite')
    allocate(adjacency(n+(n-1)))
    call fp_begin(modes,flags)
    do i=1,n
      row=connectivity(i,:)+connectivity(:,i)
      adjacency(i)%id=pack([(j,j=1,n)],row/=0);adjacency(i)%used=size(adjacency(i)%id)
    end do
    tree=structured_linkage(x,adjacency,n_clusters,return_distance,method,metric)
    call fp_end(modes,flags)
  end function
  function linkage_real_vector(x,connectivity,n_clusters,return_distance,linkage,affinity)result(tree)
    real(dp),intent(in)::x(:)
    real(dp),intent(in)::connectivity(:,:)
    integer,optional,intent(in)::n_clusters
    logical,optional,intent(in)::return_distance
    character(*),optional,intent(in)::linkage,affinity
    character(16)::method,metric
    type(hierarchy_result)::tree
    tree=linkage_real(reshape(x,[size(x),1]),connectivity,n_clusters,return_distance,linkage,affinity)
  end function
  function linkage_logical(x,connectivity,n_clusters,return_distance,linkage,affinity)result(tree)
    real(dp),intent(in)::x(:,:)
    logical,intent(in)::connectivity(:,:)
    integer,optional,intent(in)::n_clusters
    logical,optional,intent(in)::return_distance
    character(*),optional,intent(in)::linkage,affinity
    character(16)::method,metric
    type(hierarchy_result)::tree
    type(neighbor_list),allocatable::adjacency(:)
    logical::modes(size(ieee_all)),flags(size(ieee_all))
    integer::n,i,j
    logical::row(size(x,1))
    call linkage_settings(x,linkage,affinity,method,metric,.true.);n=size(x,1);call capacity(n)
    call require(all(shape(connectivity)==[n,n]),'linkage connectivity must match sample count')
    allocate(adjacency(n+(n-1)))
    call fp_begin(modes,flags)
    do i=1,n
      row=connectivity(i,:).or.connectivity(:,i)
      adjacency(i)%id=pack([(j,j=1,n)],row);adjacency(i)%used=size(adjacency(i)%id)
    end do
    tree=structured_linkage(x,adjacency,n_clusters,return_distance,method,metric)
    call fp_end(modes,flags)
  end function
  function linkage_logical_vector(x,connectivity,n_clusters,return_distance,linkage,affinity)result(tree)
    real(dp),intent(in)::x(:)
    logical,intent(in)::connectivity(:,:)
    integer,optional,intent(in)::n_clusters
    logical,optional,intent(in)::return_distance
    character(*),optional,intent(in)::linkage,affinity
    character(16)::method,metric
    type(hierarchy_result)::tree
    tree=linkage_logical(reshape(x,[size(x),1]),connectivity,n_clusters,return_distance,linkage,affinity)
  end function
  function linkage_csr(x,connectivity,n_clusters,return_distance,linkage,affinity)result(tree)
    real(dp),intent(in)::x(:,:)
    type(csr_matrix),intent(in)::connectivity
    integer,optional,intent(in)::n_clusters
    logical,optional,intent(in)::return_distance
    character(*),optional,intent(in)::linkage,affinity
    character(16)::method,metric
    type(hierarchy_result)::tree
    type(neighbor_list),allocatable::adjacency(:)
    logical::modes(size(ieee_all)),flags(size(ieee_all))
    integer::n,i,j,k,position,count
    integer,allocatable::column_ptr(:),column_rows(:),offset(:),marks(:),touched(:)
    real(dp),allocatable::column_values(:),accumulated(:)
    call linkage_settings(x,linkage,affinity,method,metric,.true.);n=size(x,1);call capacity(n)
    call connectivity%validate()
    call require(connectivity%n_rows==n.and.connectivity%n_cols==n,'linkage connectivity must match sample count')
    call require(all(ieee_is_finite(connectivity%data)),'linkage connectivity must be finite')
    allocate(adjacency(n+(n-1)))
    call fp_begin(modes,flags)
    allocate(column_ptr(n+1),column_rows(size(connectivity%data)),column_values(size(connectivity%data)))
    allocate(marks(n),touched(n),accumulated(n));marks=0;column_ptr=0
    do k=1,size(connectivity%indices)
      j=connectivity%indices(k);column_ptr(j+1)=column_ptr(j+1)+1
    end do
    column_ptr(1)=1
    do j=1,n
      column_ptr(j+1)=column_ptr(j+1)+column_ptr(j)
    end do
    offset=column_ptr(:n)
    do i=1,n
      do k=connectivity%indptr(i),connectivity%indptr(i+1)-1
        j=connectivity%indices(k);position=offset(j)
        column_rows(position)=i;column_values(position)=connectivity%data(k);offset(j)=position+1
      end do
    end do
    do i=1,n
      count=0
      do k=connectivity%indptr(i),connectivity%indptr(i+1)-1
        call accumulate(connectivity%indices(k),connectivity%data(k))
      end do
      do k=column_ptr(i),column_ptr(i+1)-1
        call accumulate(column_rows(k),column_values(k))
      end do
      adjacency(i)%id=pack(touched(:count),accumulated(touched(:count))/=0)
      adjacency(i)%used=size(adjacency(i)%id)
    end do
    tree=structured_linkage(x,adjacency,n_clusters,return_distance,method,metric)
    call fp_end(modes,flags)
  contains
    subroutine accumulate(node,value)
      integer,intent(in)::node
      real(dp),intent(in)::value
      if(marks(node)/=i)then
        marks(node)=i;accumulated(node)=0;count=count+1;touched(count)=node
      end if
      accumulated(node)=accumulated(node)+value
    end subroutine
  end function
  function linkage_csr_vector(x,connectivity,n_clusters,return_distance,linkage,affinity)result(tree)
    real(dp),intent(in)::x(:)
    type(csr_matrix),intent(in)::connectivity
    integer,optional,intent(in)::n_clusters
    logical,optional,intent(in)::return_distance
    character(*),optional,intent(in)::linkage,affinity
    character(16)::method,metric
    type(hierarchy_result)::tree
    tree=linkage_csr(reshape(x,[size(x),1]),connectivity,n_clusters,return_distance,linkage,affinity)
  end function
  ! sklearn 1.6.1 linkage_tree, max_merge/average_merge and WeightedEdge.
  ! SciPy 1.15.3 complete/average recurrences; CPython 3.13.3 heapq sift
  ! ordering. Upstream BSD/PSF notices are retained in LICENSES/.
  subroutine linkage_settings(x,linkage,affinity,method,metric,structured)
    real(dp),intent(in)::x(:,:)
    character(*),optional,intent(in)::linkage,affinity
    character(16),intent(out)::method,metric
    logical,optional,intent(in)::structured
    logical::singleton
    call check_x(x)
    if(present(linkage))call require(len_trim(linkage)<=16,'unsupported linkage name')
    if(present(affinity))call require(len_trim(affinity)<=16,'unsupported linkage affinity')
    method='complete';if(present(linkage))method=linkage
    call require(any(method==[character(16)::'single','complete','average']),'unknown linkage_tree method')
    metric='euclidean';if(present(affinity))metric=affinity
    if(present(structured))then
      if(structured)call require(any(metric==[character(16)::'euclidean','l2','manhattan','cityblock','l1', &
        'cosine','precomputed']),'structured linkage affinity is not supported by source paired_distances')
    end if
    select case(metric)
    case('l2');metric='euclidean'
    case('l1','cityblock');metric='manhattan'
    case('minkowski');metric='euclidean' ! Source default p=2, with no metric kwargs.
    end select
    if(method=='single')then
      if(metric=='p')metric='euclidean'
      if(metric=='infinity')metric='chebyshev'
    end if
    singleton=.false.
    if(present(structured))singleton=structured.and.method=='single'.and.metric=='precomputed'
    call require(size(x,1)>=2.or.singleton, &
      'linkage needs at least two samples except connected singleton precomputed single')
    call require(any(metric==[character(16)::'euclidean','manhattan','cosine','chebyshev', &
      'sqeuclidean','precomputed']),'unsupported linkage affinity')
    if(metric=='precomputed')call require(size(x,1)==size(x,2),'precomputed linkage matrix must be square')
    if(metric=='cosine')call require(all(any(x/=0,dim=2)),'cosine linkage cannot use zero vectors')
  end subroutine
  real(dp)function linkage_distance(x,i,j,metric,paired)result(value)
    real(dp),intent(in)::x(:,:)
    integer,intent(in)::i,j
    character(*),intent(in)::metric
    logical,intent(in)::paired
    real(dp)::nx,ny,product,difference,u(size(x,2)),v(size(x,2))
    integer::f
    value=0
    select case(metric)
    case('precomputed');value=x(i,j)
    case('cosine')
      nx=0;ny=0;product=0
      do f=1,size(x,2)
        nx=nx+x(i,f)*x(i,f);ny=ny+x(j,f)*x(j,f);product=product+x(i,f)*x(j,f)
      end do
      nx=sqrt(nx);ny=sqrt(ny)
      if(paired)then
        nx=sqrt(einsum_squares(x(i,:)));ny=sqrt(einsum_squares(x(j,:)))
        if(nx<10*epsilon(1._dp))nx=1
        if(ny<10*epsilon(1._dp))ny=1
        u=x(i,:)/nx;v=x(j,:)/ny
        value=.5_dp*einsum_squares(u-v)
      else
        product=product/(nx*ny)
        call require(ieee_is_finite(product),'cosine linkage exceeds finite working range')
        value=1-max(-1._dp,min(1._dp,product))
      end if
    case default
      if(paired.and.metric=='euclidean')then
        value=sqrt(einsum_squares(x(i,:)-x(j,:)))
        call require(ieee_is_finite(value),'linkage distances exceed finite working range')
        return
      else if(paired.and.metric=='manhattan')then
        value=linkage_numpy_sum(abs(x(i,:)-x(j,:)))
        call require(ieee_is_finite(value),'linkage distances exceed finite working range')
        return
      end if
      do f=1,size(x,2)
        difference=abs(x(i,f)-x(j,f))
        select case(metric)
        case('manhattan');value=value+difference
        case('chebyshev');value=max(value,difference)
        case default;value=value+difference*difference
        end select
      end do
      if(metric=='euclidean')value=sqrt(value)
    end select
    call require(ieee_is_finite(value),'linkage distances exceed finite working range')
  end function
  ! NumPy 2.2.6 einsum_sumprod.c.src, contiguous double sum-products.
  ! Copyright (c) 2011 Mark Wiebe, University of British Columbia.
  ! BSD terms retained in LICENSES/numpy-COPYING. The tested baseline uses
  ! two scalar lanes with eight-element blocks, matching its SSE2 reduction.
  real(dp)function einsum_squares(values)result(total)
    real(dp),intent(in)::values(:)
    real(dp)::squares(size(values)),a,b
    integer::i,n
    squares=values*values;n=size(values);a=0;b=0;i=1
    do while(i+7<=n)
      a=squares(i)+(squares(i+2)+(squares(i+4)+(squares(i+6)+a)))
      b=squares(i+1)+(squares(i+3)+(squares(i+5)+(squares(i+7)+b)))
      i=i+8
    end do
    do while(i<=n)
      a=a+squares(i)
      if(i<n)b=b+squares(i+1)
      i=i+2
    end do
    total=a+b
  end function
  recursive real(dp)function linkage_numpy_sum(values)result(total)
    ! NumPy _core/src/umath/loops_utils.h.src pairwise reduction, also used
    ! by this package's cluster_information and AdaBoost modules (BSD).
    real(dp),intent(in)::values(:)
    real(dp)::partial(8)
    integer::n,i,whole,split
    n=size(values)
    if(n<8)then
      total=-0._dp
      do i=1,n
        total=total+values(i)
      end do
    else if(n<=128)then
      partial=values(:8);whole=n-mod(n,8)
      do i=9,whole,8
        partial=partial+values(i:i+7)
      end do
      total=((partial(1)+partial(2))+(partial(3)+partial(4))) &
        +((partial(5)+partial(6))+(partial(7)+partial(8)))
      do i=whole+1,n
        total=total+values(i)
      end do
    else
      split=n/2;split=split-mod(split,8)
      total=linkage_numpy_sum(values(:split))+linkage_numpy_sum(values(split+1:))
    end if
  end function
  function structured_linkage(x,adjacency,n_clusters,return_distance,method,metric)result(tree)
    real(dp),intent(in)::x(:,:)
    type(neighbor_list),intent(inout)::adjacency(:)
    integer,optional,intent(in)::n_clusters
    logical,optional,intent(in)::return_distance
    character(*),intent(in)::method,metric
    type(hierarchy_result)::tree
    type(weighted_neighbors),allocatable::maps(:)
    type(edge_heap)::heap
    type(edge)::candidate
    integer,allocatable::labels(:),left(:),right(:),ids(:),order(:),mass(:)
    integer::n,nodes,clusters,ca,cb,ia,ib,i,j,k,q,a,b,na,nb,best_a,best_b,pa,pb,col
    real(dp)::best,distance,nx,ny,product,cost,u(size(x,2)),v(size(x,2))
    logical::distances
    n=size(x,1);clusters=1
    if(present(n_clusters))then
      if(n_clusters/=0)clusters=n_clusters
    end if
    ! Source's empty structured tree fails while forming paired distances or
    ! slicing an empty child array. Return a controlled diagnostic instead.
    if(method=='single')then
      call require(clusters>=1.and.clusters<=n,'single cluster count must be between one and sample count')
    else
      call require(clusters>=1.and.clusters<n,'structured complete/average needs fewer clusters than samples')
    end if
    nodes=n+(n-clusters);distances=.false.;if(present(return_distance))distances=return_distance
    tree%n_leaves=n;tree%structured=.true.
    allocate(labels(n));call components(adjacency,labels,tree%n_connected_components)
    tree%connectivity_completed=tree%n_connected_components>1
    do ca=2,tree%n_connected_components
      left=pack([(i,i=1,n)],labels==ca)
      do cb=1,ca-1
        right=pack([(i,i=1,n)],labels==cb);best=ieee_value(0._dp,ieee_positive_inf)
        do ia=1,size(left)
          i=left(ia)
          do ib=1,size(right)
            j=right(ib)
            select case(metric)
            case('euclidean')
              nx=einsum_squares(x(i,:));ny=einsum_squares(x(j,:))
              product=kernel_blas_dot(x(i,:),x(j,:),size(left),size(right),ia,ib)
              distance=(-2*product+nx)+ny
              call require(ieee_is_finite(distance),'linkage completion distances exceed working range')
              distance=sqrt(max(distance,0._dp))
            case('cosine')
              nx=sqrt(einsum_squares(x(i,:)));ny=sqrt(einsum_squares(x(j,:)))
              if(nx<10*epsilon(1._dp))nx=1
              if(ny<10*epsilon(1._dp))ny=1
              u=x(i,:)/nx;v=x(j,:)/ny
              product=kernel_blas_dot(u,v,size(left),size(right),ia,ib)
              distance=max(0._dp,min(2._dp,1-product))
            case default;distance=linkage_distance(x,i,j,metric,.false.)
            end select
            call require(ieee_is_finite(distance),'linkage completion distances exceed working range')
            if(distance<best)then
              best=distance;best_a=i;best_b=j
            end if
          end do
        end do
        call append_node(adjacency(best_a),best_b);call append_node(adjacency(best_b),best_a)
      end do
    end do
    if(method=='single')then
      tree=single_sparse_tree(x,adjacency,clusters,distances,metric,tree%n_connected_components)
      return
    end if
    allocate(maps(nodes),mass(nodes));mass=0;mass(:n)=1
    allocate(tree%parents(nodes),tree%children(nodes-n,2),tree%distances(nodes-n))
    tree%parents=[(i,i=1,nodes)]
    do i=1,n
      ids=pack(adjacency(i)%id(:adjacency(i)%used),adjacency(i)%id(:adjacency(i)%used)/=i)
      order=argsort(real(ids,dp));ids=ids(order)
      do q=1,size(ids)
        j=ids(q);cost=linkage_distance(x,i,j,metric,.true.)
        call append_weighted(maps(i),j,cost)
        if(j<i)call weighted_push(heap,edge(cost,i,j),.false.)
      end do
    end do
    ! heapq.heapify is not equivalent to inserting each item when comparisons
    ! ignore node IDs. Its bottom-up/right-on-equality traversal is retained.
    do i=heap%used/2,1,-1
      call weighted_sift(heap,i)
    end do
    do k=n+1,nodes
      do
        candidate=weighted_pop(heap);a=candidate%a;b=candidate%b
        if(mass(a)>0.and.mass(b)>0)exit
      end do
      na=mass(a);nb=mass(b);mass(k)=na+nb;mass(a)=0;mass(b)=0
      tree%parents(a)=k;tree%parents(b)=k;tree%children(k-n,:)=[b,a];tree%distances(k-n)=candidate%cost
      pa=1;pb=1
      do while(pa<=maps(a)%used.or.pb<=maps(b)%used)
        ia=huge(0);ib=huge(0)
        if(pa<=maps(a)%used)ia=maps(a)%id(pa)
        if(pb<=maps(b)%used)ib=maps(b)%id(pb)
        col=min(ia,ib)
        if(ia==ib)then
          if(method=='complete')then
            cost=max(maps(a)%cost(pa),maps(b)%cost(pb))
          else
            cost=(na*maps(a)%cost(pa)+nb*maps(b)%cost(pb))/real(na+nb,dp)
          end if
          pa=pa+1;pb=pb+1
        else if(ia<ib)then
          cost=maps(a)%cost(pa);pa=pa+1
        else
          cost=maps(b)%cost(pb);pb=pb+1
        end if
        if(mass(col)==0)cycle
        call require(ieee_is_finite(cost),'linkage merge distances exceed working range')
        call append_weighted(maps(k),col,cost);call append_weighted(maps(col),k,cost)
        call weighted_push(heap,edge(cost,k,col),.true.)
      end do
      deallocate(maps(a)%id,maps(a)%cost,maps(b)%id,maps(b)%cost)
    end do
    if(.not.distances)then
      deallocate(tree%distances);allocate(tree%distances(0))
    end if
  end function
  subroutine append_weighted(list,node,cost)
    type(weighted_neighbors),intent(inout)::list
    integer,intent(in)::node
    real(dp),intent(in)::cost
    integer,allocatable::ids(:)
    real(dp),allocatable::values(:)
    integer::capacity
    if(.not.allocated(list%id))allocate(list%id(4),list%cost(4))
    if(list%used==size(list%id))then
      call require(list%used<=huge(node)/2,'linkage adjacency capacity exceeded')
      capacity=max(4,2*list%used);allocate(ids(capacity),values(capacity))
      ids(:list%used)=list%id;values(:list%used)=list%cost
      call move_alloc(ids,list%id);call move_alloc(values,list%cost)
    end if
    list%used=list%used+1;list%id(list%used)=node;list%cost(list%used)=cost
  end subroutine
  subroutine weighted_push(heap,value,ordered)
    type(edge_heap),intent(inout)::heap
    type(edge),intent(in)::value
    logical,intent(in)::ordered
    type(edge),allocatable::grown(:)
    integer::i,parent,old
    if(.not.allocated(heap%data))allocate(heap%data(16))
    if(heap%used==size(heap%data))then
      old=size(heap%data);call require(old<=huge(old)/2,'linkage heap capacity exceeded')
      allocate(grown(2*old));grown(:old)=heap%data;call move_alloc(grown,heap%data)
    end if
    heap%used=heap%used+1;i=heap%used
    if(ordered)then
      do while(i>1)
        parent=i/2;if(value%cost>=heap%data(parent)%cost)exit
        heap%data(i)=heap%data(parent);i=parent
      end do
    end if
    heap%data(i)=value
  end subroutine
  subroutine weighted_sift(heap,start)
    type(edge_heap),intent(inout)::heap
    integer,intent(in)::start
    type(edge)::value
    integer::i,child,parent
    i=start;value=heap%data(i)
    do while(i<=heap%used/2)
      child=2*i
      if(child<heap%used)then
        if(heap%data(child)%cost>=heap%data(child+1)%cost)child=child+1
      end if
      heap%data(i)=heap%data(child);i=child
    end do
    do while(i>start)
      parent=i/2;if(value%cost>=heap%data(parent)%cost)exit
      heap%data(i)=heap%data(parent);i=parent
    end do
    heap%data(i)=value
  end subroutine
  function weighted_pop(heap)result(value)
    type(edge_heap),intent(inout)::heap
    type(edge)::value,last
    call require(heap%used>0,'linkage connectivity has no merge candidate')
    value=heap%data(1);last=heap%data(heap%used);heap%used=heap%used-1
    if(heap%used>0)then
      heap%data(1)=last;call weighted_sift(heap,1)
    end if
  end function
  ! sklearn 1.6.1 _hierarchical_fast.pyx mst_linkage_core and union-find;
  ! SciPy 1.15.3 _hierarchy.pyx mst_single_linkage/label. Fast named metrics
  ! preserve oriented children; SciPy's cosine/squared/precomputed route sorts
  ! child roots. Both use the same streamed frontier without a dense matrix.
  function single_stream_tree(x,distances,metric)result(tree)
    real(dp),intent(in)::x(:,:)
    logical,intent(in)::distances
    character(*),intent(in)::metric
    type(hierarchy_result)::tree
    integer,allocatable::raw(:,:)
    real(dp),allocatable::frontier(:),cost(:)
    logical,allocatable::visited(:)
    logical::sort_children
    integer::n,current,next,i,k
    real(dp)::value,best
    n=size(x,1);allocate(raw(n-1,2),cost(n-1),frontier(n),visited(n))
    frontier=ieee_value(0._dp,ieee_positive_inf);visited=.false.;current=1
    do k=1,n-1
      visited(current)=.true.;best=ieee_value(0._dp,ieee_positive_inf);next=0
      do i=1,n
        if(visited(i))cycle
        if(metric=='precomputed')then
          value=linkage_distance(x,min(current,i),max(current,i),metric,.false.)
        else
          value=linkage_distance(x,current,i,metric,.false.)
        end if
        if(value<frontier(i))frontier(i)=value
        if(frontier(i)<best)then
          best=frontier(i);next=i
        end if
      end do
      call require(next>0,'single-linkage frontier has no finite candidate')
      raw(k,:)=[current,next];cost(k)=best;current=next
    end do
    sort_children=.not.any(metric==[character(16)::'euclidean','manhattan','chebyshev'])
    tree=label_single(raw,cost,n,0,sort_children,distances,1)
  end function
  ! SciPy 1.15.3 sparse/csgraph/_min_spanning_tree.pyx Kruskal implementation
  ! (Copyright 2011 Jake Vanderplas, BSD; LICENSES/scipy-COPYING). Stable edge
  ! ordering, CSR output ordering and sklearn's epsilon substitution are kept.
  function single_sparse_tree(x,adjacency,clusters,distances,metric,components_count)result(tree)
    real(dp),intent(in)::x(:,:)
    type(neighbor_list),intent(in)::adjacency(:)
    integer,intent(in)::clusters,components_count
    logical,intent(in)::distances
    character(*),intent(in)::metric
    type(hierarchy_result)::tree
    integer,allocatable::rows(:),cols(:),order(:),ids(:),raw(:,:),parent(:),rank(:)
    real(dp),allocatable::cost(:),height(:)
    logical,allocatable::keep(:)
    integer(int64)::edge_count
    integer::n,e,i,j,k,pos,a,b,accepted
    real(dp)::value
    n=size(x,1);edge_count=0
    do i=1,n
      edge_count=edge_count+int(count(adjacency(i)%id(:adjacency(i)%used)/=i),int64)
    end do
    call require(edge_count<=huge(0),'single-linkage edge count exceeds native integer capacity')
    e=int(edge_count);allocate(rows(e),cols(e),cost(e),keep(e),parent(n),rank(n))
    pos=0;keep=.false.;rank=0;parent=[(i,i=1,n)]
    do i=1,n
      ids=pack(adjacency(i)%id(:adjacency(i)%used),adjacency(i)%id(:adjacency(i)%used)/=i)
      order=stable_order(real(ids,dp));ids=ids(order)
      do k=1,size(ids)
        j=ids(k);value=linkage_distance(x,i,j,metric,.true.)
        if(value==0)value=epsilon(1._dp)
        pos=pos+1;rows(pos)=i;cols(pos)=j;cost(pos)=value
      end do
    end do
    order=stable_order(cost);accepted=0
    do k=1,e
      if(accepted==n-1)exit
      j=order(k);a=mst_root(parent,rows(j));b=mst_root(parent,cols(j))
      if(a==b)cycle
      keep(j)=.true.;accepted=accepted+1
      if(rank(a)>rank(b))then
        parent(b)=a
      else if(rank(a)<rank(b))then
        parent(a)=b
      else
        parent(b)=a;rank(a)=rank(a)+1
      end if
    end do
    call require(accepted==n-1,'single-linkage completed graph did not yield a spanning tree')
    allocate(raw(n-1,2),height(n-1));pos=0
    ! csr->coo retains row/column order, rather than Kruskal acceptance order.
    do k=1,e
      if(.not.keep(k))cycle
      pos=pos+1;raw(pos,:)=[rows(k),cols(k)];height(pos)=cost(k)
      if(height(pos)==epsilon(1._dp))height(pos)=0
    end do
    tree=label_single(raw,height,n,n+(n-clusters),.false.,distances,components_count)
  end function
  function label_single(raw,cost,n,nodes,sort_children,distances,components_count)result(tree)
    integer,intent(in)::raw(:,:),n,nodes,components_count
    real(dp),intent(in)::cost(:)
    logical,intent(in)::sort_children,distances
    type(hierarchy_result)::tree
    integer,allocatable::parent(:),order(:)
    integer::k,a,b,i
    tree%n_leaves=n;tree%structured=nodes>0;tree%n_connected_components=components_count
    tree%connectivity_completed=components_count>1
    allocate(parent(n+(n-1)),tree%children(n-1,2),tree%distances(n-1),tree%parents(nodes))
    parent=[(i,i=1,size(parent))];tree%parents=[(i,i=1,nodes)];order=stable_order(cost)
    do k=1,n-1
      a=mst_root(parent,raw(order(k),1));b=mst_root(parent,raw(order(k),2))
      call require(a/=b,'single-linkage edge labels form a cycle')
      tree%children(k,:)=[a,b]
      if(sort_children)tree%children(k,:)=[min(a,b),max(a,b)]
      tree%distances(k)=cost(order(k));parent(a)=n+k;parent(b)=n+k
      if(n+k<=nodes)then
        tree%parents(a)=n+k;tree%parents(b)=n+k
      end if
    end do
    if(.not.distances)then
      deallocate(tree%distances);allocate(tree%distances(0))
    end if
  end function
  integer function mst_root(parent,node)result(root)
    integer,intent(inout)::parent(:)
    integer,intent(in)::node
    integer::i,next
    root=node
    do while(parent(root)/=root)
      root=parent(root)
    end do
    i=node
    do while(parent(i)/=i)
      next=parent(i);parent(i)=root;i=next
    end do
  end function
  subroutine capacity(n)
    integer,intent(in)::n
    call require(n<=huge(n)/2+1,'Ward node count exceeds native integer capacity')
  end subroutine
  subroutine fp_begin(modes,flags)
    logical,intent(out)::modes(:),flags(:)
    call ieee_get_flag(ieee_all,flags);call ieee_get_halting_mode(ieee_all,modes)
    call ieee_set_halting_mode(ieee_all,.false.)
  end subroutine
  subroutine fp_end(modes,flags)
    logical,intent(in)::modes(:),flags(:)
    call ieee_set_halting_mode(ieee_all,modes);call ieee_set_flag(ieee_all,flags)
  end subroutine
  integer(int64)function index_of(n,i,j)result(index)
    integer,intent(in)::n,i,j
    integer(int64)::a,b
    a=min(i,j);b=max(i,j)
    index=int(n,int64)*(a-1)-(a-1)*a/2+(b-a)
  end function
  function chain_tree(x,distances)result(tree)
    real(dp),intent(in)::x(:,:)
    logical,intent(in)::distances
    type(hierarchy_result)::tree
    real(dp),allocatable::d(:),height(:)
    integer,allocatable::sizes(:),chain(:),raw(:,:),order(:),parent(:)
    integer::n,i,j,k,a,b,na,nb,ni,length,temporary,root_a,root_b
    real(dp)::nearest,value,t,da,db
    n=size(x,1);tree%n_leaves=n
    allocate(d(int(n,int64)*(n-1)/2),sizes(n),chain(n),raw(n-1,2),height(n-1),parent(n+(n-1)))
    sizes=1;length=0
    do i=1,n
      do j=i+1,n
        value=sqrt(sum((x(i,:)-x(j,:))**2))
        call require(ieee_is_finite(value),'Ward initial distances exceed working range')
        d(index_of(n,i,j))=value
      end do
    end do
    do k=1,n-1
      if(length==0)then
        length=1
        do i=1,n
          if(sizes(i)>0)then
            chain(1)=i;exit
          end if
        end do
      end if
      do
        a=chain(length);nearest=ieee_value(0._dp,ieee_positive_inf)
        if(length>1)then
          b=chain(length-1);nearest=d(index_of(n,a,b))
        end if
        do i=1,n
          if(sizes(i)==0.or.i==a)cycle
          value=d(index_of(n,a,i))
          if(value<nearest)then
            nearest=value;b=i
          end if
        end do
        if(length>1)then
          if(b==chain(length-1))exit
        end if
        length=length+1;call require(length<=n,'Ward nearest-neighbor chain overflow')
        chain(length)=b
      end do
      length=length-2
      if(a>b)then
        temporary=a;a=b;b=temporary
      end if
      na=sizes(a);nb=sizes(b);raw(k,:)=[a,b];height(k)=nearest
      sizes(a)=0;sizes(b)=na+nb
      do i=1,n
        ni=sizes(i);if(ni==0.or.i==b)cycle
        da=d(index_of(n,i,a));db=d(index_of(n,i,b));t=1._dp/(na+nb+ni)
        value=sqrt((ni+na)*t*da*da+(ni+nb)*t*db*db-ni*t*nearest*nearest)
        call require(ieee_is_finite(value),'Ward updated distances exceed working range')
        d(index_of(n,i,b))=value
      end do
    end do
    order=argsort(height);parent=[(i,i=1,size(parent))]
    allocate(tree%children(n-1,2),tree%parents(0),tree%distances(n-1))
    do k=1,n-1
      root_a=head(parent,raw(order(k),1));root_b=head(parent,raw(order(k),2))
      tree%children(k,:)=[min(root_a,root_b),max(root_a,root_b)]
      parent(root_a)=n+k;parent(root_b)=n+k;tree%distances(k)=height(order(k))
    end do
    if(.not.distances)then
      deallocate(tree%distances);allocate(tree%distances(0))
    end if
  end function
  function linkage_chain(x,distances,method,metric)result(tree)
    real(dp),intent(in)::x(:,:)
    logical,intent(in)::distances
    character(*),intent(in)::method,metric
    type(hierarchy_result)::tree
    real(dp),allocatable::d(:),height(:)
    integer,allocatable::sizes(:),chain(:),raw(:,:),order(:),parent(:)
    integer::n,i,j,k,a,b,na,nb,ni,length,temporary,root_a,root_b
    real(dp)::nearest,value,t,da,db
    n=size(x,1);tree%n_leaves=n
    allocate(d(int(n,int64)*(n-1)/2),sizes(n),chain(n),raw(n-1,2),height(n-1),parent(n+(n-1)))
    sizes=1;length=0
    do i=1,n
      do j=i+1,n
        value=linkage_distance(x,i,j,metric,.false.)
        call require(ieee_is_finite(value),'linkage initial distances exceed working range')
        d(index_of(n,i,j))=value
      end do
    end do
    do k=1,n-1
      if(length==0)then
        length=1
        do i=1,n
          if(sizes(i)>0)then
            chain(1)=i;exit
          end if
        end do
      end if
      do
        a=chain(length);nearest=ieee_value(0._dp,ieee_positive_inf)
        if(length>1)then
          b=chain(length-1);nearest=d(index_of(n,a,b))
        end if
        do i=1,n
          if(sizes(i)==0.or.i==a)cycle
          value=d(index_of(n,a,i))
          if(value<nearest)then
            nearest=value;b=i
          end if
        end do
        if(length>1)then
          if(b==chain(length-1))exit
        end if
        length=length+1;call require(length<=n,'linkage nearest-neighbor chain overflow')
        chain(length)=b
      end do
      length=length-2
      if(a>b)then
        temporary=a;a=b;b=temporary
      end if
      na=sizes(a);nb=sizes(b);raw(k,:)=[a,b];height(k)=nearest
      sizes(a)=0;sizes(b)=na+nb
      do i=1,n
        ni=sizes(i);if(ni==0.or.i==b)cycle
        da=d(index_of(n,i,a));db=d(index_of(n,i,b))
        if(method=='complete')then
          value=max(da,db)
        else
          value=(na*da+nb*db)/(na+nb)
        end if
        call require(ieee_is_finite(value),'linkage updated distances exceed working range')
        d(index_of(n,i,b))=value
      end do
    end do
    order=argsort(height);parent=[(i,i=1,size(parent))]
    allocate(tree%children(n-1,2),tree%parents(0),tree%distances(n-1))
    do k=1,n-1
      root_a=head(parent,raw(order(k),1));root_b=head(parent,raw(order(k),2))
      tree%children(k,:)=[min(root_a,root_b),max(root_a,root_b)]
      parent(root_a)=n+k;parent(root_b)=n+k;tree%distances(k)=height(order(k))
    end do
    if(.not.distances)then
      deallocate(tree%distances);allocate(tree%distances(0))
    end if
  end function
  integer function head(parent,node)result(root)
    integer,intent(in)::parent(:),node
    root=node
    do while(parent(root)/=root)
      root=parent(root)
    end do
  end function
  subroutine append_node(list,node)
    type(neighbor_list),intent(inout)::list
    integer,intent(in)::node
    integer,allocatable::grown(:)
    integer::capacity
    if(.not.allocated(list%id))allocate(list%id(4))
    if(list%used==size(list%id))then
      call require(list%used<=huge(node)/2,'Ward adjacency capacity exceeded')
      capacity=max(4,2*list%used);allocate(grown(capacity))
      grown(:list%used)=list%id(:list%used);call move_alloc(grown,list%id)
    end if
    list%used=list%used+1;list%id(list%used)=node
  end subroutine
  subroutine components(adjacency,labels,count)
    type(neighbor_list),intent(in)::adjacency(:)
    integer,intent(out)::labels(:),count
    integer::queue(size(labels)),i,j,node,first,last
    labels=0;count=0
    do i=1,size(labels)
      if(labels(i)/=0)cycle
      count=count+1;first=1;last=1;queue(1)=i;labels(i)=count
      do while(first<=last)
        node=queue(first);first=first+1
        do j=1,adjacency(node)%used
          if(labels(adjacency(node)%id(j))/=0)cycle
          last=last+1;queue(last)=adjacency(node)%id(j);labels(queue(last))=count
        end do
      end do
    end do
  end subroutine
  function structured_tree(x,adjacency,n_clusters,return_distance)result(tree)
    real(dp),intent(in)::x(:,:)
    type(neighbor_list),intent(inout)::adjacency(:)
    integer,optional,intent(in)::n_clusters
    logical,optional,intent(in)::return_distance
    type(hierarchy_result)::tree
    type(edge_heap)::heap
    type(edge)::candidate
    integer,allocatable::labels(:),left(:),right(:),neighbors(:)
    real(dp),allocatable::mass(:),moments(:,:)
    logical,allocatable::active(:),seen(:)
    integer::n,nodes,clusters,i,j,a,b,k,q,ca,cb,ia,ib,best_a,best_b,r,neighbor_count
    real(dp)::best,distance,product,nx,ny
    logical::distances
    n=size(x,1);clusters=1
    if(present(n_clusters))then
      if(n_clusters/=0)clusters=n_clusters
    end if
    call require(clusters>=1.and.clusters<=n,'Ward cluster count must be between one and sample count')
    nodes=n+(n-clusters);distances=.false.;if(present(return_distance))distances=return_distance
    tree%n_leaves=n;tree%structured=.true.
    allocate(labels(n));call components(adjacency,labels,tree%n_connected_components)
    tree%connectivity_completed=tree%n_connected_components>1
    do ca=2,tree%n_connected_components
      left=pack([(i,i=1,n)],labels==ca)
      do cb=1,ca-1
        right=pack([(i,i=1,n)],labels==cb);best=ieee_value(0._dp,ieee_positive_inf)
        do ia=1,size(left)
          i=left(ia);nx=sum(x(i,:)**2)
          do ib=1,size(right)
            j=right(ib);ny=sum(x(j,:)**2)
            product=kernel_blas_dot(x(i,:),x(j,:),size(left),size(right),ia,ib)
            distance=sqrt(max((-2*product+nx)+ny,0._dp))
            call require(ieee_is_finite(distance),'Ward completion distances exceed working range')
            if(distance<best)then
              best=distance;best_a=i;best_b=j
            end if
          end do
        end do
        call append_node(adjacency(best_a),best_b);call append_node(adjacency(best_b),best_a)
      end do
    end do
    allocate(mass(nodes),moments(nodes,size(x,2)),active(nodes),seen(nodes))
    allocate(neighbors(nodes))
    mass=0;mass(:n)=1;moments=0;moments(:n,:)=x;active=.false.;active(:n)=.true.
    allocate(tree%parents(nodes),tree%children(nodes-n,2),tree%distances(nodes-n))
    tree%parents=[(i,i=1,nodes)]
    do i=1,n
      do q=1,adjacency(i)%used
        j=adjacency(i)%id(q);if(j>=i)cycle
        call push(heap,edge(ward_cost(i,j),i,j))
      end do
    end do
    do k=n+1,nodes
      do
        candidate=pop(heap);a=candidate%a;b=candidate%b
        if(active(a).and.active(b))exit
      end do
      tree%parents(a)=k;tree%parents(b)=k;tree%children(k-n,:)=[b,a]
      tree%distances(k-n)=sqrt(2*candidate%cost)
      active(a)=.false.;active(b)=.false.;active(k)=.true.
      mass(k)=mass(a)+mass(b);moments(k,:)=moments(a,:)+moments(b,:)
      neighbor_count=0;seen=.false.;seen(k)=.true.
      do r=1,2
        i=a;if(r==2)i=b
        do q=1,adjacency(i)%used
          j=head(tree%parents,adjacency(i)%id(q))
          if(seen(j))cycle
          seen(j)=.true.;neighbor_count=neighbor_count+1;neighbors(neighbor_count)=j
        end do
      end do
      do q=1,neighbor_count
        j=neighbors(q);call append_node(adjacency(j),k)
        call push(heap,edge(ward_cost(k,j),k,j))
      end do
      adjacency(k)%id=neighbors(:neighbor_count);adjacency(k)%used=neighbor_count
    end do
    if(.not.distances)then
      deallocate(tree%distances);allocate(tree%distances(0))
    end if
  contains
    real(dp)function ward_cost(a,b)result(cost)
      integer,intent(in)::a,b
      real(dp)::sum_squares,difference
      integer::feature
      sum_squares=0
      do feature=1,size(x,2)
        difference=moments(a,feature)/mass(a)-moments(b,feature)/mass(b)
        sum_squares=sum_squares+difference*difference
      end do
      cost=sum_squares*((mass(a)*mass(b))/(mass(a)+mass(b)))
      call require(ieee_is_finite(cost),'Ward inertia exceeds working range')
    end function
  end function
  logical function less(a,b)result(value)
    type(edge),intent(in)::a,b
    value=a%cost<b%cost
    if(a%cost==b%cost)then
      value=a%a<b%a
      if(a%a==b%a)value=a%b<b%b
    end if
  end function
  subroutine push(heap,value)
    type(edge_heap),intent(inout)::heap
    type(edge),intent(in)::value
    type(edge),allocatable::grown(:)
    integer::i,parent,old,capacity
    if(.not.allocated(heap%data))allocate(heap%data(16))
    if(heap%used==size(heap%data))then
      old=size(heap%data);call require(old<=huge(old)/2,'Ward heap capacity exceeded')
      capacity=2*old;allocate(grown(capacity));grown(:old)=heap%data
      call move_alloc(grown,heap%data)
    end if
    heap%used=heap%used+1;i=heap%used
    do while(i>1)
      parent=i/2;if(.not.less(value,heap%data(parent)))exit
      heap%data(i)=heap%data(parent);i=parent
    end do
    heap%data(i)=value
  end subroutine
  function pop(heap)result(value)
    type(edge_heap),intent(inout)::heap
    type(edge)::value,last
    integer::i,child
    call require(heap%used>0,'Ward connectivity has no merge candidate')
    value=heap%data(1);last=heap%data(heap%used);heap%used=heap%used-1;i=1
    do while(i<=heap%used/2)
      child=2*i
      if(child<heap%used)then
        if(less(heap%data(child+1),heap%data(child)))child=child+1
      end if
      if(.not.less(heap%data(child),last))exit
      heap%data(i)=heap%data(child);i=child
    end do
    if(heap%used>0)heap%data(i)=last
  end function
end module
