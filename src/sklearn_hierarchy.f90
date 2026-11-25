! Ward trees: sklearn 1.6.1 cluster/_agglomerative.py/_hierarchical_fast.pyx.
! Unstructured chain/relabeling: SciPy 1.15.3 cluster/_hierarchy.pyx and
! _hierarchy_distance_update.pxi. Retain BSD notices in LICENSES/ and
! THIRD_PARTY_NOTICES.md. Native typed storage/adapters are MIT.
module sklearn_hierarchy
  use sklearn_common,only:dp,require,check_x,argsort
  use sklearn_sparse,only:csr_matrix
  use sklearn_kernel_dispatch,only:kernel_blas_dot
  use,intrinsic::iso_fortran_env,only:int64
  use,intrinsic::ieee_arithmetic,only:ieee_is_finite,ieee_value,ieee_positive_inf
  use,intrinsic::ieee_exceptions
  implicit none
  private
  public::ward_tree,hierarchy_result
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
