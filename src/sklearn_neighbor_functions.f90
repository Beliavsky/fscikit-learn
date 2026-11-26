! Functional sklearn 1.6.1 neighbors/_graph.py and stable CSR value ordering (BSD).
! Native typed storage and streaming adapters are MIT; see THIRD_PARTY_NOTICES.md.
module sklearn_neighbor_functions
 use sklearn_common,only:dp,require
 use sklearn_sparse,only:csr_matrix
 use sklearn_pairwise_reductions,only:row_view,check_view,get_row,row_distance
 use sklearn_pairwise_chunks,only:pairwise_distances_chunked,pairwise_distance_iterator,distance_chunk
 use sklearn_neighbor_selection,only:argkmin_order,pairwise_knn_order
 use sklearn_neighbor_tree,only:neighbor_kd_tree
 use sklearn_neighbor_gram,only:neighbor_row_norm,neighbor_middle_term
 use,intrinsic::ieee_arithmetic
 use,intrinsic::ieee_exceptions
 use,intrinsic::iso_fortran_env,only:int64
 implicit none
 private
 public::kneighbors_graph,radius_neighbors_graph,sort_graph_by_row_values
 public::brute_kneighbors_graph,brute_radius_neighbors_graph
 interface kneighbors_graph
  module procedure kn_graph_dense,kn_graph_sparse,kn_graph_dense_auto,kn_graph_sparse_auto
 end interface
 interface radius_neighbors_graph
  module procedure rn_graph_dense,rn_graph_sparse,rn_graph_dense_auto,rn_graph_sparse_auto
 end interface
contains
 function brute_kneighbors_graph(x,n_neighbors,mode,metric,p,include_self)result(graph)
  real(dp),target,intent(in)::x(:,:)
  integer,intent(in)::n_neighbors
  character(*),optional,intent(in)::mode,metric
  real(dp),optional,intent(in)::p
  logical,intent(in)::include_self
  type(csr_matrix)::graph
  type(row_view)::v
  v%dense=>x;v%n=size(x,1);v%p=size(x,2)
  graph=graph_from_view(v,n_neighbors,0._dp,.false.,mode,metric,p,include_self,force_brute=.true.)
 end function
 function brute_radius_neighbors_graph(x,radius,mode,metric,p,include_self)result(graph)
  real(dp),target,intent(in)::x(:,:)
  real(dp),intent(in)::radius
  character(*),optional,intent(in)::mode,metric
  real(dp),optional,intent(in)::p
  logical,intent(in)::include_self
  type(csr_matrix)::graph
  type(row_view)::v
  v%dense=>x;v%n=size(x,1);v%p=size(x,2)
  graph=graph_from_view(v,0,radius,.true.,mode,metric,p,include_self,force_brute=.true.)
 end function
 function kn_graph_dense(x,n_neighbors,mode,metric,p,include_self)result(graph)
  real(dp),target,intent(in)::x(:,:)
  integer,intent(in)::n_neighbors
  character(*),optional,intent(in)::mode,metric
  real(dp),optional,intent(in)::p
  logical,optional,intent(in)::include_self
  type(csr_matrix)::graph
  type(row_view)::v
  logical::same
  same=.false.;if(present(include_self))same=include_self
  v%dense=>x;v%n=size(x,1);v%p=size(x,2)
  graph=graph_from_view(v,n_neighbors,0._dp,.false.,mode,metric,p,same)
 end function
 function kn_graph_dense_auto(x,n_neighbors,mode,metric,p,include_self)result(graph)
  real(dp),target,intent(in)::x(:,:)
  integer,intent(in)::n_neighbors
  character(*),optional,intent(in)::mode,metric
  real(dp),optional,intent(in)::p
  character(*),intent(in)::include_self
  type(csr_matrix)::graph
  type(row_view)::v
  logical::same
  call require(include_self=='auto','include_self string must be auto')
  same=.true.;if(present(mode))same=mode=='connectivity'
  v%dense=>x;v%n=size(x,1);v%p=size(x,2)
  graph=graph_from_view(v,n_neighbors,0._dp,.false.,mode,metric,p,same)
 end function
 function kn_graph_sparse(x,n_neighbors,mode,metric,p,include_self)result(graph)
  type(csr_matrix),target,intent(in)::x
  integer,intent(in)::n_neighbors
  character(*),optional,intent(in)::mode,metric
  real(dp),optional,intent(in)::p
  logical,optional,intent(in)::include_self
  type(csr_matrix)::graph
  type(row_view)::v
  logical::same
  same=.false.;if(present(include_self))same=include_self
  v%sparse=>x;v%n=x%n_rows;v%p=x%n_cols
  graph=graph_from_view(v,n_neighbors,0._dp,.false.,mode,metric,p,same)
 end function
 function kn_graph_sparse_auto(x,n_neighbors,mode,metric,p,include_self)result(graph)
  type(csr_matrix),target,intent(in)::x
  integer,intent(in)::n_neighbors
  character(*),optional,intent(in)::mode,metric
  real(dp),optional,intent(in)::p
  character(*),intent(in)::include_self
  type(csr_matrix)::graph
  type(row_view)::v
  logical::same
  call require(include_self=='auto','include_self string must be auto')
  same=.true.;if(present(mode))same=mode=='connectivity'
  v%sparse=>x;v%n=x%n_rows;v%p=x%n_cols
  graph=graph_from_view(v,n_neighbors,0._dp,.false.,mode,metric,p,same)
 end function
 function rn_graph_dense(x,radius,mode,metric,p,include_self)result(graph)
  real(dp),target,intent(in)::x(:,:)
  real(dp),intent(in)::radius
  character(*),optional,intent(in)::mode,metric
  real(dp),optional,intent(in)::p
  logical,optional,intent(in)::include_self
  type(csr_matrix)::graph
  type(row_view)::v
  logical::same
  same=.false.;if(present(include_self))same=include_self
  v%dense=>x;v%n=size(x,1);v%p=size(x,2)
  graph=graph_from_view(v,0,radius,.true.,mode,metric,p,same)
 end function
 function rn_graph_dense_auto(x,radius,mode,metric,p,include_self)result(graph)
  real(dp),target,intent(in)::x(:,:)
  real(dp),intent(in)::radius
  character(*),optional,intent(in)::mode,metric
  real(dp),optional,intent(in)::p
  character(*),intent(in)::include_self
  type(csr_matrix)::graph
  type(row_view)::v
  logical::same
  call require(include_self=='auto','include_self string must be auto')
  same=.true.;if(present(mode))same=mode=='connectivity'
  v%dense=>x;v%n=size(x,1);v%p=size(x,2)
  graph=graph_from_view(v,0,radius,.true.,mode,metric,p,same)
 end function
 function rn_graph_sparse(x,radius,mode,metric,p,include_self)result(graph)
  type(csr_matrix),target,intent(in)::x
  real(dp),intent(in)::radius
  character(*),optional,intent(in)::mode,metric
  real(dp),optional,intent(in)::p
  logical,optional,intent(in)::include_self
  type(csr_matrix)::graph
  type(row_view)::v
  logical::same
  same=.false.;if(present(include_self))same=include_self
  v%sparse=>x;v%n=x%n_rows;v%p=x%n_cols
  graph=graph_from_view(v,0,radius,.true.,mode,metric,p,same)
 end function
 function rn_graph_sparse_auto(x,radius,mode,metric,p,include_self)result(graph)
  type(csr_matrix),target,intent(in)::x
  real(dp),intent(in)::radius
  character(*),optional,intent(in)::mode,metric
  real(dp),optional,intent(in)::p
  character(*),intent(in)::include_self
  type(csr_matrix)::graph
  type(row_view)::v
  logical::same
  call require(include_self=='auto','include_self string must be auto')
  same=.true.;if(present(mode))same=mode=='connectivity'
  v%sparse=>x;v%n=x%n_rows;v%p=x%n_cols
  graph=graph_from_view(v,0,radius,.true.,mode,metric,p,same)
 end function
 function stable_order(values)result(order)
  real(dp),intent(in)::values(:)
  integer,allocatable::order(:),work(:)
  integer::n,width,first,middle,last,left,right,out,i
  logical::take_left
  n=size(values);order=[(i,i=1,n)];allocate(work(n));width=1
  do while(width<n)
   first=1
   do while(first<=n)
    middle=first+min(width,n-first+1);last=middle+min(width,n-middle+1)-1
    left=first;right=middle
    do out=first,last
     if(left>=middle)then
      take_left=.false.
     else if(right>last)then
      take_left=.true.
     else if(ieee_is_nan(values(order(left))))then
      take_left=ieee_is_nan(values(order(right)))
     else if(ieee_is_nan(values(order(right))))then
      take_left=.true.
     else
      take_left=values(order(left))<=values(order(right))
     end if
     if(take_left)then
      work(out)=order(left);left=left+1
     else
      work(out)=order(right);right=right+1
     end if
    end do
    first=last+1
   end do
   order=work
   if(width>=n-width)exit
   width=2*width
  end do
 end function
 logical function source_sorted(graph)result(sorted)
  type(csr_matrix),intent(in)::graph
  logical,allocatable::inversion(:),boundary(:)
  integer::i,k,n
  n=max(0,size(graph%data)-1);allocate(inversion(n),boundary(n));inversion=.false.;boundary=.false.
  do i=1,n
   if(ieee_is_nan(graph%data(i)).or.ieee_is_nan(graph%data(i+1)))cycle
   inversion(i)=graph%data(i)>graph%data(i+1)
  end do
  do i=2,graph%n_rows
   k=graph%indptr(i)-1
   if(k==0)then
    ! sklearn's global adjacent-value check treats a leading empty-row
    ! boundary as Python index -1; preserve its empty-array rejection too.
    call require(n>0,'Source value-order scan cannot index a leading empty graph boundary')
    k=n
   end if
   if(k<=n)boundary(k)=.true.
  end do
  sorted=count(inversion)==count(inversion.and.boundary)
 end function
 function sort_graph_by_row_values(graph,copy,warn_when_not_sorted,was_sorted)result(ordered)
  type(csr_matrix),intent(inout)::graph
  logical,optional,intent(in)::copy,warn_when_not_sorted
  logical,optional,intent(out)::was_sorted
  type(csr_matrix)::ordered
  integer,allocatable::order(:)
  integer::i,first,last
  logical::same,keep,modes(size(ieee_all)),flags(size(ieee_all))
  call graph%validate();keep=.false.;if(present(copy))keep=copy
  call ieee_get_halting_mode(ieee_all,modes);call ieee_get_flag(ieee_all,flags)
  call ieee_set_halting_mode(ieee_all,.false.)
  same=source_sorted(graph);if(present(was_sorted))was_sorted=same
  ordered=graph
  if(.not.same)then
   do i=1,graph%n_rows
    first=graph%indptr(i);last=graph%indptr(i+1)-1
    order=stable_order(graph%data(first:last))+first-1
    ordered%data(first:last)=graph%data(order);ordered%indices(first:last)=graph%indices(order)
   end do
   if(.not.keep)graph=ordered
  end if
  call ieee_set_flag(ieee_all,flags);call ieee_set_halting_mode(ieee_all,modes)
 end function
 function graph_from_view(v,k,radius,by_radius,mode,metric,p,include_self,force_brute)result(graph)
  type(row_view),intent(in)::v
  integer,intent(in)::k
  real(dp),intent(in)::radius
  logical,intent(in)::by_radius,include_self
  logical,optional,intent(in)::force_brute
  character(*),optional,intent(in)::mode,metric
  real(dp),optional,intent(in)::p
  type(csr_matrix)::graph,stored
  type(pairwise_distance_iterator)::state
  type(distance_chunk)::chunk
  type(neighbor_kd_tree)::tree
  real(dp),allocatable::dist(:,:),u(:),w(:),values(:),new_values(:),selection(:),tree_distance(:),row_norms(:)
  integer,allocatable::columns(:),new_columns(:),order(:),ids(:),tree_ids(:)
  character(16)::kind,output_mode
  real(dp)::power
  logical::sparse,modes(size(ieee_all)),flags(size(ieee_all)),available,forced,use_tree,use_heap,square_keys,power_keys
  integer::n,used,capacity,first,span,count_rows,i,j,row,blockrow,trainrow,take,slot,remove
  integer::fit_k,distance_kind
  integer::query_id,query_chunk,train_chunk,query_offset,train_offset
  integer(int64)::limit
  call check_view(v,.false.);n=v%n;sparse=associated(v%sparse)
  kind='minkowski';if(present(metric))kind=metric
  output_mode='connectivity';if(present(mode))output_mode=mode
  power=2;if(present(p))power=p
  call require(output_mode=='connectivity'.or.output_mode=='distance','invalid functional graph mode')
  call require(any(kind==[character(16)::'minkowski','euclidean','sqeuclidean','l2','manhattan', &
   'cityblock','l1','chebyshev','cosine','precomputed']),'unsupported functional graph metric')
  call require(.not.ieee_is_nan(power),'functional graph p is NaN')
  call require(power>0,'functional graph p must be positive')
  if(kind=='minkowski')then
   if(power==1)kind='manhattan'
   if(power==2)kind='euclidean'
   if(.not.ieee_is_finite(power))kind='chebyshev'
  end if
  if(sparse)call require(any(kind==[character(16)::'euclidean','l2','manhattan','cityblock','l1', &
   'cosine','precomputed']),'unsupported sparse functional graph metric')
  forced=.false.;if(present(force_brute))forced=force_brute
  fit_k=k;if(by_radius)fit_k=5
  use_tree=.not.sparse.and..not.forced.and.v%p<=15.and.fit_k<n/2
  use_tree=use_tree.and.any(kind==[character(16)::'euclidean','l2','manhattan','cityblock','l1','chebyshev','minkowski'])
  if(kind=='minkowski'.and.power<1)use_tree=.false.
  use_heap=.not.by_radius.and.kind/='cosine'.and.kind/='precomputed'
  if(sparse)then
   use_heap=use_heap.and.kind/='euclidean'.and.size(v%sparse%data)>0
  end if
  square_keys=.not.use_tree.and.any(kind==[character(16)::'euclidean','l2','sqeuclidean'])
  power_keys=.not.use_tree.and.kind=='minkowski'
  if(kind=='precomputed')then
   call require(v%n==v%p,'functional precomputed graph must be square')
   if(sparse)then
    call require(all(v%sparse%data>=0),'functional precomputed distances must be nonnegative')
   else
    call require(all(v%dense>=0),'functional precomputed distances must be nonnegative')
   end if
  end if
  if(by_radius)then
   call require(.not.ieee_is_nan(radius),'functional graph radius is NaN')
   call require(radius>=0,'functional graph radius must be nonnegative')
   limit=int(n,int64)*n;capacity=int(min(limit,1024_int64))
  else
   call require(k>0,'functional graph n_neighbors must be positive')
   call require(k<=n,'functional graph neighbors exceed fitted samples')
   if(.not.include_self)call require(k<n,'functional self-excluding graph needs fewer neighbors than samples')
   limit=int(n,int64)*k
   call require(limit<huge(1),'functional graph exceeds integer capacity');capacity=int(limit)
  end if
  graph%n_rows=n;graph%n_cols=n;allocate(graph%indptr(n+1),columns(capacity),values(capacity))
  graph%indptr(1)=1;used=0
  call ieee_get_halting_mode(ieee_all,modes);call ieee_get_flag(ieee_all,flags)
  call ieee_set_halting_mode(ieee_all,.false.)
  if(use_tree)then
   call tree%fit(v%dense,metric=kind,p=power,leaf_size=30)
   do row=1,n
    if(by_radius)then
     call tree%query_radius(v%dense(row,:),radius,tree_distance,tree_ids)
    else
     take=k;if(.not.include_self)take=k+1
     call tree%query(v%dense(row,:),take,tree_distance,tree_ids)
    end if
    call select_row(tree_distance,tree_ids,.true.,preselected=.true.)
   end do
  else if(kind=='precomputed'.and.sparse)then
   stored=v%sparse;stored=sort_graph_by_row_values(stored,copy=.true.,warn_when_not_sorted=.false.)
   do row=1,n
    first=stored%indptr(row)
    call select_row(stored%data(first:stored%indptr(row+1)-1),stored%indices(first:stored%indptr(row+1)-1),.true.)
   end do
  else if(kind=='precomputed')then
   do row=1,n
    call select_row(v%dense(row,:),[(j,j=1,n)],.false.)
   end do
  else if(sparse.and.any(kind==[character(16)::'manhattan','cityblock','l1','l2']))then
   span=max(1,min(n,1048576/n))
   do first=0,n-1,span
    count_rows=min(span,n-first);if(allocated(dist))deallocate(dist)
    allocate(dist(count_rows,n))
    do trainrow=1,n
     do blockrow=1,count_rows
      dist(blockrow,trainrow)=raw_csr_distance(v%sparse,first+blockrow,trainrow,kind=='l2',square_keys)
     end do
    end do
    do i=1,count_rows
     row=first+i;call select_row(dist(i,:),[(j,j=1,n)],.false.)
    end do
   end do
  else if(.not.sparse.and.(kind=='euclidean'.or.kind=='l2'.or.kind=='sqeuclidean'.or.power_keys))then
   allocate(u(v%p),w(v%p));span=max(1,min(n,1048576/n))
   distance_kind=1
   if(kind=='l2')distance_kind=3
   if(power_keys)distance_kind=6
   if(distance_kind==1)then
    allocate(row_norms(n))
    do i=1,n;row_norms(i)=neighbor_row_norm(v%dense(i,:));end do
   end if
   do first=0,n-1,span
    count_rows=min(span,n-first);if(allocated(dist))deallocate(dist)
    allocate(dist(count_rows,n))
    do trainrow=1,n
     call get_row(v,trainrow,w)
     do blockrow=1,count_rows
      call get_row(v,first+blockrow,u)
      if(distance_kind==1)then
       query_id=first+blockrow;query_offset=mod(query_id-1,256)+1;train_offset=mod(trainrow-1,256)+1
       query_chunk=min(256,n-(query_id-query_offset));train_chunk=min(256,n-(trainrow-train_offset))
       dist(blockrow,trainrow)=(row_norms(query_id)+ &
        neighbor_middle_term(u,w,query_chunk,train_chunk,query_offset,train_offset))+row_norms(trainrow)
       if(ieee_is_nan(dist(blockrow,trainrow)))dist(blockrow,trainrow)=0
       dist(blockrow,trainrow)=max(0._dp,dist(blockrow,trainrow))
      else
       dist(blockrow,trainrow)=row_distance(u,w,distance_kind,power,.false.,.false.,.false.)
      end if
     end do
    end do
    do i=1,count_rows
     row=first+i;call select_row(dist(i,:),[(j,j=1,n)],.false.)
    end do
   end do
  else
   if(sparse)then
    if(kind=='minkowski')then
     state=pairwise_distances_chunked(v%sparse,metric=kind,p=power,working_memory=8._dp)
    else if(square_keys.and.kind=='euclidean')then
     state=pairwise_distances_chunked(v%sparse,metric=kind,working_memory=8._dp,squared=.true.)
    else
     state=pairwise_distances_chunked(v%sparse,metric=kind,working_memory=8._dp)
    end if
   else
    if(kind=='minkowski')then
     state=pairwise_distances_chunked(v%dense,metric=kind,p=power,working_memory=8._dp)
    else
     state=pairwise_distances_chunked(v%dense,metric=kind,working_memory=8._dp)
    end if
   end if
   do
    available=state%next(chunk);if(.not.available)exit
    do i=1,size(chunk%distance%values,1)
     row=chunk%distance%start+i
     call select_row(chunk%distance%values(i,:),[(j,j=1,n)],.false.)
    end do
   end do
  end if
  graph%indices=columns(:used);graph%data=values(:used)
  call ieee_set_flag(ieee_all,flags);call ieee_set_halting_mode(ieee_all,modes)
 contains
  subroutine append(column,value)
   integer,intent(in)::column
   real(dp),intent(in)::value
   integer::next
   if(used==capacity)then
    call require(capacity<huge(1)-1,'functional graph exceeds integer capacity')
    next=int(min(int(huge(1)-1,int64),max(16_int64,2_int64*capacity+1)))
    allocate(new_columns(next),new_values(next))
    new_columns(:used)=columns(:used);new_values(:used)=values(:used)
    call move_alloc(new_columns,columns);call move_alloc(new_values,values);capacity=next
   end if
   used=used+1;columns(used)=column;values(used)=value
   if(output_mode=='connectivity')values(used)=1
  end subroutine
  subroutine select_row(d,cols,already_sorted,preselected)
   real(dp),intent(in)::d(:)
   integer,intent(in)::cols(:)
   logical,intent(in)::already_sorted
   logical,optional,intent(in)::preselected
   integer::q
   real(dp)::cutoff,value
   logical::ready
   ready=.false.;if(present(preselected))ready=preselected
   if(by_radius)then
    cutoff=radius
    if(square_keys.and.kind/='sqeuclidean')cutoff=radius*radius
    if(power_keys)cutoff=radius**power
    do q=1,size(d)
     if(.not.include_self.and.cols(q)==row)cycle
     if(ready.or.d(q)<=cutoff)then
      value=d(q)
      if(square_keys.and.kind/='sqeuclidean')value=sqrt(max(value,0._dp))
      if(power_keys)value=max(value,0._dp)**(1._dp/power)
      call append(cols(q),value)
     end if
    end do
   else
    take=k;if(.not.include_self)take=k+1
    call require(size(d)>=take,'not enough stored neighbors for functional graph')
    if(already_sorted)then
     order=[(q,q=1,take)]
     selection=d(order)
    else if(use_heap)then
     selection=d(:take);order=argkmin_order(d,take,n,selection)
    else
     order=pairwise_knn_order(d,take);selection=d(order)
    end if
    if(square_keys.and.kind/='sqeuclidean')selection=sqrt(max(selection,0._dp))
    if(power_keys)selection=max(selection,0._dp)**(1._dp/power)
    remove=0
    if(.not.include_self)then
     do q=1,take
      if(cols(order(q))==row)then
       call require(remove==0,'duplicate selected self entries in functional graph')
       remove=q
      end if
     end do
     if(remove==0)remove=1
    end if
    do q=1,take
     if(q==remove)cycle
     slot=order(q);call append(cols(slot),selection(q))
    end do
   end if
   graph%indptr(row+1)=used+1
  end subroutine
 end function
 real(dp)function raw_csr_distance(a,row1,row2,euclidean,squared)result(value)
  ! sklearn _dist_metrics.pyx.tp Manhattan/Euclidean dist_csr traverses
  ! stored buffers directly, including noncanonical indices and duplicates.
  type(csr_matrix),intent(in)::a
  integer,intent(in)::row1,row2
  logical,intent(in)::euclidean
  logical,intent(in)::squared
  integer::i,j,last1,last2
  real(dp)::difference
  i=a%indptr(row1);j=a%indptr(row2);last1=a%indptr(row1+1);last2=a%indptr(row2+1);value=0
  do while(i<last1.and.j<last2)
   if(a%indices(i)==a%indices(j))then
    difference=a%data(i)-a%data(j);i=i+1;j=j+1
   else if(a%indices(i)<a%indices(j))then
    difference=a%data(i);i=i+1
   else
    difference=a%data(j);j=j+1
   end if
   if(euclidean)then
    value=value+difference*difference
   else
    value=value+abs(difference)
   end if
  end do
  do while(i<last1)
   if(euclidean)then
    value=value+a%data(i)*a%data(i)
   else
    value=value+abs(a%data(i))
   end if
   i=i+1
  end do
  do while(j<last2)
   if(euclidean)then
    value=value+a%data(j)*a%data(j)
   else
    value=value+abs(a%data(j))
   end if
   j=j+1
  end do
  if(euclidean.and..not.squared)value=sqrt(value)
 end function
end module
