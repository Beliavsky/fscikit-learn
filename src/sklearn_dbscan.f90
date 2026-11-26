! sklearn 1.6.1 cluster/_dbscan.py and Lars Buitinck's _dbscan_inner.pyx (BSD).
! Owned typed storage adapters are native MIT contributions.
module sklearn_dbscan
 use sklearn_common,only:dp,require,check_x,argsort
 use sklearn_sparse,only:csr_matrix
 use sklearn_neighbors,only:nearest_neighbors
 use sklearn_neighbor_functions,only:radius_neighbors_graph
 use sklearn_pairwise_chunks,only:pairwise_distances_chunked,pairwise_distance_iterator,distance_chunk
 use sklearn_ieee_math,only:ieee_difference
 use,intrinsic::ieee_arithmetic,only:ieee_is_finite,ieee_is_nan
 implicit none
 private
 public::dbscan
 type::dbscan
  real(dp)::eps=.5_dp,p=2
  integer::min_samples=5,leaf_size=30,n_features_in_=0
  character(16)::metric='euclidean',algorithm='auto',fit_method_='brute'
  integer,allocatable::labels_(:),core_sample_indices_(:)
  real(dp),allocatable::components_(:,:)
  type(csr_matrix),allocatable::components_csr_
 contains
  procedure,private::fit_dense,fit_sparse,predict_dense,predict_sparse
  generic::fit=>fit_dense,fit_sparse
  generic::fit_predict=>predict_dense,predict_sparse
 end type
contains
 subroutine controls(self)
  class(dbscan),intent(in)::self
  call require(.not.ieee_is_nan(self%eps),'DBSCAN eps is NaN')
  call require(self%eps>0,'DBSCAN eps must be positive')
  call require(self%min_samples>0,'DBSCAN min_samples must be positive')
  call require(self%leaf_size>0,'DBSCAN leaf_size must be positive')
  call require(any(self%algorithm==[character(16)::'auto','brute','kd_tree','ball_tree']),'invalid DBSCAN algorithm')
  call require(.not.ieee_is_nan(self%p),'DBSCAN p is NaN')
  call require(self%p>=0,'DBSCAN p must be nonnegative')
 end subroutine
 subroutine fit_dense(self,x,sample_weight)
  class(dbscan),intent(inout)::self
  real(dp),target,intent(in)::x(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  type(nearest_neighbors)::search
  type(csr_matrix)::graph
  call controls(self);call check_x(x)
  search%radius=self%eps;search%algorithm=self%algorithm;search%metric=self%metric
  search%p=self%p;search%leaf_size=self%leaf_size
  call search%fit(x)
  graph=search%radius_neighbors_graph(search%x_,mode='connectivity')
  call cluster(self,graph,sample_weight)
  self%fit_method_=search%fit_method_;self%n_features_in_=size(x,2)
  self%components_=x(self%core_sample_indices_,:)
  if(allocated(self%components_csr_))deallocate(self%components_csr_)
 end subroutine
 subroutine fit_sparse(self,x,sample_weight)
  class(dbscan),intent(inout)::self
  type(csr_matrix),intent(in)::x
  real(dp),optional,intent(in)::sample_weight(:)
  type(csr_matrix)::graph,work
  call controls(self);call x%validate()
  call require(x%n_rows>0.and.x%n_cols>0,'DBSCAN requires nonempty input')
  call require(all(ieee_is_finite(x%data)),'DBSCAN input contains NaN or infinity')
  if(self%algorithm=='kd_tree'.or.self%algorithm=='ball_tree') &
   call require(self%metric/='cosine'.and.self%metric/='precomputed','metric is not supported by the requested tree')
  work=x
  if(self%metric=='precomputed')work=explicit_diagonal(x)
  if(self%metric=='cosine')then
   graph=sparse_cosine_graph(work,self%eps)
  else
   graph=radius_neighbors_graph(work,self%eps,metric=self%metric,p=self%p,include_self=.true.)
  end if
  call cluster(self,graph,sample_weight)
  self%fit_method_='brute';self%n_features_in_=x%n_cols
  if(size(self%core_sample_indices_)>0)then
   self%components_csr_=select_rows(work,self%core_sample_indices_)
   if(allocated(self%components_))deallocate(self%components_)
  else
   if(allocated(self%components_csr_))deallocate(self%components_csr_)
   if(allocated(self%components_))deallocate(self%components_)
   allocate(self%components_(0,x%n_cols))
  end if
 end subroutine
 function sparse_cosine_graph(x,radius)result(value)
  type(csr_matrix),intent(in)::x
  real(dp),intent(in)::radius
  type(csr_matrix)::value
  type(csr_matrix)::training
  type(pairwise_distance_iterator)::iterator
  type(distance_chunk)::chunk
  integer,allocatable::indices(:),grown(:)
  integer::i,j,k,row,capacity,next
  ! NearestNeighbors copies CSR training data. Explicit queries therefore use
  ! two distinct matrices, unlike dense DBSCAN's aliased X/X cosine products.
  training=x;iterator=pairwise_distances_chunked(x,training,metric='cosine')
  value%n_rows=x%n_rows;value%n_cols=x%n_rows
  allocate(value%indptr(x%n_rows+1));value%indptr(1)=1
  capacity=max(16,x%n_rows);allocate(indices(capacity));k=0
  do while(iterator%next(chunk))
   do i=1,size(chunk%distance%values,1)
    row=chunk%distance%start+i
    do j=1,x%n_rows
     if(ieee_is_nan(chunk%distance%values(i,j)))cycle
     if(chunk%distance%values(i,j)>radius)cycle
     if(k==capacity)then
      call require(capacity<huge(1)/2,'DBSCAN graph exceeds integer capacity')
      next=2*capacity;allocate(grown(next));grown(:k)=indices(:k)
      call move_alloc(grown,indices);capacity=next
     end if
     k=k+1;indices(k)=j
    end do
    value%indptr(row+1)=k+1
   end do
  end do
  value%indices=indices(:k);allocate(value%data(k));value%data=1
 end function
 function predict_dense(self,x,sample_weight)result(labels)
  class(dbscan),intent(inout)::self
  real(dp),intent(in)::x(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  integer::labels(size(x,1))
  call self%fit(x,sample_weight);labels=self%labels_
 end function
 function predict_sparse(self,x,sample_weight)result(labels)
  class(dbscan),intent(inout)::self
  type(csr_matrix),intent(in)::x
  real(dp),optional,intent(in)::sample_weight(:)
  integer::labels(x%n_rows)
  call self%fit(x,sample_weight);labels=self%labels_
 end function
 recursive real(dp)function numpy_sum(a)result(total)
  real(dp),intent(in)::a(:)
  real(dp)::part(8),left,right
  integer::n,i,j,whole,split
  n=size(a);total=-0._dp
  if(n<8)then
   do i=1,n;total=ieee_difference(total,-a(i));end do
  else if(n<=128)then
   part=a(:8);whole=n-mod(n,8)
   do i=9,whole,8
    do j=1,8;part(j)=ieee_difference(part(j),-a(i+j-1));end do
   end do
   left=ieee_difference(ieee_difference(part(1),-part(2)),-ieee_difference(part(3),-part(4)))
   right=ieee_difference(ieee_difference(part(5),-part(6)),-ieee_difference(part(7),-part(8)))
   total=ieee_difference(left,-right)
   do i=whole+1,n;total=ieee_difference(total,-a(i));end do
  else
   split=n/2;split=split-mod(split,8)
   total=ieee_difference(numpy_sum(a(:split)),-numpy_sum(a(split+1:)))
  end if
 end function
 subroutine cluster(self,graph,sample_weight)
  class(dbscan),intent(inout)::self
  type(csr_matrix),intent(in)::graph
  real(dp),optional,intent(in)::sample_weight(:)
  logical::core(graph%n_rows)
  integer,allocatable::stack(:)
  integer::i,j,k,item,top,label,n,first,last
  n=graph%n_rows
  if(present(sample_weight))then
   call require(size(sample_weight)==n,'DBSCAN weight count mismatch')
   call require(all(ieee_is_finite(sample_weight)),'DBSCAN weights must be finite')
  end if
  do i=1,n
   first=graph%indptr(i);last=graph%indptr(i+1)-1
   if(present(sample_weight))then
    core(i)=numpy_sum(sample_weight(graph%indices(first:last)))>=self%min_samples
   else
    core(i)=last-first+1>=self%min_samples
   end if
  end do
  self%labels_=spread(-1,1,n);label=0
  call require(size(graph%indices)<huge(1),'DBSCAN stack exceeds integer capacity')
  allocate(stack(size(graph%indices)+1));top=0
  do i=1,n
   if(self%labels_(i)/=-1.or..not.core(i))cycle
   item=i
   do
    if(self%labels_(item)==-1)then
     self%labels_(item)=label
     if(core(item))then
      do j=graph%indptr(item),graph%indptr(item+1)-1
       k=graph%indices(j)
       if(self%labels_(k)==-1)then
        top=top+1;stack(top)=k
       end if
      end do
     end if
    end if
    if(top==0)exit
    item=stack(top);top=top-1
   end do
   label=label+1
  end do
  self%core_sample_indices_=pack([(i,i=1,n)],core)
 end subroutine
 function explicit_diagonal(x)result(value)
  type(csr_matrix),intent(in)::x
  type(csr_matrix)::value
  type(csr_matrix)::work
  real(dp)::diagonal(x%n_rows)
  integer::i,j,k,missing,diagonal_counts(x%n_rows)
  logical::fast_insert,sorted,added
  call require(x%n_rows==x%n_cols,'DBSCAN precomputed distances must be square')
  diagonal=0;diagonal_counts=0
  do i=1,x%n_rows
   do j=x%indptr(i),x%indptr(i+1)-1
    if(x%indices(j)/=i)cycle
    diagonal_counts(i)=diagonal_counts(i)+1
    diagonal(i)=ieee_difference(diagonal(i),-x%data(j))
   end do
  end do
  ! SciPy setdiag canonicalizes the whole matrix if a sampled diagonal has
  ! duplicates. Otherwise stored off-diagonal duplicates remain independent.
  work=x
  if(any(diagonal_counts>1))work=canonical_rows(x)
  missing=count(diagonal_counts==0)
  ! SciPy uses _insert_many for an almost diagonal matrix. Unlike the COO
  ! path, this preserves existing diagonal positions and prepends new entries
  ! before optionally sorting columns. Row order matters for signed sums.
  fast_insert=.false.;sorted=.true.
  if(missing>0)fast_insert=real(size(work%data)-(x%n_rows-missing),dp)<real(size(work%data),dp)*.001_dp
  if(fast_insert)then
   do i=1,x%n_rows
    do j=work%indptr(i)+1,work%indptr(i+1)-1
     if(work%indices(j)<work%indices(j-1))sorted=.false.
    end do
   end do
  end if
  call require(size(work%data)<=huge(1)-missing-1,'DBSCAN diagonal exceeds integer capacity')
  value%n_rows=x%n_rows;value%n_cols=x%n_cols
  allocate(value%indptr(x%n_rows+1),value%indices(size(work%data)+missing),value%data(size(work%data)+missing))
  k=1;value%indptr(1)=1
  do i=1,x%n_rows
   added=diagonal_counts(i)>0
   if(fast_insert.and..not.sorted.and..not.added)then
    value%indices(k)=i;value%data(k)=diagonal(i);k=k+1;added=.true.
   end if
   do j=work%indptr(i),work%indptr(i+1)-1
    if(missing>0.and..not.fast_insert.and.work%indices(j)==i)cycle
    if(fast_insert.and..not.added.and.work%indices(j)>i)then
     value%indices(k)=i;value%data(k)=diagonal(i);k=k+1;added=.true.
    end if
    value%indices(k)=work%indices(j);value%data(k)=work%data(j)
    if(work%indices(j)==i)value%data(k)=diagonal(i)
    k=k+1
   end do
   if((missing>0.and..not.fast_insert).or.(fast_insert.and..not.added))then
    value%indices(k)=i;value%data(k)=diagonal(i);k=k+1
   end if
   value%indptr(i+1)=k
  end do
 end function
 function canonical_rows(x)result(value)
  type(csr_matrix),intent(in)::x
  type(csr_matrix)::value
  integer,allocatable::order(:),columns(:)
  real(dp),allocatable::data(:)
  integer::i,j,k,first,last,at,column
  allocate(columns(size(x%data)),data(size(x%data)),value%indptr(x%n_rows+1))
  k=0;value%indptr(1)=1
  do i=1,x%n_rows
   first=x%indptr(i);last=x%indptr(i+1)-1
   order=argsort(real(x%indices(first:last),dp));column=0
   do j=1,size(order)
    at=first+order(j)-1
    if(x%indices(at)==column)then
     data(k)=ieee_difference(data(k),-x%data(at))
    else
     k=k+1;column=x%indices(at);columns(k)=column;data(k)=x%data(at)
    end if
   end do
   value%indptr(i+1)=k+1
  end do
  value%n_rows=x%n_rows;value%n_cols=x%n_cols
  value%indices=columns(:k);value%data=data(:k)
 end function
 function select_rows(x,rows)result(value)
  type(csr_matrix),intent(in)::x
  integer,intent(in)::rows(:)
  type(csr_matrix)::value
  integer::i,j,k,n
  n=0
  do i=1,size(rows);j=rows(i);n=n+x%indptr(j+1)-x%indptr(j);end do
  value%n_rows=size(rows);value%n_cols=x%n_cols
  allocate(value%indptr(size(rows)+1),value%indices(n),value%data(n))
  k=1;value%indptr(1)=1
  do i=1,size(rows)
   do j=x%indptr(rows(i)),x%indptr(rows(i)+1)-1
    value%indices(k)=x%indices(j);value%data(k)=x%data(j);k=k+1
   end do
   value%indptr(i+1)=k
  end do
 end function
end module
