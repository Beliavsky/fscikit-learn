! sklearn 1.6.1 neighbors/_base.py query dispatch and exclusion rules (BSD).
! Scalar kernels/selection retain their existing upstream notices.
! Owned Fortran results and dense block adapters are native MIT contributions.
module sklearn_neighbor_query
 use sklearn_common,only:dp,require,check_x
 use sklearn_neighbor_tree,only:neighbor_kd_tree
 use sklearn_neighbor_ball_tree,only:neighbor_ball_tree
 use sklearn_neighbor_selection,only:argkmin_order,pairwise_knn_order,neighbor_simultaneous_sort
 use sklearn_neighbor_gram,only:neighbor_row_norm,neighbor_middle_term
 use sklearn_pairwise_reductions,only:metric_kind,row_distance,extended_distance
 use sklearn_pairwise_chunks,only:pairwise_distances_chunked,pairwise_distance_iterator,distance_chunk
 use sklearn_sparse,only:csr_matrix
 use,intrinsic::ieee_arithmetic
 use,intrinsic::ieee_exceptions
 use,intrinsic::iso_fortran_env,only:int64
 implicit none
 private
 public::neighbor_query_state
 type::neighbor_query_state
  private
  character(16)::metric='minkowski',method='brute'
  real(dp)::power=2
  integer::n=0,p=0,kind=0
  type(neighbor_kd_tree)::kd
  type(neighbor_ball_tree)::ball
 contains
  procedure::fit=>state_fit
  procedure::kneighbors=>state_knn
  procedure::radius_graph=>state_radius
  procedure::fit_method=>state_method
 end type
contains
 function state_method(self)result(method)
  class(neighbor_query_state),intent(in)::self
  character(16)::method
  method=self%method
 end function
 subroutine state_fit(self,training,metric,p,algorithm,leaf_size,n_neighbors)
  class(neighbor_query_state),intent(inout)::self
  real(dp),intent(in)::training(:,:),p
  character(*),intent(in)::metric,algorithm
  integer,intent(in)::leaf_size,n_neighbors
  logical::kd_ok,ball_ok
  character(16)::tree_metric
  type(neighbor_kd_tree)::empty_kd
  type(neighbor_ball_tree)::empty_ball
  call check_x(training)
  call require(n_neighbors>0,'n_neighbors must be positive')
  call require(leaf_size>0,'neighbor leaf_size must be positive')
  call require(.not.ieee_is_nan(p),'neighbor p is NaN')
  call require(p>0,'neighbor p must be positive')
  call require(any(algorithm==[character(16)::'auto','brute','kd_tree','ball_tree']),'unknown neighbor algorithm')
  if(algorithm=='brute')call require(metric/='p'.and.metric/='infinity','metric is not valid for brute neighbors')
  self%n=size(training,1);self%p=size(training,2);self%metric=metric;self%power=p
  if(metric=='minkowski')then
   if(p==1)self%metric='manhattan'
   if(p==2)self%metric='euclidean'
   if(.not.ieee_is_finite(p))self%metric='chebyshev'
  end if
  ! The Source alias "p" does not receive the constructor's separate p value.
  if(metric=='p')self%power=2
  self%kind=0
  if(self%metric/='precomputed')self%kind=metric_kind(self%metric)
  if(metric=='p')self%kind=3
  call require(any(self%kind==[1,2,3,4,5,6,7,9,10,11,13,14,15,16,17,18,19]).or. &
   self%metric=='precomputed','unsupported neighbor query metric')
  if(self%metric=='precomputed')then
   call require(self%n==self%p,'precomputed neighbor fit must be square')
   call require(all(training>=0),'precomputed neighbor distances must be nonnegative')
  end if
  kd_ok=any(self%metric==[character(16)::'euclidean','l2','manhattan','cityblock','l1', &
   'chebyshev','infinity','minkowski','p'])
  ball_ok=kd_ok.or.any(self%metric==[character(16)::'haversine','braycurtis','canberra','hamming', &
   'dice','jaccard','rogerstanimoto','russellrao','sokalmichener','sokalsneath'])
  self%method=algorithm
  if(algorithm=='auto')then
   self%method='brute'
   if(self%metric/='precomputed'.and.self%p<=15.and.n_neighbors<self%n/2)then
    if(kd_ok)self%method='kd_tree'
    if(.not.kd_ok.and.ball_ok)self%method='ball_tree'
   end if
   if(self%metric=='minkowski'.and.self%power<1)self%method='brute'
  end if
  if(self%method=='kd_tree')call require(kd_ok,'metric is not supported by KDTree')
  if(self%method=='ball_tree')call require(ball_ok,'metric is not supported by BallTree')
  if(self%method/='brute'.and.self%metric=='minkowski') &
   call require(self%power>=1,'tree neighbors require Minkowski p at least one')
  tree_metric=self%metric;if(tree_metric=='p')tree_metric='minkowski'
  self%kd=empty_kd;self%ball=empty_ball
  if(self%method=='kd_tree')call self%kd%fit(training,tree_metric,self%power,leaf_size)
  if(self%method=='ball_tree')call self%ball%fit(training,tree_metric,self%power,leaf_size)
 end subroutine
 subroutine state_knn(self,training,x,distances,indices,n_neighbors,exclude_self)
  class(neighbor_query_state),intent(in)::self
  real(dp),target,intent(in)::training(:,:)
  real(dp),target,optional,intent(in)::x(:,:)
  real(dp),allocatable,intent(out)::distances(:,:)
  integer,allocatable,intent(out)::indices(:,:)
  integer,intent(in)::n_neighbors
  logical,optional,intent(in)::exclude_self
  type(csr_matrix)::unused
  call query_rows(self,training,x,n_neighbors,0._dp,.false.,exclude_self,distances,indices,unused)
 end subroutine
 function state_radius(self,training,x,radius,exclude_self,sort_results)result(graph)
  class(neighbor_query_state),intent(in)::self
  real(dp),target,intent(in)::training(:,:)
  real(dp),target,optional,intent(in)::x(:,:)
  real(dp),intent(in)::radius
  logical,optional,intent(in)::exclude_self
  logical,optional,intent(in)::sort_results
  type(csr_matrix)::graph
  real(dp),allocatable::unused_d(:,:)
  integer,allocatable::unused_i(:,:)
  call query_rows(self,training,x,0,radius,.true.,exclude_self,unused_d,unused_i,graph,sort_results)
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
 logical function fallback(self)result(value)
  class(neighbor_query_state),intent(in)::self
  value=any(self%kind==[7,13,14,15,16,17,18,19])
 end function
 real(dp)function actual(self,key)result(value)
  class(neighbor_query_state),intent(in)::self
  real(dp),intent(in)::key
  value=key
  if(any(self%kind==[1,3]))value=sqrt(key)
  if(self%kind==6)value=key**(1._dp/self%power)
  if(self%kind==9)value=2*asin(sqrt(key))
 end function
 subroutine query_rows(self,training,x,k,radius,by_radius,exclude_self,distances,indices,graph,sort_results)
  class(neighbor_query_state),intent(in)::self
  real(dp),target,intent(in)::training(:,:)
  real(dp),target,optional,intent(in)::x(:,:)
  integer,intent(in)::k
  real(dp),intent(in)::radius
  logical,intent(in)::by_radius
  logical,optional,intent(in)::exclude_self
  real(dp),allocatable,intent(out)::distances(:,:)
  integer,allocatable,intent(out)::indices(:,:)
  type(csr_matrix),intent(out)::graph
  logical,optional,intent(in)::sort_results
  type(pairwise_distance_iterator)::state
  type(distance_chunk)::chunk
  real(dp),allocatable::block(:,:),u(:),v(:),tnorm(:),qnorm(:),values(:),grown_values(:),drow(:),selected(:)
  integer,allocatable::columns(:),grown_columns(:),ids(:),order(:)
  real(dp)::cutoff,key,middle
  integer::m,take,row,i,j,first,span,nrows,offset_q,offset_t,chunk_q,chunk_t,used,capacity,q,remove,out,next
  logical::skip,available,heap,sorted,modes(size(ieee_all)),flags(size(ieee_all))
  call require(self%n>0,'neighbor query state is not fitted')
  call require(size(training,1)==self%n.and.size(training,2)==self%p,'fitted neighbor storage shape changed')
  sorted=.false.;if(present(sort_results))sorted=sort_results
  m=self%n;skip=.not.present(x)
  if(present(x))then
   call check_x(x);call require(size(x,2)==self%p,'neighbor query feature count mismatch');m=size(x,1)
   if(self%metric=='precomputed')call require(all(x>=0),'precomputed neighbor distances must be nonnegative')
  end if
  if(present(exclude_self))skip=exclude_self
  if(skip)call require(.not.present(x),'self exclusion requires an omitted query array')
  if(self%kind==9)call require(self%p==2,'haversine requires two features')
  if(by_radius.and.self%method=='brute'.and..not.fallback(self).and.self%metric/='precomputed')then
   if(.not.ieee_is_nan(radius))call require(radius>=0,'brute neighbor radius must be nonnegative')
  end if
  take=k
  if(.not.by_radius)then
   call require(k>0.and.k<=self%n,'n_neighbors exceeds training size')
   if(skip)call require(k<self%n,'self query requires fewer neighbors than training samples')
   if(skip)take=k+1
   allocate(distances(m,k),indices(m,k))
  end if
  graph%n_rows=m;graph%n_cols=self%n;used=0
  if(by_radius)then
   capacity=int(min(1024_int64,int(m,int64)*self%n))
   allocate(columns(capacity),values(capacity),graph%indptr(m+1));graph%indptr(1)=1
  end if
  call ieee_get_halting_mode(ieee_all,modes);call ieee_get_flag(ieee_all,flags)
  call ieee_set_halting_mode(ieee_all,.false.)
  cutoff=radius
  if(any(self%kind==[1,3]))cutoff=radius*radius
  if(self%kind==6)cutoff=radius**self%power
  if(self%kind==9)cutoff=sin(.5_dp*radius)**2
  heap=.not.fallback(self).and.self%metric/='precomputed'
  allocate(u(self%p),v(self%p))
  if(self%method/='brute')then
   do row=1,m
    call point(row,u)
    if(self%method=='kd_tree')then
     if(by_radius)then
      call self%kd%query_radius(u,radius,drow,ids)
     else
      call self%kd%query(u,take,drow,ids)
     end if
    else
     if(by_radius)then
      call self%ball%query_radius(u,radius,drow,ids)
     else
      call self%ball%query(u,take,drow,ids)
     end if
    end if
    call consume(drow,ids,.true.)
   end do
  else if(fallback(self))then
   if(present(x))then
    state=pairwise_distances_chunked(x,training,metric=self%metric,working_memory=8._dp)
   else
    state=pairwise_distances_chunked(training,metric=self%metric,working_memory=8._dp)
   end if
   if(self%kind==7)call state%set_reference_memory(1024._dp)
   do
    available=state%next(chunk);if(.not.available)exit
    do i=1,size(chunk%distance%values,1)
     row=chunk%distance%start+i;call consume(chunk%distance%values(i,:),[(j,j=1,self%n)],.false.)
    end do
   end do
  else
   allocate(tnorm(self%n),qnorm(m));tnorm=0;qnorm=0
   if(any(self%kind==[1,2]))then
    do j=1,self%n;tnorm(j)=neighbor_row_norm(training(j,:));end do
    do i=1,m;call point(i,u);qnorm(i)=neighbor_row_norm(u);end do
   end if
   span=max(1,min(m,1048576/self%n))
   do first=0,m-1,span
    nrows=min(span,m-first);if(allocated(block))deallocate(block)
    allocate(block(nrows,self%n))
    do j=1,self%n
     v=training(j,:)
     do i=1,nrows
      row=first+i;call point(row,u)
      if(self%metric=='precomputed')then
       block(i,j)=u(j)
      else if(any(self%kind==[1,2]))then
       offset_q=mod(row-1,256)+1;offset_t=mod(j-1,256)+1
       chunk_q=min(256,m-(row-offset_q));chunk_t=min(256,self%n-(j-offset_t))
       middle=neighbor_middle_term(u,v,chunk_q,chunk_t,offset_q,offset_t)
       key=(qnorm(row)+middle)+tnorm(j)
       if(ieee_is_nan(key))key=0
       block(i,j)=max(0._dp,key)
      else if(self%kind<=9)then
       block(i,j)=row_distance(u,v,self%kind,self%power,.false.,.false.,.false.)
      else
       block(i,j)=extended_distance(u,v,self%kind,.false.,.false.,u/=0,v/=0)
      end if
     end do
    end do
    do i=1,nrows
     row=first+i;call consume(block(i,:),[(j,j=1,self%n)],.false.)
    end do
   end do
  end if
  if(by_radius)then
   graph%indices=columns(:used);graph%data=values(:used)
  end if
  call ieee_set_flag(ieee_all,flags);call ieee_set_halting_mode(ieee_all,modes)
 contains
  subroutine point(at,vector)
   integer,intent(in)::at
   real(dp),intent(out)::vector(:)
   if(present(x))then
    vector=x(at,:)
   else
    vector=training(at,:)
   end if
  end subroutine
  subroutine append(id,value)
   integer,intent(in)::id
   real(dp),intent(in)::value
   if(used==capacity)then
    call require(capacity<huge(1)-1,'neighbor graph exceeds integer capacity')
    next=int(min(int(huge(1)-1,int64),max(16_int64,2_int64*capacity+1)))
    allocate(grown_columns(next),grown_values(next))
    grown_columns(:used)=columns(:used);grown_values(:used)=values(:used)
    call move_alloc(grown_columns,columns);call move_alloc(grown_values,values);capacity=next
   end if
   used=used+1;columns(used)=id;values(used)=value
  end subroutine
  subroutine consume(d,cols,preselected)
   real(dp),intent(in)::d(:)
   integer,intent(in)::cols(:)
   logical,intent(in)::preselected
   real(dp)::value
   real(dp),allocatable::radius_values(:)
   integer,allocatable::radius_ids(:),radius_order(:)
   if(by_radius)then
    if(preselected)then
     radius_values=d;radius_ids=cols
    else
     radius_values=pack(d,d<=cutoff);radius_ids=pack(cols,d<=cutoff)
    end if
    if(sorted)then
     if(preselected.or.heap)then
      call neighbor_simultaneous_sort(radius_values,radius_ids)
     else
      radius_order=stable_order(radius_values)
      radius_values=radius_values(radius_order);radius_ids=radius_ids(radius_order)
     end if
    end if
    do q=1,size(radius_values)
     if(skip.and.radius_ids(q)==row)cycle
     value=radius_values(q);if(.not.preselected)value=actual(self,value)
     call append(radius_ids(q),value)
    end do
    graph%indptr(row+1)=used+1
   else
    if(preselected)then
     order=[(q,q=1,take)];selected=d(:take)
    else if(heap)then
     selected=d(:take);order=argkmin_order(d,take,m,selected)
    else
     order=pairwise_knn_order(d,take);selected=d(order)
    end if
    if(.not.preselected)then
     do q=1,take;selected(q)=actual(self,selected(q));end do
    end if
    remove=0
    if(skip)then
     do q=1,take
      if(cols(order(q))/=row)cycle
      call require(remove==0,'duplicate selected self entries');remove=q
     end do
     if(remove==0)remove=1
    end if
    out=0
    do q=1,take
     if(q==remove)cycle
     out=out+1;distances(row,out)=selected(q);indices(row,out)=cols(order(q))
    end do
   end if
  end subroutine
 end subroutine
end module
