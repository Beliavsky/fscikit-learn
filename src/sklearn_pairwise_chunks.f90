! Streaming distance composition: sklearn 1.6.1 metrics/pairwise.py (BSD).
! SciPy distance kernels retain existing SciPy BSD notices; native ownership is MIT.
module sklearn_pairwise_chunks
 use sklearn_common,only:dp,require
 use sklearn_sparse,only:csr_matrix
 use sklearn_pairwise_reductions,only:row_view,check_view,get_row,get_haversine_row,metric_kind, &
  row_distance,extended_distance,parameter_distance,covariance_precision
 use sklearn_kernel_dispatch,only:pairwise_kernels,kernel_blas_dot
 use,intrinsic::ieee_arithmetic
 use,intrinsic::ieee_exceptions
 implicit none
 private
 public::pairwise_distances_chunked,pairwise_distance_iterator,distance_block,distance_chunk, &
  distance_reduction,pairwise_chunk_reducer
 abstract interface
  function row_metric(x,y)result(value)
   import dp
   real(dp),intent(in)::x(:),y(:)
   real(dp)::value
  end function
 end interface
 type::distance_block
  integer::start=0,n_rows=0
  real(dp),allocatable::values(:,:)
  type(csr_matrix)::sparse
  logical::is_sparse=.false.
 end type
 type::distance_reduction
  real(dp),allocatable::real_vector(:),real_matrix(:,:)
  integer,allocatable::integer_vector(:),integer_matrix(:,:)
  type(csr_matrix)::sparse
  type(distance_reduction),allocatable::tuple(:)
 end type
 type::distance_chunk
  type(distance_block)::distance
  type(distance_reduction)::reduced
  logical::is_reduced=.false.
 end type
 type,abstract::pairwise_chunk_reducer
 contains
  procedure(reduce_block),deferred::apply
 end type
 abstract interface
  subroutine reduce_block(self,block,reduced)
   import pairwise_chunk_reducer,distance_block,distance_reduction
   class(pairwise_chunk_reducer),intent(inout)::self
   type(distance_block),intent(inout)::block
   type(distance_reduction),intent(out)::reduced
  end subroutine
 end interface
 type::pairwise_distance_iterator
  private
  real(dp),allocatable::xd(:,:),yd(:,:),weights(:),variance(:),precision(:,:)
  type(csr_matrix)::xs,ys
  logical::sparse_x=.false.,sparse_y=.false.,self_pair=.false.,square=.false.,exceeded=.false.
  integer::kind=1,n=0,m=0,p=0,position=1,rows=1
  real(dp)::power=2,missing=0
  procedure(row_metric),pointer,nopass::callback=>null()
 contains
  procedure::next=>next_chunk
  procedure::chunk_rows
  procedure::exceeds_working_memory
 end type
 interface pairwise_distances_chunked
  module procedure chunks_dd,chunks_dd_callback
  module procedure chunks_ss,chunks_ss_callback
  module procedure chunks_ds,chunks_ds_callback
  module procedure chunks_sd,chunks_sd_callback
 end interface
contains
 function chunks_dd(x,y,metric,working_memory,p,squared,w,v,vi,missing_values)result(state)
  real(dp),target,intent(in)::x(:,:)
  real(dp),target,optional,intent(in)::y(:,:)
  real(dp),optional,intent(in)::working_memory
  type(pairwise_distance_iterator)::state
  character(*),optional,intent(in)::metric
  real(dp),optional,intent(in)::p,w(:),v(:),vi(:,:),missing_values
  logical,optional,intent(in)::squared
  state%xd=x
  state%sparse_x=.false.
  state%yd=x;state%self_pair=.true.
  if(present(y))then
  state%yd=y
  state%sparse_y=.false.
   state%self_pair=associated_pointer_d(x,y)
  end if
  call prepare(state,metric,working_memory,p,squared,w,v,vi,missing_values)
 end function
 function chunks_dd_callback(x,y,metric,working_memory)result(state)
  real(dp),target,intent(in)::x(:,:)
  real(dp),target,optional,intent(in)::y(:,:)
  real(dp),optional,intent(in)::working_memory
  type(pairwise_distance_iterator)::state
  procedure(row_metric)::metric
  state%xd=x
  state%sparse_x=.false.
  state%yd=x;state%self_pair=.true.
  if(present(y))then
  state%yd=y
  state%sparse_y=.false.
   state%self_pair=associated_pointer_d(x,y)
  end if
  state%callback=>metric
  call prepare(state,working_memory=working_memory)
 end function
 function chunks_ss(x,y,metric,working_memory,p,squared,w,v,vi,missing_values)result(state)
  type(csr_matrix),target,intent(in)::x
  type(csr_matrix),target,optional,intent(in)::y
  real(dp),optional,intent(in)::working_memory
  type(pairwise_distance_iterator)::state
  character(*),optional,intent(in)::metric
  real(dp),optional,intent(in)::p,w(:),v(:),vi(:,:),missing_values
  logical,optional,intent(in)::squared
  state%xs=x
  state%sparse_x=.true.
  state%ys=x;state%self_pair=.true.
  if(present(y))then
  state%ys=y
  state%sparse_y=.true.
   state%self_pair=associated_pointer_s(x,y)
  end if
  state%sparse_y=.true.
  call prepare(state,metric,working_memory,p,squared,w,v,vi,missing_values)
 end function
 function chunks_ss_callback(x,y,metric,working_memory)result(state)
  type(csr_matrix),target,intent(in)::x
  type(csr_matrix),target,optional,intent(in)::y
  real(dp),optional,intent(in)::working_memory
  type(pairwise_distance_iterator)::state
  procedure(row_metric)::metric
  state%xs=x
  state%sparse_x=.true.
  state%ys=x;state%self_pair=.true.
  if(present(y))then
  state%ys=y
  state%sparse_y=.true.
   state%self_pair=associated_pointer_s(x,y)
  end if
  state%sparse_y=.true.
  state%callback=>metric
  call prepare(state,working_memory=working_memory)
 end function
 function chunks_ds(x,y,metric,working_memory,p,squared,w,v,vi,missing_values)result(state)
  real(dp),target,intent(in)::x(:,:)
  type(csr_matrix),target,intent(in)::y
  real(dp),optional,intent(in)::working_memory
  type(pairwise_distance_iterator)::state
  character(*),optional,intent(in)::metric
  real(dp),optional,intent(in)::p,w(:),v(:),vi(:,:),missing_values
  logical,optional,intent(in)::squared
  state%xd=x
  state%sparse_x=.false.
  state%ys=y
  state%sparse_y=.true.
  call prepare(state,metric,working_memory,p,squared,w,v,vi,missing_values)
 end function
 function chunks_ds_callback(x,y,metric,working_memory)result(state)
  real(dp),target,intent(in)::x(:,:)
  type(csr_matrix),target,intent(in)::y
  real(dp),optional,intent(in)::working_memory
  type(pairwise_distance_iterator)::state
  procedure(row_metric)::metric
  state%xd=x
  state%sparse_x=.false.
  state%ys=y
  state%sparse_y=.true.
  state%callback=>metric
  call prepare(state,working_memory=working_memory)
 end function
 function chunks_sd(x,y,metric,working_memory,p,squared,w,v,vi,missing_values)result(state)
  type(csr_matrix),target,intent(in)::x
  real(dp),target,intent(in)::y(:,:)
  real(dp),optional,intent(in)::working_memory
  type(pairwise_distance_iterator)::state
  character(*),optional,intent(in)::metric
  real(dp),optional,intent(in)::p,w(:),v(:),vi(:,:),missing_values
  logical,optional,intent(in)::squared
  state%xs=x
  state%sparse_x=.true.
  state%yd=y
  state%sparse_y=.false.
  call prepare(state,metric,working_memory,p,squared,w,v,vi,missing_values)
 end function
 function chunks_sd_callback(x,y,metric,working_memory)result(state)
  type(csr_matrix),target,intent(in)::x
  real(dp),target,intent(in)::y(:,:)
  real(dp),optional,intent(in)::working_memory
  type(pairwise_distance_iterator)::state
  procedure(row_metric)::metric
  state%xs=x
  state%sparse_x=.true.
  state%yd=y
  state%sparse_y=.false.
  state%callback=>metric
  call prepare(state,working_memory=working_memory)
 end function
 logical function associated_pointer_d(x,y)result(same)
  real(dp),target,intent(in)::x(:,:),y(:,:)
  real(dp),pointer::a(:,:),b(:,:)
  a=>x;b=>y;same=associated(a,b)
 end function
 logical function associated_pointer_s(x,y)result(same)
  type(csr_matrix),target,intent(in)::x,y
  type(csr_matrix),pointer::a,b
  a=>x;b=>y;same=associated(a,b)
 end function
 integer function chunk_rows(self)result(rows)
  class(pairwise_distance_iterator),intent(in)::self
  rows=self%rows
 end function
 logical function exceeds_working_memory(self)result(exceeded)
  class(pairwise_distance_iterator),intent(in)::self
  exceeded=self%exceeded
 end function
 subroutine views(self,a,b)
  class(pairwise_distance_iterator),target,intent(in)::self
  type(row_view),intent(out)::a,b
  if(self%sparse_x)then
   a%sparse=>self%xs;a%n=self%xs%n_rows;a%p=self%xs%n_cols
  else
   a%dense=>self%xd;a%n=size(self%xd,1);a%p=size(self%xd,2)
  end if
  if(self%sparse_y)then
   b%sparse=>self%ys;b%n=self%ys%n_rows;b%p=self%ys%n_cols
  else
   b%dense=>self%yd;b%n=size(self%yd,1);b%p=size(self%yd,2)
  end if
 end subroutine
 subroutine prepare(self,metric,working_memory,p,squared,w,v,vi,missing_values)
  type(pairwise_distance_iterator),target,intent(inout)::self
  character(*),optional,intent(in)::metric
  real(dp),optional,intent(in)::working_memory,p,w(:),v(:),vi(:,:),missing_values
  logical,optional,intent(in)::squared
  type(row_view)::a,b
  real(dp)::memory,capacity,mean_,total
   integer::j,i
  logical::modes(size(ieee_all)),flags(size(ieee_all))
  call ieee_get_halting_mode(ieee_all,modes);call ieee_get_flag(ieee_all,flags)
  call ieee_set_halting_mode(ieee_all,.false.)
  self%kind=0
  if(.not.associated(self%callback))then
   self%kind=1
   if(present(metric))then
    if(metric=='precomputed')then
     self%kind=-1
    else
     call require(metric/='p'.and.metric/='infinity'.and.metric/='matching','unknown chunked distance metric')
     self%kind=metric_kind(metric)
    end if
   end if
  end if
  call views(self,a,b);call check_view(a,self%kind==8);call check_view(b,self%kind==8)
  self%n=a%n;self%m=b%n;self%p=a%p
  if(self%kind==-1)then
   call require(a%p==b%n,'precomputed distance columns must equal reference sample count')
   if(self%sparse_x)then
    call require(all(self%xs%data>=0),'precomputed distances must be nonnegative')
   else
    call require(all(self%xd>=0),'precomputed distances must be nonnegative')
   end if
  else
   call require(a%p==b%p,'chunked distance feature counts differ')
  end if
  if(self%sparse_x.or.self%sparse_y)then
   call require(any(self%kind==[-1,0,1,3,4,7,9]),'this chunked distance metric requires dense inputs')
  end if
  if(self%kind==9)call require(self%p==2,'haversine requires two features')
  memory=1024;if(present(working_memory))memory=working_memory
  call require(.not.ieee_is_nan(memory),'working_memory must not be NaN')
  call require(ieee_is_finite(memory),'working_memory must be finite')
  call require(memory>=0,'working_memory must be nonnegative')
  if(self%kind==-1)then
   self%rows=self%n;self%exceeded=.false.
  else
   call require(ieee_is_finite(memory),'working_memory must be finite for computed distances')
   capacity=memory*1048576._dp
   call require(ieee_is_finite(capacity),'working_memory exceeds finite working range')
   capacity=anint((capacity-modulo(capacity,8._dp*self%m))/(8._dp*self%m))
   self%exceeded=capacity<1
   self%rows=self%n
   if(capacity<self%n)self%rows=max(1,int(floor(capacity)))
  end if
  if(present(p))then
   call require(self%kind==6,'p is only supported for Minkowski chunks')
   call require(.not.ieee_is_nan(p),'Minkowski p must not be NaN');self%power=p
  end if
  if(present(squared))then
   call require(any(self%kind==[1,3,8]),'squared is only supported for Euclidean chunks')
   self%square=squared
  end if
  self%missing=ieee_value(0._dp,ieee_quiet_nan)
  if(present(missing_values))then
   call require(self%kind==8,'missing_values requires nan_euclidean')
   self%missing=missing_values
   if(.not.ieee_is_nan(missing_values))then
    call check_view(a,.false.);call check_view(b,.false.)
   end if
  end if
  if(present(w))then
   call require(self%kind==6,'this binding supports w for Minkowski chunks only')
   call require(size(w)==self%p,'Minkowski weight count differs from features')
   call require(all(ieee_is_finite(w)).and.all(w>=0),'Minkowski weights must be finite and nonnegative')
   self%weights=w
  end if
  if(present(v))then
   call require(self%kind==22,'V requires seuclidean')
   call require(size(v)==self%p,'V length differs from features');self%variance=v
  end if
  if(present(vi))then
   call require(self%kind==23,'VI requires mahalanobis')
   call require(all(shape(vi)==[self%p,self%p]),'VI shape differs from features');self%precision=vi
  end if
  if(self%kind==22.and..not.allocated(self%variance))then
   call require(self%self_pair,'seuclidean requires V for distinct input objects')
   allocate(self%variance(self%p));self%variance=0
   do j=1,self%p
    mean_=sum(self%xd(:,j))/self%n;total=0
    do i=1,self%n
     total=total+(self%xd(i,j)-mean_)**2
    end do
    self%variance(j)=total/real(self%n-1,dp)
   end do
  end if
  if(self%kind==23.and..not.allocated(self%precision))then
   call require(self%self_pair,'mahalanobis requires VI for distinct input objects')
   call require(self%p>1,'automatic mahalanobis covariance requires multiple features')
   call covariance_precision(a,self%precision)
  end if
  call ieee_set_flag(ieee_all,flags);call ieee_set_halting_mode(ieee_all,modes)
 end subroutine
 function slice_rows(a,first,last)result(b)
  type(csr_matrix),intent(in)::a
  integer,intent(in)::first,last
  type(csr_matrix)::b
  integer::lo,hi
  lo=a%indptr(first);hi=a%indptr(last+1)-1
  b%n_rows=last-first+1;b%n_cols=a%n_cols
  b%indptr=a%indptr(first:last+1)-lo+1;b%indices=a%indices(lo:hi);b%data=a%data(lo:hi)
 end function
 ! NumPy 2.2.6 einsum_sumprod.c.src; (c) 2011 Mark Wiebe, UBC, retained NumPy BSD notice.
 real(dp)function norm(row)result(value)
  real(dp),intent(in)::row(:)
  real(dp)::a,b
  integer::i,n,whole
  n=size(row);whole=n-mod(n,8);a=0;b=0
  do i=1,whole,8
   a=row(i)**2+(row(i+2)**2+(row(i+4)**2+(row(i+6)**2+a)))
   b=row(i+1)**2+(row(i+3)**2+(row(i+5)**2+(row(i+7)**2+b)))
  end do
  do i=whole+1,n,2
   a=a+row(i)**2
   if(i+1<=n)b=b+row(i+1)**2
  end do
  value=a+b
 end function
 function norms(a)result(values)
  type(row_view),intent(in)::a
  real(dp)::values(a%n),value
  integer::i,k
  do i=1,a%n
   if(associated(a%dense))then
    values(i)=norm(a%dense(i,:))
   else
    value=0
    do k=a%sparse%indptr(i),a%sparse%indptr(i+1)-1
     value=value+a%sparse%data(k)**2
    end do
    values(i)=value
   end if
  end do
 end function
 subroutine normalize_view(a)
  type(row_view),intent(inout)::a
  real(dp)::scale(a%n)
  integer::i,k
  scale=sqrt(norms(a))
  do i=1,a%n
   if(associated(a%dense))then
    if(scale(i)<10*epsilon(1._dp))scale(i)=1
    a%dense(i,:)=a%dense(i,:)/scale(i)
   else
    if(scale(i)==0)cycle
    do k=a%sparse%indptr(i),a%sparse%indptr(i+1)-1
     a%sparse%data(k)=a%sparse%data(k)/scale(i)
    end do
   end if
  end do
 end subroutine
 function gram(a,b)result(values)
  type(row_view),intent(in)::a,b
  real(dp),allocatable::values(:,:)
  if(associated(a%dense))then
   if(associated(b%dense))then
    values=pairwise_kernels(a%dense,b%dense,'linear')
   else
    values=pairwise_kernels(a%dense,b%sparse,'linear')
   end if
  else
   if(associated(b%dense))then
    values=pairwise_kernels(a%sparse,b%dense,'linear')
   else
    values=pairwise_kernels(a%sparse,b%sparse,'linear')
   end if
  end if
 end function
 function raw_gram(x,y)result(values)
  ! Permit nonfinite intermediate squares, as NumPy dot does after validation.
  real(dp),intent(in)::x(:,:),y(:,:)
  real(dp)::values(size(x,1),size(y,1))
  integer::i,j
  do j=1,size(y,1)
   do i=1,size(x,1)
    values(i,j)=kernel_blas_dot(x(i,:),y(j,:),size(x,1),size(y,1),i,j)
   end do
  end do
 end function
 logical function next_chunk(self,chunk,reducer)result(available)
  class(pairwise_distance_iterator),target,intent(inout)::self
  type(distance_chunk),intent(out)::chunk
  class(pairwise_chunk_reducer),optional,intent(inout)::reducer
  type(row_view)::a,b,original_x,original_y
  real(dp),allocatable,target::xd(:,:),yd(:,:)
  type(csr_matrix),target::xs,ys
  real(dp),allocatable::u(:),v(:),nx(:),ny(:),empty_precision(:,:)
  real(dp)::value,num,den
  integer::first,last,i,j,k
  logical,allocatable::mx(:,:),my(:,:)
  logical::whole,modes(size(ieee_all)),flags(size(ieee_all)),mask(self%p)
  available=self%position<=self%n
  if(.not.available)return
  call ieee_get_halting_mode(ieee_all,modes);call ieee_get_flag(ieee_all,flags)
  call ieee_set_halting_mode(ieee_all,.false.)
  first=self%position;last=min(self%n,first+min(self%rows-1,self%n-first))
  whole=self%self_pair.and.first==1.and.last==self%n
  chunk%distance%start=first-1;chunk%distance%n_rows=last-first+1
  if(self%kind==-1)then
   chunk%distance%is_sparse=self%sparse_x
   if(self%sparse_x)then
    chunk%distance%sparse=self%xs
   else
    chunk%distance%values=self%xd
   end if
  else
   call views(self,original_x,original_y);b=original_y
   if(self%sparse_x)then
    xs=slice_rows(self%xs,first,last);a%sparse=>xs;a%n=xs%n_rows;a%p=self%p
   else
    a%dense=>self%xd(first:last,:);a%n=last-first+1;a%p=self%p
   end if
   if(any(self%kind==[1,3,7]))then
    if(self%kind==7)then
     if(self%sparse_x)then
      xs=a%sparse;a%sparse=>xs
     else
      xd=a%dense;a%dense=>xd
     end if
     if(self%sparse_y)then
      ys=b%sparse;b%sparse=>ys
     else
      yd=b%dense;b%dense=>yd
     end if
     call normalize_view(a);call normalize_view(b)
     chunk%distance%values=1-gram(a,b)
     where(chunk%distance%values<0)chunk%distance%values=0
     where(chunk%distance%values>2)chunk%distance%values=2
    else
     nx=norms(a);ny=norms(b);chunk%distance%values=-2*gram(a,b)
     do j=1,b%n
      do i=1,a%n
       value=(chunk%distance%values(i,j)+nx(i))+ny(j)
       if(.not.ieee_is_nan(value))value=max(0._dp,value)
       if(.not.self%square)value=sqrt(value)
       chunk%distance%values(i,j)=value
      end do
     end do
    end if
    if(self%self_pair.and.(self%kind/=7.or.whole))then
     do i=1,a%n
      chunk%distance%values(i,first+i-1)=0
     end do
    end if
   else if(self%kind==8)then
    xd=self%xd(first:last,:);yd=self%yd
    if(ieee_is_nan(self%missing))then
     mx=ieee_is_nan(xd);my=ieee_is_nan(yd)
    else
     mx=xd==self%missing;my=yd==self%missing
    end if
    where(mx)xd=0
    where(my)yd=0
    a%dense=>xd;b%dense=>yd;nx=norms(a);ny=norms(b)
    chunk%distance%values=-2*pairwise_kernels(xd,yd,'linear')
    do j=1,b%n
     do i=1,a%n
      value=(chunk%distance%values(i,j)+nx(i))+ny(j)
      if(.not.ieee_is_nan(value))value=max(0._dp,value)
      chunk%distance%values(i,j)=value
     end do
    end do
    if(whole)then
     do i=1,a%n
      chunk%distance%values(i,i)=0
     end do
    end if
    chunk%distance%values=chunk%distance%values-raw_gram(xd**2,merge(1._dp,0._dp,my))
    chunk%distance%values=chunk%distance%values-raw_gram(merge(1._dp,0._dp,mx),yd**2)
    do j=1,b%n
     do i=1,a%n
      k=count(.not.mx(i,:).and..not.my(j,:))
      if(k==0)then
       value=ieee_value(0._dp,ieee_quiet_nan)
      else
       value=chunk%distance%values(i,j)
       if(.not.ieee_is_nan(value))value=max(0._dp,value)
       if(whole.and.i==j)value=0
       value=value/real(k,dp)*self%p
       if(.not.self%square)value=sqrt(value)
      end if
      chunk%distance%values(i,j)=value
     end do
    end do
   else
    allocate(chunk%distance%values(a%n,b%n),u(self%p),v(self%p),empty_precision(0,0));mask=.true.
    do i=1,a%n
     call get_row(a,i,u)
     if(self%kind==9)call get_haversine_row(a,i,u)
     do j=1,b%n
      if(associated(self%callback).and.whole.and.i>=j)cycle
      if(whole.and.self%kind/=0.and.self%kind/=9.and.i==j)then
       chunk%distance%values(i,j)=0;cycle
      end if
      call get_row(b,j,v)
      if(self%kind==9)call get_haversine_row(b,j,v)
      if(associated(self%callback))then
       value=self%callback(u,v)
      else if(self%kind==10)then
       num=0;den=0
       do k=1,self%p
        num=num+abs(u(k)-v(k));den=den+abs(u(k)+v(k))
       end do
       value=num/den
      else if(self%kind>=10.and.self%kind<=21)then
       value=extended_distance(u,v,self%kind,.false.,.false.,mask,mask)
      else if(self%kind==22)then
       value=parameter_distance(u,v,22,self%power,v=self%variance,vi=empty_precision);value=sqrt(value)
      else if(self%kind==23)then
       value=parameter_distance(u,v,23,self%power,vi=self%precision)
      else if(self%kind==6.and.allocated(self%weights).and..not.ieee_is_finite(self%power))then
       value=0
       do k=1,self%p
        if(self%weights(k)>0)value=max(value,abs(u(k)-v(k)))
       end do
      else if(self%kind==6.and.allocated(self%weights))then
       value=parameter_distance(u,v,6,self%power,w=self%weights,vi=empty_precision)
       value=value**(1/self%power)
      else
       value=row_distance(u,v,self%kind,self%power,.false.,.false.,.false.)
       select case(self%kind)
       case(2);value=sum((u-v)**2)
       case(6);if(ieee_is_finite(self%power))value=value**(1/self%power)
       case(9);value=2*asin(sqrt(value))
       end select
      end if
      chunk%distance%values(i,j)=value
      if(associated(self%callback).and.whole)chunk%distance%values(j,i)=value
     end do
    end do
    if(associated(self%callback).and.whole)then
     do i=1,a%n
      call get_row(a,i,u);chunk%distance%values(i,i)=self%callback(u,u)
     end do
    end if
   end if
  end if
  if(present(reducer))then
   chunk%is_reduced=.true.;call reducer%apply(chunk%distance,chunk%reduced)
   call check_reduction(chunk%reduced,last-first+1)
   if(allocated(chunk%distance%values))deallocate(chunk%distance%values)
   chunk%distance%sparse=csr_matrix()
  end if
  self%position=last+1
  call ieee_set_flag(ieee_all,flags);call ieee_set_halting_mode(ieee_all,modes)
 end function
 recursive subroutine check_reduction(value,n,in_tuple)
  type(distance_reduction),intent(in)::value
  integer,intent(in)::n
  logical,optional,intent(in)::in_tuple
  integer::count_,i
  count_=0
  if(allocated(value%real_vector))then
   count_=count_+1;call require(size(value%real_vector)==n,'reduced chunk row count differs')
  end if
  if(allocated(value%real_matrix))then
   count_=count_+1;call require(size(value%real_matrix,1)==n,'reduced chunk row count differs')
  end if
  if(allocated(value%integer_vector))then
   count_=count_+1;call require(size(value%integer_vector)==n,'reduced chunk row count differs')
  end if
  if(allocated(value%integer_matrix))then
   count_=count_+1;call require(size(value%integer_matrix,1)==n,'reduced chunk row count differs')
  end if
  if(allocated(value%sparse%data))then
   count_=count_+1;call value%sparse%validate()
   call require(value%sparse%n_rows==n,'reduced chunk row count differs')
  end if
  if(allocated(value%tuple))then
   if(present(in_tuple))call require(.not.in_tuple,'reduced chunk nested tuples are not supported')
   count_=count_+1
   do i=1,size(value%tuple)
    call check_reduction(value%tuple(i),n,.true.)
   end do
  end if
  if(present(in_tuple))call require(count_==1,'reduced chunk tuple members must be sequences')
  call require(count_<=1,'reduced chunk has multiple payload types; use a tuple')
 end subroutine
end module
