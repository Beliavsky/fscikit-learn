! Adapted from sklearn 1.6.1 metrics/_dist_metrics.pyx.tp (retained sklearn BSD).
! Owned typed state and factory adapters are native MIT contributions.
module sklearn_distance_metrics
 use sklearn_common,only:dp,require
 use sklearn_sparse,only:csr_matrix
 use sklearn_pairwise_reductions,only:row_view,check_view,get_row,get_haversine_row,extended_distance
 use,intrinsic::ieee_arithmetic
 use,intrinsic::ieee_exceptions
 implicit none
 private
 public::distance_metric,get_metric
 abstract interface
  function point_function(x,y)result(value)
   import dp
   real(dp),intent(in)::x(:),y(:)
   real(dp)::value
  end function
 end interface
 type::distance_metric
  private
  integer::kind=0
  real(dp)::power=2
  real(dp),allocatable::weights(:),variance(:),precision(:,:),buffer(:)
  logical,allocatable::undefined(:,:)
  procedure(point_function),pointer,nopass::callback=>null()
 contains
  procedure,private::pair_dd,pair_ds,pair_sd,pair_ss
  generic::pairwise=>pair_dd,pair_ds,pair_sd,pair_ss
  procedure::dist_to_rdist
  procedure::rdist_to_dist
  procedure::type_name
  procedure::undefined_pairs
 end type
 interface get_metric
  module procedure get_named,get_callback
 end interface
contains
 subroutine quiet(modes,flags)
  logical,intent(out)::modes(:),flags(:)
  call ieee_get_halting_mode(ieee_all,modes);call ieee_get_flag(ieee_all,flags)
  call ieee_set_halting_mode(ieee_all,.false.)
 end subroutine
 subroutine restore(modes,flags)
  logical,intent(in)::modes(:),flags(:)
  call ieee_set_flag(ieee_all,flags);call ieee_set_halting_mode(ieee_all,modes)
 end subroutine
 function get_named(name,p,w,v,vi,func)result(metric)
  character(*),intent(in)::name
  real(dp),optional,intent(in)::p,w(:),v(..),vi(:,:)
  procedure(point_function),optional::func
  type(distance_metric)::metric
  real(dp),allocatable::covariance(:,:)
  logical::modes(size(ieee_all)),flags(size(ieee_all)),minkowski
  call quiet(modes,flags)
  allocate(metric%weights(0),metric%variance(0),metric%precision(0,0),metric%buffer(0))
  minkowski=name=='minkowski'.or.name=='p'
  if(minkowski)then
   if(present(p))metric%power=p
   metric%kind=6
   if(.not.present(w))then
    if(metric%power==1)metric%kind=4
    if(metric%power==2)metric%kind=3
    if(.not.ieee_is_finite(metric%power).and..not.ieee_is_nan(metric%power))metric%kind=5
   end if
   if(metric%kind==6)then
    call require(.not.(metric%power<=0),'Minkowski p must be greater than zero')
    call require(ieee_is_finite(metric%power).or.ieee_is_nan(metric%power),'weighted Minkowski p must be finite')
    if(present(w))then
     call require(size(w)>0,'Minkowski weights must be nonempty')
     call require(all(ieee_is_finite(w)),'Minkowski weights must be finite')
     call require(all(w>=0),'Minkowski weights must be nonnegative');metric%weights=w
    end if
   end if
  else
   call require(.not.present(p).and..not.present(w),'p and w require Minkowski')
   select case(name)
   case('euclidean','l2');metric%kind=3
   case('manhattan','cityblock','l1');metric%kind=4
   case('chebyshev','infinity');metric%kind=5
   case('seuclidean');metric%kind=22
   case('mahalanobis');metric%kind=23
   case('hamming');metric%kind=13
   case('canberra');metric%kind=11
   case('braycurtis');metric%kind=10
   case('matching');metric%kind=21
   case('jaccard');metric%kind=15
   case('dice');metric%kind=14
   case('kulsinski');metric%kind=24
   case('rogerstanimoto');metric%kind=16
   case('russellrao');metric%kind=17
   case('sokalmichener');metric%kind=18
   case('sokalsneath');metric%kind=19
   case('haversine');metric%kind=9
   case('pyfunc');metric%kind=25
   case default;call require(.false.,'unrecognized DistanceMetric')
   end select
  end if
  call require(.not.present(v).or.any(metric%kind==[22,23]),'V is not applicable to this metric')
  call require(.not.present(vi).or.metric%kind==23,'VI requires Mahalanobis')
  call require(.not.present(func).or.metric%kind==25,'func requires pyfunc')
  if(metric%kind==22)then
   call require(present(v),'SEuclidean requires V')
   select rank(v)
   rank(1);metric%variance=v
   rank default;call require(.false.,'SEuclidean V must be a vector')
   end select
  else if(metric%kind==23)then
   if(present(vi))then
    call require(size(vi,1)==size(vi,2),'Mahalanobis VI must be square');metric%precision=vi
   else
    call require(present(v),'Mahalanobis requires V or VI')
    select rank(v)
    rank(2);covariance=v
    rank default;call require(.false.,'Mahalanobis V must be a matrix')
    end select
    metric%precision=inverse(covariance)
   end if
   metric%buffer=spread(0._dp,1,size(metric%precision,1))
  else if(metric%kind==25)then
   call require(present(func),'pyfunc requires func');metric%callback=>func
  end if
  call restore(modes,flags)
 end function
 function get_callback(func)result(metric)
  procedure(point_function)::func
  type(distance_metric)::metric
  metric=get_named('pyfunc',func=func)
 end function
 function inverse(a)result(value)
  real(dp),intent(in)::a(:,:)
  real(dp)::value(size(a,1),size(a,2)),work(size(a,1),size(a,2)),row(size(a,2)),pivot,factor
  integer::i,k,j,n
  n=size(a,1);call require(size(a,2)==n,'Mahalanobis V must be square')
  ! NumPy inv solves A X = I through GESV. Preserve partial pivoting and
  ! reciprocal scaling: Gauss-Jordan changes cancellation in CSR distances.
  work=a;value=0
  do i=1,n;value(i,i)=1;end do
  do k=1,n
   ! Ordered comparison retains an initial NaN, as BLAS IDAMAX does.
   j=k
   do i=k+1,n
    if(abs(work(i,k))>abs(work(j,k)))j=i
   end do
   pivot=work(j,k)
   call require(pivot/=0,'singular Mahalanobis covariance')
   row=work(k,:);work(k,:)=work(j,:);work(j,:)=row
   row=value(k,:);value(k,:)=value(j,:);value(j,:)=row
   if(abs(pivot)>=tiny(pivot))then
    factor=1._dp/pivot;work(k+1:,k)=work(k+1:,k)*factor
   else
    work(k+1:,k)=work(k+1:,k)/pivot
   end if
   do i=k+1,n
    work(i,k+1:)=work(i,k+1:)-work(i,k)*work(k,k+1:)
   end do
  end do
  ! Unit lower solve followed by an upper solve with reciprocal diagonals.
  do k=1,n
   do i=k+1,n
    value(i,:)=value(i,:)-work(i,k)*value(k,:)
   end do
  end do
  do k=n,1,-1
   factor=1._dp/work(k,k);value(k,:)=value(k,:)*factor
   do i=1,k-1
    value(i,:)=value(i,:)-work(i,k)*value(k,:)
   end do
  end do
 end function
 function type_name(self)result(name)
  class(distance_metric),intent(in)::self
  character(32)::name
  select case(self%kind)
  case(3);name='EuclideanDistance64'
  case(4);name='ManhattanDistance64'
  case(5);name='ChebyshevDistance64'
  case(6);name='MinkowskiDistance64'
  case(9);name='HaversineDistance64'
  case(10);name='BrayCurtisDistance64'
  case(11);name='CanberraDistance64'
  case(13);name='HammingDistance64'
  case(14);name='DiceDistance64'
  case(15);name='JaccardDistance64'
  case(16);name='RogersTanimotoDistance64'
  case(17);name='RussellRaoDistance64'
  case(18);name='SokalMichenerDistance64'
  case(19);name='SokalSneathDistance64'
  case(21);name='MatchingDistance64'
  case(22);name='SEuclideanDistance64'
  case(23);name='MahalanobisDistance64'
  case(24);name='KulsinskiDistance64'
  case(25);name='PyFuncDistance64'
  case default;name='Uninitialized'
  end select
 end function
 impure elemental real(dp)function dist_to_rdist(self,d)result(value)
  class(distance_metric),intent(in)::self
  real(dp),intent(in)::d
  logical::modes(size(ieee_all)),flags(size(ieee_all))
  call require(self%kind>0,'DistanceMetric is uninitialized');call quiet(modes,flags)
  value=d
  select case(self%kind)
  case(3,22,23);value=d*d
  case(6);value=numpy_power(d,self%power)
  case(9);value=sin(.5_dp*d)**2
  end select
  call restore(modes,flags)
 end function
 impure elemental real(dp)function rdist_to_dist(self,d)result(value)
  class(distance_metric),intent(in)::self
  real(dp),intent(in)::d
  logical::modes(size(ieee_all)),flags(size(ieee_all))
  call require(self%kind>0,'DistanceMetric is uninitialized');call quiet(modes,flags)
  value=d
  select case(self%kind)
  case(3,22,23);value=sqrt(d)
  case(6);value=numpy_power(d,1._dp/self%power)
  case(9);value=2*asin(sqrt(d))
  end select
  call restore(modes,flags)
 end function
 real(dp)function numpy_power(x,p)result(value)
  real(dp),intent(in)::x,p
  ! NumPy's power ufunc specializes exponent one-half to sqrt, including -Inf.
  if(p==.5_dp)then
   value=sqrt(x)
  else
   value=x**p
  end if
 end function
 subroutine haversine_point(a,i,csr_path,point)
  type(row_view),intent(in)::a
  integer,intent(in)::i
  logical,intent(in)::csr_path
  real(dp),intent(out)::point(2)
  integer::k
  point=0
  if(associated(a%sparse))then
   call get_haversine_row(a,i,point)
  else if(csr_path)then
   ! Mixed input routines represent dense rows using a full CSR index vector.
   do k=1,a%p
    if(k==1)then
     point(1)=a%dense(i,k)
    else
     point(2)=a%dense(i,k)
    end if
   end do
  else
   point=a%dense(i,:)
  end if
 end subroutine
 function pair_dd(self,x,y)result(d)
  class(distance_metric),intent(inout)::self
  real(dp),target,intent(in)::x(:,:)
  real(dp),target,optional,intent(in)::y(:,:)
  real(dp),allocatable::d(:,:)
  type(row_view)::a,b
  a%dense=>x;a%n=size(x,1);a%p=size(x,2);b=a
  if(present(y))then
   b%dense=>y;b%n=size(y,1);b%p=size(y,2)
  end if
  d=compute(self,a,b,associated(a%dense,b%dense))
 end function
 function pair_ss(self,x,y)result(d)
  class(distance_metric),intent(inout)::self
  type(csr_matrix),target,intent(in)::x
  type(csr_matrix),target,optional,intent(in)::y
  real(dp),allocatable::d(:,:)
  type(row_view)::a,b
  a%sparse=>x;a%n=x%n_rows;a%p=x%n_cols;b=a
  if(present(y))then
   b%sparse=>y;b%n=y%n_rows;b%p=y%n_cols
  end if
  d=compute(self,a,b,associated(a%sparse,b%sparse))
 end function
 function pair_ds(self,x,y)result(d)
  class(distance_metric),intent(inout)::self
  real(dp),target,intent(in)::x(:,:)
  type(csr_matrix),target,intent(in)::y
  real(dp),allocatable::d(:,:)
  type(row_view)::a,b
  a%dense=>x;a%n=size(x,1);a%p=size(x,2)
  b%sparse=>y;b%n=y%n_rows;b%p=y%n_cols
  d=compute(self,a,b,.false.)
 end function
 function pair_sd(self,x,y)result(d)
  class(distance_metric),intent(inout)::self
  type(csr_matrix),target,intent(in)::x
  real(dp),target,intent(in)::y(:,:)
  real(dp),allocatable::d(:,:)
  type(row_view)::a,b
  a%sparse=>x;a%n=x%n_rows;a%p=x%n_cols
  b%dense=>y;b%n=size(y,1);b%p=size(y,2)
  d=compute(self,a,b,.false.)
 end function
 subroutine mask(a,i,values)
  type(row_view),intent(in)::a
  integer,intent(in)::i
  logical,intent(out)::values(:)
  integer::k
  values=.true.
  if(associated(a%sparse))then
   values=.false.
   do k=a%sparse%indptr(i),a%sparse%indptr(i+1)-1;values(a%sparse%indices(k))=.true.;end do
  end if
 end subroutine
 function compute(self,a,b,same)result(d)
  class(distance_metric),intent(inout)::self
  type(row_view),intent(in)::a,b
  logical,intent(in)::same
  real(dp)::d(a%n,b%n),u(a%p),v(a%p),value,tmp,den,tt,neq,hu(2),hv(2)
  logical::mu(a%p),mv(a%p),sparse,unsafe,modes(size(ieee_all)),flags(size(ieee_all))
  integer::i,j,k,l,first
  call require(self%kind>0,'DistanceMetric is uninitialized')
  call check_view(a,.false.);call check_view(b,.false.);call require(a%p==b%p,'distance feature counts differ')
  if(size(self%weights)>0)call require(size(self%weights)==a%p,'Minkowski weight count differs from features')
  if(self%kind==22)call require(size(self%variance)==a%p,'SEuclidean V length differs from features')
  if(self%kind==23)call require(size(self%precision,1)==a%p,'Mahalanobis V/VI size differs from features')
  sparse=associated(a%sparse).or.associated(b%sparse)
  if(self%kind==9.and..not.sparse)call require(a%p==2,'dense haversine requires two features')
  call quiet(modes,flags)
  if(allocated(self%undefined))deallocate(self%undefined)
  allocate(self%undefined(a%n,b%n));self%undefined=.false.
  do i=1,a%n
   call get_row(a,i,u);call mask(a,i,mu)
   if(self%kind==9)call haversine_point(a,i,sparse,hu)
   first=1;if(same)first=i
   do j=first,b%n
    call get_row(b,j,v);call mask(b,j,mv)
    if(self%kind==9)call haversine_point(b,j,sparse,hv)
    value=0
    if(sparse.and..not.any(self%kind==[9,10,25]))then
     value=csr_point(self,a,b,i,j)
    else
    select case(self%kind)
    case(3,4,5,6,22)
     do k=1,a%p
      if(sparse.and..not.(mu(k).or.mv(k)))cycle
      tmp=abs(u(k)-v(k))
      select case(self%kind)
      case(3);value=value+tmp*tmp
      case(4);value=value+tmp
      case(5);value=max(value,tmp)
      case(6)
       tmp=tmp**self%power
       if(size(self%weights)>0)tmp=tmp*self%weights(k)
       value=value+tmp
      case(22);value=value+(tmp*tmp)/self%variance(k)
      end select
     end do
     call check_reduced(value)
     value=self%rdist_to_dist(value)
    case(9)
     tmp=sin(.5_dp*(hu(1)-hv(1)));den=sin(.5_dp*(hu(2)-hv(2)))
     value=2*asin(sqrt(tmp*tmp+cos(hu(1))*cos(hv(1))*den*den))
    case(10:21)
     if(self%kind==10.and.sparse)then
      value=bray_csr(a,b,i,j,unsafe);self%undefined(i,j)=unsafe
      if(same)self%undefined(j,i)=unsafe
     else
      value=extended_distance(u,v,self%kind,associated(a%sparse),associated(b%sparse),mu,mv)
     end if
    case(23)
     where(mu.or.mv)self%buffer=u-v
     do k=1,a%p
      tmp=0
      do l=1,a%p;tmp=tmp+self%precision(k,l)*self%buffer(l);end do
      value=value+tmp*self%buffer(k)
     end do
     call check_reduced(value);value=sqrt(value)
    case(24)
     tt=count(u/=0.and.v/=0);neq=count((u/=0).neqv.(v/=0))
     value=(neq-tt+a%p)/(neq+a%p)
    case(25)
     if(sparse)then
      ! PyFuncDistance inherits this CSR sentinel from the Source base class.
      value=-999
     else
      value=self%callback(u,v)

     end if
    end select
    end if
    call require(value/=-1,'DistanceMetric returned the reserved Cython error value -1')
    d(i,j)=value;if(same)d(j,i)=value
   end do
  end do
  call restore(modes,flags)
 end function
 real(dp)function csr_point(self,a,b,row_a,row_b)result(value)
  class(distance_metric),intent(inout)::self
  type(row_view),intent(in)::a,b
  integer,intent(in)::row_a,row_b
  integer::i,j,ei,ej,ix,jx,column,k,l
  real(dp)::x,y,d,den,tt,neq,tmp
  logical::matched,bu,bv,unsafe
  call row_bounds(a,row_a,i,ei);call row_bounds(b,row_b,j,ej)
  value=0;tt=0;neq=0;unsafe=.false.
  do while(i<=ei.or.j<=ej)
   matched=.false.;x=0;y=0
   if(i>ei)then
    column=raw_column(b,j);y=raw_value(b,row_b,j,unsafe);j=j+1
   else if(j>ej)then
    column=raw_column(a,i);x=raw_value(a,row_a,i,unsafe);i=i+1
   else
    ix=raw_column(a,i);jx=raw_column(b,j)
    if(ix==jx)then
     column=ix;matched=.true.
     x=raw_value(a,row_a,i,unsafe);y=raw_value(b,row_b,j,unsafe);i=i+1;j=j+1
    else if(ix<jx)then
     column=ix;x=raw_value(a,row_a,i,unsafe);i=i+1
    else
     column=jx;y=raw_value(b,row_b,j,unsafe);j=j+1
    end if
   end if
   d=abs(x-y)
   select case(self%kind)
   case(3);value=value+d*d
   case(4);value=value+d
   case(5);value=max(value,d)
   case(6)
    tmp=d**self%power
    if(size(self%weights)>0)tmp=tmp*self%weights(column)
    value=value+tmp
   case(11)
    if(matched)then
     den=abs(x)+abs(y);value=value+d/den
    else
     value=value+1
    end if
   case(13);if(x/=y)value=value+1
   case(14:21,24)
    bu=x/=0;bv=y/=0
    if(bu.and.bv)tt=tt+1
    if(bu.neqv.bv)neq=neq+1
   case(22);value=value+(d*d)/self%variance(column)
   case(23);self%buffer(column)=x-y
   end select
  end do
  select case(self%kind)
  case(3,22);call check_reduced(value);value=sqrt(value)
  case(6);value=value**(1._dp/self%power)
  case(13);value=value/a%p
  case(14);value=neq/(2*tt+neq)
  case(15)
   value=0;if(tt+neq>0)value=neq/(tt+neq)
  case(16,18);value=2*neq/(a%p+neq)
  case(17);value=(a%p-tt)/a%p
  case(19);value=2*neq/(2*neq+tt)
  case(21);value=neq/a%p
  case(23)
   value=0
   do k=1,a%p
    tmp=0
    do l=1,a%p;tmp=tmp+self%precision(k,l)*self%buffer(l);end do
    value=value+tmp*self%buffer(k)
   end do
   call check_reduced(value);value=sqrt(value)
  case(24);value=(neq-tt+a%p)/(neq+a%p)
  end select
 end function
 subroutine check_reduced(value)
  real(dp),intent(in)::value
  call require(value/=-1,'DistanceMetric returned the reserved Cython reduced error value -1')
 end subroutine
 function undefined_pairs(self)result(mask)
  class(distance_metric),intent(in)::self
  logical,allocatable::mask(:,:)
  if(allocated(self%undefined))then
   mask=self%undefined
  else
   allocate(mask(0,0))
  end if
 end function
 subroutine row_bounds(a,row,first,last)
  type(row_view),intent(in)::a
  integer,intent(in)::row
  integer,intent(out)::first,last
  first=1;last=a%p
  if(associated(a%sparse))then
   first=a%sparse%indptr(row);last=a%sparse%indptr(row+1)-1
  end if
 end subroutine
 integer function raw_column(a,cursor)result(column)
  type(row_view),intent(in)::a
  integer,intent(in)::cursor
  column=cursor
  if(associated(a%sparse))column=a%sparse%indices(cursor)
 end function
 real(dp)function raw_value(a,row,cursor,unsafe)result(value)
  type(row_view),intent(in)::a
  integer,intent(in)::row,cursor
  logical,intent(inout)::unsafe
  integer::r,c
  value=ieee_value(0._dp,ieee_quiet_nan)
  if(associated(a%sparse))then
   if(cursor>=1.and.cursor<=size(a%sparse%data))then
    value=a%sparse%data(cursor)
   else
    unsafe=.true.
   end if
  else
   ! Reconstruct the C-layout dense-row pointer used by Source's fake CSR view.
   r=row+(cursor-1)/a%p;c=mod(cursor-1,a%p)+1
   if(r>=1.and.r<=a%n.and.cursor>=1)then
    value=a%dense(r,c)
   else
    unsafe=.true.
   end if
  end if
 end function
 real(dp)function bray_csr(a,b,row_a,row_b,unsafe)result(value)
  type(row_view),intent(in)::a,b
  integer,intent(in)::row_a,row_b
  logical,intent(out)::unsafe
  integer::i,j,last_i,last_j,ix,jx
  real(dp)::num,den,x,y
  unsafe=.false.;num=0;den=0
  call row_bounds(a,row_a,i,last_i);call row_bounds(b,row_b,j,last_j)
  do while(i<=last_i.and.j<=last_j)
   ix=raw_column(a,i);jx=raw_column(b,j)
   if(ix==jx)then
    x=raw_value(a,row_a,i,unsafe);y=raw_value(b,row_b,j,unsafe)
    num=num+abs(x-y);den=den+(abs(x)+abs(y));i=i+1;j=j+1
   else if(ix<jx)then
    x=abs(raw_value(a,row_a,i,unsafe));num=num+x;den=den+x;i=i+1
   else
    y=abs(raw_value(b,row_b,j,unsafe));num=num+y;den=den+y;j=j+1
   end if
  end do
  ! These tails intentionally read the exhausted OTHER cursor, as in Source.
  ! Reads within the global allocation are reproducible, even across row ends.
  ! Undefined reads beyond that allocation are marked and never dereferenced.
  if(i==last_i+1)then
   do while(j<=last_j)
    x=abs(raw_value(a,row_a,i,unsafe));num=num+x;den=den+x;j=j+1
   end do
  else
   do while(i<=last_i)
    y=abs(raw_value(b,row_b,j,unsafe));num=num+y;den=den+y;i=i+1
   end do
  end if
  value=num/den
  if(unsafe)value=ieee_value(0._dp,ieee_quiet_nan)
 end function
end module
