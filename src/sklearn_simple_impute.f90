! sklearn 1.6.1 impute/_base.py SimpleImputer, masked statistics and indicators.
! Retained sklearn/NumPy BSD notices are under LICENSES/; native adapters are MIT.
module sklearn_simple_impute
 use sklearn_common,only:dp,require,argsort
 use sklearn_sparse,only:csr_matrix
 use sklearn_feature_extraction,only:feature_string
 use sklearn_ieee_math,only:ieee_difference,ieee_quotient
 use,intrinsic::iso_fortran_env,only:int64
 use,intrinsic::ieee_arithmetic,only:ieee_is_nan,ieee_is_finite
 implicit none
 private
 public::simple_imputer,imputer_statistic
 real(dp),parameter::qnan=transfer(int(z'7FF8000000000000',int64),0._dp)
 abstract interface
  function imputer_statistic(x) result(value)
   import dp
   real(dp),intent(in)::x(:)
   real(dp)::value
  end function
 end interface
 type::simple_indicator
  integer::n_features_in_=0
  integer,allocatable::features_(:) ! Native indices are one-based.
 end type
 type::simple_imputer
  character(16)::strategy='mean'
  real(dp)::fill_value=0
  real(dp),allocatable::statistics_(:)
  real(dp)::missing_values=qnan
  logical::add_indicator=.false.,keep_empty_features=.false.
  integer::n_features_in_=0
  type(simple_indicator)::indicator_
  type(feature_string),allocatable::feature_names_in_(:)
  procedure(imputer_statistic),pointer,nopass::statistic=>null()
 contains
  procedure,private::fit_dense,fit_csr,transform_dense,transform_csr,fit_transform_dense,fit_transform_csr
  generic::fit=>fit_dense,fit_csr
  generic::transform=>transform_dense,transform_csr
  generic::fit_transform=>fit_transform_dense,fit_transform_csr
  procedure::inverse_transform=>inverse_dense
  procedure::get_feature_names_out=>feature_names
 end type
contains
 logical function missing(self,x) result(value)
  class(simple_imputer),intent(in)::self
  real(dp),intent(in)::x
  value=ieee_is_nan(x)
  if(ieee_is_nan(self%missing_values)) return
  if(value) then
   value=.false.
  else
   value=x==self%missing_values
  end if
 end function
 subroutine dense_check(self,x)
  class(simple_imputer),intent(in)::self
  real(dp),intent(in)::x(:,:)
  call require(size(x,1)>0.and.size(x,2)>0,'imputer X is empty')
  if(ieee_is_nan(self%missing_values)) then
   call require(all(ieee_is_finite(x).or.ieee_is_nan(x)),'SimpleImputer input contains infinity')
  else
   call require(all(ieee_is_finite(x)),'SimpleImputer input contains NaN or infinity')
  end if
 end subroutine
 subroutine sparse_check(self,x)
  class(simple_imputer),intent(in)::self
  type(csr_matrix),intent(in)::x
  call x%validate()
  call require(x%n_rows>0.and.x%n_cols>0,'imputer X is empty')
  if(ieee_is_nan(self%missing_values)) then
   call require(all(ieee_is_finite(x%data).or.ieee_is_nan(x%data)),'SimpleImputer input contains infinity')
  else
   call require(all(ieee_is_finite(x%data)),'SimpleImputer input contains NaN or infinity')
   call require(self%missing_values/=0,'SimpleImputer sparse missing_values cannot be zero')
  end if
 end subroutine
 subroutine initialize(self,p,names)
  class(simple_imputer),intent(inout)::self
  integer,intent(in)::p
  type(feature_string),intent(in),optional::names(:)
  call require(self%strategy=='mean'.or.self%strategy=='median'.or.self%strategy=='most_frequent'.or. &
   self%strategy=='constant'.or.self%strategy=='callable','unknown imputer strategy')
  if(self%strategy=='callable') call require(associated(self%statistic),'SimpleImputer callable is not associated')
  self%n_features_in_=p
  if(allocated(self%statistics_)) deallocate(self%statistics_)
  allocate(self%statistics_(p));self%statistics_=qnan
  if(allocated(self%feature_names_in_)) deallocate(self%feature_names_in_)
  if(present(names)) then
   call require(size(names)==p,'SimpleImputer feature name count mismatch')
   self%feature_names_in_=names
  end if
  if(allocated(self%indicator_%features_)) deallocate(self%indicator_%features_)
  self%indicator_%n_features_in_=0
 end subroutine
 subroutine fit_indicator(self,has_missing)
  class(simple_imputer),intent(inout)::self
  logical,intent(in)::has_missing(:)
  integer::j
  if(self%add_indicator) then
   self%indicator_%features_=pack([(j,j=1,size(has_missing))],has_missing)
   self%indicator_%n_features_in_=size(has_missing)
  end if
 end subroutine
 recursive real(dp) function numpy_total(a) result(value)
  real(dp),intent(in)::a(:)
  real(dp)::lane(8)
  integer::i,j,n,last,split
  n=size(a);value=-0._dp
  if(n<8) then
   do i=1,n
    value=ieee_difference(value,-a(i))
   end do
  else if(n<=128) then
   lane=a(:8);last=n-modulo(n,8)
   do i=9,last,8
    do j=1,8
     lane(j)=ieee_difference(lane(j),-a(i+j-1))
    end do
   end do
   value=ieee_difference(ieee_difference(ieee_difference(lane(1),-lane(2)), &
    -ieee_difference(lane(3),-lane(4))), &
    -ieee_difference(ieee_difference(lane(5),-lane(6)),-ieee_difference(lane(7),-lane(8))))
   do i=last+1,n
    value=ieee_difference(value,-a(i))
   end do
  else
   split=n/2;split=split-modulo(split,8)
   value=ieee_difference(numpy_total(a(:split)),-numpy_total(a(split+1:)))
  end if
 end function
 real(dp) function ordered_total(a,pairwise) result(value)
  real(dp),intent(in)::a(:)
  logical,intent(in)::pairwise
  integer::i
  value=0
  if(pairwise) then
   value=numpy_total(a)
  else
   do i=1,size(a)
    value=ieee_difference(value,-a(i))
   end do
  end if
 end function
 real(dp) function element(a,zeros,negative,rank) result(value)
  real(dp),intent(in)::a(:)
  integer,intent(in)::zeros,negative,rank ! Zero-based rank, including Python negatives.
  integer::idx
  if(rank<negative) then
   idx=rank
  else if(rank-negative<zeros) then
   value=0;return
  else
   idx=rank-zeros
  end if
  if(idx<0) idx=idx+size(a)
  call require(idx>=0.and.idx<size(a),'SimpleImputer invalid sparse median rank')
  value=a(idx+1)
 end function
 real(dp) function median(a,zeros,dense) result(value)
  real(dp),intent(in)::a(:)
  integer,intent(in)::zeros
  logical,intent(in)::dense
  real(dp),allocatable::sorted(:)
  real(dp)::left,right
  integer::n,m,negative
  value=qnan;n=size(a)+zeros
  if(n==0) return
  sorted=a(argsort(a));negative=count(sorted<0);m=floor(real(n,dp)/2)
  right=element(sorted,zeros,negative,m)
  if(modulo(n,2)==1.and..not.dense) then
   value=right;return
  end if
  left=right
  if(modulo(n,2)==0) left=element(sorted,zeros,negative,m-1)
  value=ieee_quotient(ieee_difference(left,-right),2._dp)
 end function
 real(dp) function mode(a,zeros) result(value)
  real(dp),intent(in)::a(:)
  integer,intent(in)::zeros
  real(dp),allocatable::sorted(:)
  integer::i,j,best,n
  value=0;best=0;sorted=a(argsort(a));i=1
  do while(i<=size(sorted))
   j=i+1
   do while(j<=size(sorted))
    if(sorted(j)/=sorted(i)) exit
    j=j+1
   end do
   n=j-i
   if(n>best) then
    value=sorted(i);best=n
   end if
   i=j
  end do
  if(best==0.and.zeros==0) then
   value=qnan
  else if(best<zeros) then
   value=0
  else if(best==zeros) then
   value=min(value,0._dp)
  end if
 end function
 subroutine fit_dense(self,x,feature_names)
  class(simple_imputer),intent(inout)::self
  real(dp),intent(in)::x(:,:)
  type(feature_string),intent(in),optional::feature_names(:)
  real(dp),allocatable::a(:),terms(:)
  logical::mask(size(x,1)),has_missing(size(x,2)),masked
  real(dp)::total
  integer::j,i,n
  call dense_check(self,x);call initialize(self,size(x,2),feature_names)
  do j=1,size(x,2)
   do i=1,size(x,1)
    mask(i)=missing(self,x(i,j))
   end do
   has_missing(j)=any(mask);a=pack(x(:,j),.not.mask);n=size(a)
   select case(self%strategy)
   case('constant');self%statistics_(j)=self%fill_value
   case('callable');self%statistics_(j)=self%statistic(a)
   case('mean')
    terms=x(:,j);where(mask) terms=0
    total=ordered_total(terms,size(x,2)==1)
    self%statistics_(j)=ieee_quotient(total,real(n,dp))
    masked=.not.ieee_is_finite(self%statistics_(j))
    if(ieee_is_finite(total)) masked=masked.or.abs(total)*tiny(1._dp)>=real(n,dp)
    ! NumPy masked division masks nonfinite results as well as empty counts.
    if(masked) then
     self%statistics_(j)=qnan;if(self%keep_empty_features) self%statistics_(j)=0
    end if
   case('median');self%statistics_(j)=median(a,0,.true.)
   case('most_frequent');self%statistics_(j)=mode(a,0)
   end select
   if(n==0.and.self%keep_empty_features.and.self%strategy/='constant'.and.self%strategy/='callable') &
    self%statistics_(j)=0
  end do
  call fit_indicator(self,has_missing)
 end subroutine
 subroutine grouped(x,ptr,values)
  type(csr_matrix),intent(in)::x
  integer,allocatable,intent(out)::ptr(:)
  real(dp),allocatable,intent(out)::values(:)
  integer::counts(x%n_cols),next(x%n_cols),i,j,k
  counts=0
  do k=1,size(x%data)
   counts(x%indices(k))=counts(x%indices(k))+1
  end do
  allocate(ptr(x%n_cols+1),values(size(x%data)));ptr(1)=1
  do j=1,x%n_cols
   ptr(j+1)=ptr(j)+counts(j)
  end do
  next=ptr(:x%n_cols)
  do i=1,x%n_rows
   do k=x%indptr(i),x%indptr(i+1)-1
    j=x%indices(k);values(next(j))=x%data(k);next(j)=next(j)+1
   end do
  end do
 end subroutine
 subroutine fit_csr(self,x,feature_names)
  class(simple_imputer),intent(inout)::self
  type(csr_matrix),intent(in)::x
  type(feature_string),intent(in),optional::feature_names(:)
  integer,allocatable::ptr(:)
  real(dp),allocatable::values(:),a(:)
  logical,allocatable::valid(:)
  logical::has_missing(x%n_cols)
  integer::j,i,first,last,zeros,d
  call sparse_check(self,x);call initialize(self,x%n_cols,feature_names)
  call require(x%n_cols<huge(1),'SimpleImputer feature capacity overflow')
  call grouped(x,ptr,values)
  do j=1,x%n_cols
   first=ptr(j);last=ptr(j+1)-1;d=last-first+1
   if(allocated(valid)) deallocate(valid)
   allocate(valid(d))
   do i=1,d
    valid(i)=.not.missing(self,values(first+i-1))
   end do
   has_missing(j)=any(.not.valid);a=pack(values(first:last),valid)
   zeros=x%n_rows-d+count(a==0);a=pack(a,a/=0)
   if(self%strategy=='constant') then
    self%statistics_(j)=self%fill_value
   else if(size(a)==0.and.self%keep_empty_features) then
    self%statistics_(j)=0
   else
    select case(self%strategy)
    case('mean')
     if(size(a)+zeros/=0) self%statistics_(j)=ieee_quotient(numpy_total(a),real(size(a)+zeros,dp))
    case('median');self%statistics_(j)=median(a,zeros,.false.)
    case('most_frequent');self%statistics_(j)=mode(a,zeros)
    case('callable');self%statistics_(j)=self%statistic(a)
    end select
   end if
  end do
  call fit_indicator(self,has_missing)
 end subroutine
 subroutine ready(self,p,feature_names)
  class(simple_imputer),intent(in)::self
  integer,intent(in)::p
  type(feature_string),intent(in),optional::feature_names(:)
  integer::j
  call require(allocated(self%statistics_),'simple_imputer is not fitted')
  call require(size(self%statistics_)==p,'imputer feature count mismatch')
  if(present(feature_names)) then
   call require(size(feature_names)==p,'SimpleImputer feature name count mismatch')
   if(allocated(self%feature_names_in_)) then
    do j=1,p
     call require(feature_names(j)%text==self%feature_names_in_(j)%text,'SimpleImputer feature names differ')
    end do
   end if
  end if
 end subroutine
 function retained(self) result(keep)
  class(simple_imputer),intent(in)::self
  logical::keep(size(self%statistics_))
  keep=.true.
  if(self%strategy/='constant'.and..not.self%keep_empty_features) keep=.not.ieee_is_nan(self%statistics_)
 end function
 integer function indicator_count(self) result(n)
  class(simple_imputer),intent(in)::self
  n=0
  if(self%add_indicator) then
   call require(allocated(self%indicator_%features_),'SimpleImputer missing fitted indicator')
   n=size(self%indicator_%features_)
  end if
 end function
 function transform_dense(self,x,feature_names) result(z)
  class(simple_imputer),intent(in)::self
  real(dp),intent(in)::x(:,:)
  type(feature_string),intent(in),optional::feature_names(:)
  real(dp),allocatable::z(:,:)
  logical::keep(size(x,2))
  integer::i,j,k,n,ind
  call ready(self,size(x,2),feature_names);call dense_check(self,x)
  keep=retained(self);n=count(keep);ind=indicator_count(self);allocate(z(size(x,1),n+ind));k=0
  do j=1,size(x,2)
   if(.not.keep(j)) cycle
   k=k+1;z(:,k)=x(:,j)
   do i=1,size(x,1)
    if(missing(self,x(i,j))) z(i,k)=self%statistics_(j)
   end do
  end do
  do j=1,ind
   do i=1,size(x,1)
    z(i,n+j)=merge(1._dp,0._dp,missing(self,x(i,self%indicator_%features_(j))))
   end do
  end do
 end function
 function transform_csr(self,x,feature_names) result(z)
  class(simple_imputer),intent(in)::self
  type(csr_matrix),intent(in)::x
  type(feature_string),intent(in),optional::feature_names(:)
  type(csr_matrix)::z
  integer,allocatable::column(:),indicator(:)
  logical,allocatable::keep(:)
  integer::i,j,k,n,ind,slot
  integer(int64)::capacity
  call ready(self,x%n_cols,feature_names);call sparse_check(self,x)
  keep=retained(self);n=count(keep);ind=indicator_count(self)
  allocate(column(x%n_cols),indicator(x%n_cols));column=0;indicator=0;j=0
  do i=1,x%n_cols
   if(keep(i)) then
    j=j+1;column(i)=j
   end if
  end do
  do i=1,ind
   indicator(self%indicator_%features_(i))=n+i
  end do
  capacity=0
  do k=1,size(x%data)
   if(keep(x%indices(k))) capacity=capacity+1
   if(indicator(x%indices(k))>0.and.missing(self,x%data(k))) capacity=capacity+1
  end do
  call require(capacity<int(huge(1),int64),'SimpleImputer sparse output capacity overflow')
  z%n_rows=x%n_rows;z%n_cols=n+ind
  allocate(z%indptr(x%n_rows+1),z%indices(int(capacity)),z%data(int(capacity)));slot=1;z%indptr(1)=1
  do i=1,x%n_rows
   do k=x%indptr(i),x%indptr(i+1)-1
    j=x%indices(k);if(.not.keep(j)) cycle
    z%indices(slot)=column(j);z%data(slot)=x%data(k)
    if(missing(self,x%data(k))) z%data(slot)=self%statistics_(j)
    slot=slot+1
   end do
   do k=x%indptr(i),x%indptr(i+1)-1
    j=x%indices(k)
    if(indicator(j)==0.or..not.missing(self,x%data(k))) cycle
    z%indices(slot)=indicator(j);z%data(slot)=1;slot=slot+1
   end do
   z%indptr(i+1)=slot
  end do
 end function
 function inverse_dense(self,x) result(z)
  class(simple_imputer),intent(in)::self
  real(dp),intent(in)::x(:,:)
  real(dp),allocatable::z(:,:)
  logical,allocatable::mask(:,:)
  integer::p,k,cut,stop,i,j,original,imputed,width
  call require(allocated(self%statistics_),'simple_imputer is not fitted')
  call require(self%add_indicator,'SimpleImputer inverse_transform requires add_indicator=True')
  k=indicator_count(self);p=size(self%statistics_);cut=size(x,2)-k;stop=cut
  if(cut<0) stop=max(size(x,2)+cut,0)
  width=size(x,2)-stop
  call require(width==k.or.width==1,'SimpleImputer inverse indicator shape mismatch')
  allocate(z(size(x,1),p),mask(size(x,1),p));z=0;mask=.false.
  do j=1,k
   do i=1,size(x,1)
    mask(i,self%indicator_%features_(j))=.true.
    if(.not.ieee_is_nan(x(i,stop+min(j,width)))) &
     mask(i,self%indicator_%features_(j))=x(i,stop+min(j,width))/=0
   end do
  end do
  original=1;imputed=1
  do while(imputed<=stop)
   call require(original<=p,'SimpleImputer inverse feature mapping exceeds original width')
   if(.not.all(mask(:,original))) then
    z(:,original)=x(:,imputed);imputed=imputed+1
   end if
   original=original+1
  end do
  do j=1,p
   where(mask(:,j)) z(:,j)=self%missing_values
  end do
 end function
 function feature_names(self,input_features) result(names)
  class(simple_imputer),intent(in)::self
  type(feature_string),intent(in),optional::input_features(:)
  type(feature_string),allocatable::names(:),input(:)
  integer::j,n,k,ind
  character(48)::buffer
  call ready(self,self%n_features_in_,input_features)
  allocate(input(self%n_features_in_))
  if(present(input_features)) then
   input=input_features
  else if(allocated(self%feature_names_in_)) then
   input=self%feature_names_in_
  else
   do j=1,size(input)
    write(buffer,'(i0)') j-1;input(j)%text='x'//trim(buffer)
   end do
  end if
  n=count(.not.ieee_is_nan(self%statistics_));ind=indicator_count(self);allocate(names(n+ind));k=0
  do j=1,size(input)
   if(ieee_is_nan(self%statistics_(j))) cycle
   k=k+1;names(k)=input(j)
  end do
  do j=1,ind
   names(n+j)%text='missingindicator_'//input(self%indicator_%features_(j))%text
  end do
 end function
 function fit_transform_dense(self,x,feature_names) result(z)
  class(simple_imputer),intent(inout)::self
  real(dp),intent(in)::x(:,:)
  type(feature_string),intent(in),optional::feature_names(:)
  real(dp),allocatable::z(:,:)
  call self%fit(x,feature_names);z=self%transform(x,feature_names)
 end function
 function fit_transform_csr(self,x,feature_names) result(z)
  class(simple_imputer),intent(inout)::self
  type(csr_matrix),intent(in)::x
  type(feature_string),intent(in),optional::feature_names(:)
  type(csr_matrix)::z
  call self%fit(x,feature_names);z=self%transform(x,feature_names)
 end function
end module
