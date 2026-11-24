! Numeric sklearn 1.6.1 OneHotEncoder, preprocessing/_encoders.py (BSD).
! Typed owned CSR and inverse object masks are original MIT integration.
module sklearn_one_hot_encoding
 use sklearn_common,only:dp,require
 use sklearn_numpy_ordering,only:numpy_argsort
 use sklearn_target_encoding,only:onehot_category=>target_category
 use sklearn_feature_extraction,only:feature_string
 use sklearn_sparse,only:csr_matrix
 use sklearn_ieee_math,only:sub=>ieee_difference
 use,intrinsic::iso_fortran_env,only:int64
 use,intrinsic::ieee_arithmetic
 implicit none
 private
 public::one_hot_encoder,onehot_category,onehot_inverse_result,onehot_name_combiner,onehot_name_value
 type::onehot_name_value
  real(dp)::number=0
  logical::is_string=.false.
  character(:),allocatable::text
 end type
 abstract interface
  function onehot_name_combiner(feature,category) result(text)
   import feature_string,onehot_name_value
   character(*),intent(in)::feature
   type(onehot_name_value),intent(in)::category
   type(feature_string)::text
  end function
 end interface
 type::onehot_inverse_result
  real(dp),allocatable::values(:,:)
  logical,allocatable::is_none(:,:),is_infrequent(:,:)
 contains
  procedure::as_numeric=>inverse_values
 end type
 type::category_mapping
  integer,allocatable::counts(:),codes(:),representatives(:)
  logical,allocatable::rare(:)
 end type
 type::one_hot_encoder
  logical::handle_unknown_ignore=.false.
  character(24)::handle_unknown='error'
  character(12)::drop='none'
  real(dp),allocatable::drop_values(:)
  integer::min_frequency_count=0,max_categories=0,n_features_out_=0,n_features_in_=0
  real(dp)::min_frequency_fraction=0
  type(onehot_category),allocatable::categories(:),categories_(:),infrequent_categories_(:)
  integer,allocatable::drop_idx_(:)
  type(feature_string),allocatable::feature_names_in_(:)
  procedure(onehot_name_combiner),pointer,nopass::feature_name_combiner=>null()
  type(category_mapping),allocatable,private::maps_(:)
  integer,allocatable,private::drop_group_(:),widths_(:)
  logical,private::grouping_=.false.
 contains
  procedure::fit=>onehot_fit
  procedure::transform=>onehot_transform
  procedure::transform_csr=>onehot_csr
  procedure::fit_transform=>onehot_fit_transform
  procedure::fit_transform_csr=>onehot_fit_csr
  procedure::get_feature_names_out=>onehot_names
  procedure,private::inverse_dense,inverse_csr
  generic::inverse_transform=>inverse_dense,inverse_csr
 end type
contains
 function mode(self) result(value)
  class(one_hot_encoder),intent(in)::self
  character(24)::value
  value=self%handle_unknown
  if(self%handle_unknown_ignore) value='ignore'
 end function
 subroutine data_check(x)
  real(dp),intent(in)::x(:,:)
  call require(size(x,1)>0.and.size(x,2)>0,'OneHotEncoder needs a nonempty matrix')
  call require(all(ieee_is_finite(x).or.ieee_is_nan(x)),'OneHotEncoder rejects infinity')
 end subroutine
 logical function equal(a,b)
  real(dp),intent(in)::a,b
  equal=.false.
  if(ieee_is_nan(a).or.ieee_is_nan(b)) then
   equal=ieee_is_nan(a).and.ieee_is_nan(b)
  else
   equal=a==b
  end if
 end function
 function unique(x) result(values)
  real(dp),intent(in)::x(:)
  real(dp),allocatable::values(:),a(:),work(:)
  integer,allocatable::order(:)
  integer::i,n
  a=pack(x,.not.ieee_is_nan(x));order=numpy_argsort(a);a=a(order)
  allocate(work(size(x)));n=0
  do i=1,size(a)
   if(i>1) then
    if(a(i)==a(i-1)) cycle
   end if
   n=n+1;work(n)=a(i)
  end do
  if(any(ieee_is_nan(x))) then
   n=n+1;work(n)=ieee_value(0._dp,ieee_quiet_nan)
  end if
  values=work(:n)
 end function
 integer function find(values,x) result(k)
  real(dp),intent(in)::values(:),x
  integer::left,right,middle
  k=0;right=size(values)
  if(right==0) return
  if(ieee_is_nan(x)) then
   if(ieee_is_nan(values(right))) k=right
   return
  end if
  if(ieee_is_nan(values(right))) right=right-1
  left=1
  do while(left<=right)
   middle=left+(right-left)/2
   if(values(middle)==x) then
    k=middle;return
   else if(values(middle)<x) then
    left=middle+1
   else
    right=middle-1
   end if
  end do
 end function
 function stable_counts(counts) result(ids)
  integer,intent(in)::counts(:)
  integer::ids(size(counts)),work(size(counts)),n,w,left,mid,right,i,j,k
  n=size(counts);ids=[(i,i=1,n)];w=1
  do while(w<n)
   left=1
   do while(left<=n)
    mid=left+min(w,n-left+1);right=mid+min(w,n-mid+1);i=left;j=mid
    do k=left,right-1
     if(i>=mid) then
      work(k)=ids(j);j=j+1
     else if(j>=right) then
      work(k)=ids(i);i=i+1
     else if(counts(ids(i))<=counts(ids(j))) then
      work(k)=ids(i);i=i+1
     else
      work(k)=ids(j);j=j+1
     end if
    end do
    left=right
   end do
   ids=work;if(w>n/2) exit
   w=w*2
  end do
 end function
 subroutine fit_names(self,p,names)
  class(one_hot_encoder),intent(inout)::self
  integer,intent(in)::p
  type(feature_string),optional,intent(in)::names(:)
  integer::i
  if(allocated(self%feature_names_in_)) deallocate(self%feature_names_in_)
  if(present(names)) then
   call require(size(names)==p,'OneHotEncoder feature name count mismatch')
   do i=1,p
    call require(allocated(names(i)%text),'OneHotEncoder feature name is unallocated')
   end do
   self%feature_names_in_=names
  end if
 end subroutine
 subroutine onehot_fit(self,x,feature_names)
  class(one_hot_encoder),intent(inout)::self
  real(dp),intent(in)::x(:,:)
  type(feature_string),optional,intent(in)::feature_names(:)
  real(dp),allocatable::values(:),sorted(:)
  integer,allocatable::order(:)
  logical,allocatable::rare(:)
  integer::i,j,k,p,ncat,nrare,frequent,groups,code,dropindex,raw
  character(24)::unknown
  call data_check(x);p=size(x,2);unknown=mode(self)
  call require(unknown=='error'.or.unknown=='ignore'.or.unknown=='infrequent_if_exist'.or.unknown=='warn', &
   'invalid OneHotEncoder handle_unknown')
  call require(self%drop=='none'.or.self%drop=='first'.or.self%drop=='if_binary','invalid OneHotEncoder drop mode')
  call require(self%min_frequency_count>=0,'OneHotEncoder minimum count must be nonnegative')
  call require(ieee_is_finite(self%min_frequency_fraction),'OneHotEncoder frequency fraction must be finite')
  call require(self%min_frequency_fraction>=0.and.self%min_frequency_fraction<1, &
   'OneHotEncoder frequency fraction must be in [0,1)')
  call require(self%min_frequency_count==0.or.self%min_frequency_fraction==0, &
   'OneHotEncoder count and fraction are mutually exclusive')
  call require(self%max_categories>=0,'OneHotEncoder max_categories must be nonnegative')
  if(allocated(self%drop_values)) call require(size(self%drop_values)==p,'OneHotEncoder drop value count mismatch')
  self%n_features_in_=p;call fit_names(self,p,feature_names)
  if(allocated(self%categories)) call require(size(self%categories)==p,'OneHotEncoder category feature count mismatch')
  if(allocated(self%categories_)) deallocate(self%categories_,self%maps_,self%widths_)
  allocate(self%categories_(p),self%maps_(p),self%widths_(p))
  if(allocated(self%drop_idx_)) deallocate(self%drop_idx_,self%drop_group_)
  if(self%drop/='none'.or.allocated(self%drop_values)) then
   allocate(self%drop_idx_(p),self%drop_group_(p));self%drop_idx_=0;self%drop_group_=0
  end if
  self%grouping_=self%min_frequency_count>0.or.self%min_frequency_fraction>0.or.self%max_categories>0
  if(allocated(self%infrequent_categories_)) deallocate(self%infrequent_categories_)
  if(self%grouping_) allocate(self%infrequent_categories_(p))
  do j=1,p
   if(allocated(self%categories)) then
    call require(allocated(self%categories(j)%values),'OneHotEncoder category values are unallocated')
    values=self%categories(j)%values
    call require(size(values)>0,'OneHotEncoder categories must be nonempty')
    sorted=unique(values)
    call require(size(sorted)==size(values),'OneHotEncoder categories contain duplicates')
    do i=1,size(values)
     call require(equal(values(i),sorted(i)),'OneHotEncoder numeric categories must be sorted with NaN last')
    end do
   else
    values=unique(x(:,j))
   end if
   self%categories_(j)%values=values;ncat=size(values)
   allocate(self%maps_(j)%counts(ncat),self%maps_(j)%codes(ncat));self%maps_(j)%counts=0
   do i=1,size(x,1)
    k=find(values,x(i,j))
    if(k==0) then
     call require(unknown/='error','OneHotEncoder unknown category during fit')
    else
     self%maps_(j)%counts(k)=self%maps_(j)%counts(k)+1
    end if
   end do
   rare=self%maps_(j)%counts<self%min_frequency_count
   if(self%min_frequency_fraction>0) rare=real(self%maps_(j)%counts,dp)< &
    self%min_frequency_fraction*real(size(x,1),dp)
   if(self%max_categories>0.and.self%max_categories<ncat-count(rare)+1) then
    frequent=self%max_categories-1
    if(frequent==0) then
     rare=.true.
    else
     order=stable_counts(self%maps_(j)%counts);rare(order(:ncat-frequent))=.true.
    end if
   end if
   self%maps_(j)%rare=rare;nrare=count(rare);frequent=ncat-nrare;groups=frequent+merge(1,0,nrare>0)
   if(nrare>0) self%infrequent_categories_(j)%values=pack(values,rare)
   allocate(self%maps_(j)%representatives(groups));self%maps_(j)%representatives=0;code=0
   do i=1,ncat
    if(rare(i)) then
     self%maps_(j)%codes(i)=groups
    else
     code=code+1;self%maps_(j)%codes(i)=code;self%maps_(j)%representatives(code)=i
    end if
   end do
   dropindex=0
   if(allocated(self%drop_values)) then
    raw=find(values,self%drop_values(j));call require(raw>0,'OneHotEncoder drop category was not found')
    call require(.not.rare(raw),'OneHotEncoder cannot explicitly drop an infrequent category')
    dropindex=self%maps_(j)%codes(raw)
   else if(self%drop=='first'.or.self%drop=='if_binary'.and.groups==2) then
    dropindex=1
   end if
   if(allocated(self%drop_group_)) then
    self%drop_group_(j)=dropindex
    if(dropindex>0) then
     do i=1,ncat
      if(self%maps_(j)%codes(i)/=dropindex) cycle
      self%drop_idx_(j)=i;exit
     end do
    end if
   end if
   self%widths_(j)=groups-merge(1,0,dropindex>0)
  end do
  call require(sum(int(self%widths_,int64))<huge(1),'OneHotEncoder encoded width overflow')
  self%n_features_out_=sum(self%widths_)
 end subroutine
 subroutine fitted(self)
  class(one_hot_encoder),intent(in)::self
  call require(allocated(self%categories_),'one_hot_encoder is not fitted')
 end subroutine
 function input_names(self,input_features) result(names)
  class(one_hot_encoder),intent(in)::self
  type(feature_string),optional,intent(in)::input_features(:)
  type(feature_string),allocatable::names(:)
  character(32)::text
  integer::j
  if(present(input_features)) then
   call require(size(input_features)==self%n_features_in_,'OneHotEncoder input name count mismatch')
   do j=1,size(input_features)
    call require(allocated(input_features(j)%text),'OneHotEncoder feature name is unallocated')
    if(allocated(self%feature_names_in_)) then
     call require(len(input_features(j)%text)==len(self%feature_names_in_(j)%text), &
      'OneHotEncoder input feature names differ')
     call require(input_features(j)%text==self%feature_names_in_(j)%text,'OneHotEncoder input feature names differ')
    end if
   end do
   names=input_features
  else if(allocated(self%feature_names_in_)) then
   names=self%feature_names_in_
  else
   allocate(names(self%n_features_in_))
   do j=1,size(names)
    write(text,'(i0)') j-1;names(j)%text='x'//trim(text)
   end do
  end if
 end function
 function encoded(self,x,feature_names) result(code)
  class(one_hot_encoder),intent(in)::self
  real(dp),intent(in)::x(:,:)
  type(feature_string),optional,intent(in)::feature_names(:)
  integer::code(size(x,1),size(x,2)),i,j,k,dropindex
  type(feature_string),allocatable::ignored(:)
  character(24)::unknown
  logical::valid,has_rare
  call fitted(self);call data_check(x)
  call require(size(x,2)==self%n_features_in_,'OneHotEncoder feature count mismatch')
  if(present(feature_names)) ignored=input_names(self,feature_names)
  unknown=mode(self)
  do j=1,size(x,2)
   dropindex=0;if(allocated(self%drop_group_)) dropindex=self%drop_group_(j)
   has_rare=any(self%maps_(j)%rare)
   do i=1,size(x,1)
    k=find(self%categories_(j)%values,x(i,j));valid=k>0
    if(.not.valid) then
     call require(unknown/='error','unknown category during OneHotEncoder transform')
     if(has_rare.and.unknown=='infrequent_if_exist') then
      k=size(self%maps_(j)%representatives);valid=.true.
     end if
    else
     k=self%maps_(j)%codes(k)
    end if
    if(.not.valid.or.k==dropindex) then
     code(i,j)=0
    else
     if(dropindex>0.and.k>dropindex) k=k-1
     code(i,j)=k
    end if
   end do
  end do
 end function
 function onehot_csr(self,x,feature_names) result(z)
  class(one_hot_encoder),intent(in)::self
  real(dp),intent(in)::x(:,:)
  type(feature_string),optional,intent(in)::feature_names(:)
  type(csr_matrix)::z
  integer,allocatable::code(:,:)
  integer::i,j,offset,pos
  code=encoded(self,x,feature_names)
  call require(count(code>0,kind=int64)<huge(1),'OneHotEncoder sparse stored count overflow')
  z%n_rows=size(x,1);z%n_cols=self%n_features_out_
  allocate(z%indptr(z%n_rows+1),z%indices(count(code>0)),z%data(count(code>0)));z%data=1;pos=1
  do i=1,z%n_rows
   z%indptr(i)=pos;offset=0
   do j=1,size(x,2)
    if(code(i,j)>0) then
     z%indices(pos)=offset+code(i,j);pos=pos+1
    end if
    offset=offset+self%widths_(j)
   end do
  end do
  z%indptr(z%n_rows+1)=pos;call z%validate()
 end function
 function onehot_transform(self,x,feature_names) result(z)
  class(one_hot_encoder),intent(in)::self
  real(dp),intent(in)::x(:,:)
  type(feature_string),optional,intent(in)::feature_names(:)
  real(dp),allocatable::z(:,:)
  type(csr_matrix)::s
  s=self%transform_csr(x,feature_names);z=s%to_dense()
 end function
 function onehot_fit_transform(self,x,feature_names) result(z)
  class(one_hot_encoder),intent(inout)::self
  real(dp),intent(in)::x(:,:)
  type(feature_string),optional,intent(in)::feature_names(:)
  real(dp),allocatable::z(:,:)
  call self%fit(x,feature_names);z=self%transform(x,feature_names)
 end function
 function onehot_fit_csr(self,x,feature_names) result(z)
  class(one_hot_encoder),intent(inout)::self
  real(dp),intent(in)::x(:,:)
  type(feature_string),optional,intent(in)::feature_names(:)
  type(csr_matrix)::z
  call self%fit(x,feature_names);z=self%transform_csr(x,feature_names)
 end function
 recursive real(dp) function total(a) result(value)
  real(dp),intent(in)::a(:)
  real(dp)::r(8)
  integer::n,i,j,m,half
  n=size(a);value=-0._dp
  if(n<8) then
   do i=1,n
    value=sub(value,-a(i))
   end do
  else if(n<=128) then
   r=a(:8);m=n-mod(n,8)
   do i=9,m,8
    do j=1,8
     r(j)=sub(r(j),-a(i+j-1))
    end do
   end do
   value=sub(sub(sub(r(1),-r(2)),-sub(r(3),-r(4))),-sub(sub(r(5),-r(6)),-sub(r(7),-r(8))))
   do i=m+1,n
    value=sub(value,-a(i))
   end do
  else
   half=n/2;half=half-mod(half,8);value=sub(total(a(:half)),-total(a(half+1:)))
  end if
 end function
 logical function iszero(x)
  real(dp),intent(in)::x
  iszero=.false.;if(ieee_is_nan(x)) return
  iszero=x==0
 end function
 integer function argmax(row) result(index)
  real(dp),intent(in)::row(:)
  integer::i
  index=1
  do i=1,size(row)
   if(ieee_is_nan(row(i))) then
    index=i;return
   end if
   if(row(i)>row(index)) index=i
  end do
 end function
 subroutine result_init(self,n,z)
  class(one_hot_encoder),intent(in)::self
  integer,intent(in)::n
  type(onehot_inverse_result),intent(out)::z
  allocate(z%values(n,self%n_features_in_),z%is_none(n,self%n_features_in_), &
   z%is_infrequent(n,self%n_features_in_))
  z%values=ieee_value(0._dp,ieee_quiet_nan);z%is_none=.false.;z%is_infrequent=.false.
 end subroutine
 subroutine assign_group(self,z,i,j,group)
  class(one_hot_encoder),intent(in)::self
  type(onehot_inverse_result),intent(inout)::z
  integer,intent(in)::i,j,group
  integer::raw
  call require(group>=1.and.group<=size(self%maps_(j)%representatives), &
   'OneHotEncoder inverse grouped index is invalid')
  raw=self%maps_(j)%representatives(group)
  if(raw==0) then
   z%is_infrequent(i,j)=.true.
  else
   z%values(i,j)=self%categories_(j)%values(raw)
  end if
 end subroutine
 subroutine inverse_row(self,row,sumrow,i,j,z,zero_position,zero_count)
  class(one_hot_encoder),intent(in)::self
  real(dp),intent(in)::row(:),sumrow
  integer,intent(in)::i,j,zero_position,zero_count
  type(onehot_inverse_result),intent(inout)::z
  integer::dropindex,group
  character(24)::unknown
  logical::has_rare,ignore
  unknown=mode(self);has_rare=any(self%maps_(j)%rare)
  dropindex=0;if(allocated(self%drop_group_)) dropindex=self%drop_group_(j)
  if(size(row)==0) then
   ! Baseline uses the grouped index into the raw category array here.
   call require(dropindex>0,'OneHotEncoder empty inverse category block')
   z%values(i,j)=self%categories_(j)%values(dropindex);return
  end if
  if(iszero(sumrow)) then
   ignore=unknown=='ignore'.or.(unknown=='infrequent_if_exist'.or.unknown=='warn').and..not.has_rare
   if(ignore) then
    if(dropindex==0) then
     z%is_none(i,j)=.true.
    else
     ! sklearn's ignore path also indexes raw categories by the grouped drop index.
     z%values(i,j)=self%categories_(j)%values(dropindex)
    end if
   else
    call require(allocated(self%drop_group_),'OneHotEncoder inverse row has zero sum without a drop')
    if(dropindex>0) then
     call assign_group(self,z,i,j,dropindex)
    else
     ! Baseline cats[None] broadcasts its whole grouped category vector to
     ! zero-sum rows when drop='if_binary' leaves this feature undropped.
     group=size(self%maps_(j)%representatives)
     call require(group==1.or.group==zero_count,'OneHotEncoder inverse zero-sum category broadcast mismatch')
     call assign_group(self,z,i,j,merge(1,zero_position,group==1))
    end if
   end if
  else
   group=argmax(row)
   if(dropindex>0.and.group>=dropindex) group=group+1
   call assign_group(self,z,i,j,group)
  end if
 end subroutine
 function inverse_dense(self,x) result(z)
  class(one_hot_encoder),intent(in)::self
  real(dp),intent(in)::x(:,:)
  type(onehot_inverse_result)::z
  integer::i,j,k,offset,nzero,position
  real(dp),allocatable::sums(:)
  call fitted(self)
  call require(size(x,1)>0.and.size(x,2)>0,'OneHotEncoder inverse requires nonempty encoded input')
  call require(all(ieee_is_finite(x)),'OneHotEncoder inverse rejects NaN or infinity')
  call require(size(x,2)==self%n_features_out_,'OneHotEncoder inverse encoded width mismatch')
  call result_init(self,size(x,1),z);offset=0
  do j=1,self%n_features_in_
   k=self%widths_(j)
   sums=[(total(x(i,offset+1:offset+k)),i=1,size(x,1))];nzero=0;position=0
   do i=1,size(sums)
    if(iszero(sums(i))) nzero=nzero+1
   end do
   do i=1,size(x,1)
    if(iszero(sums(i))) position=position+1
    call inverse_row(self,x(i,offset+1:offset+k),sums(i),i,j,z,position,nzero)
   end do
   offset=offset+k
  end do
 end function
 function inverse_csr(self,x) result(z)
  class(one_hot_encoder),intent(in)::self
  type(csr_matrix),intent(in)::x
  type(onehot_inverse_result)::z
  real(dp),allocatable::row(:),sums(:)
  integer::i,j,t,k,offset,nzero,position
  call fitted(self);call x%validate()
  call require(x%n_rows>0.and.x%n_cols>0,'OneHotEncoder inverse requires nonempty encoded input')
  call require(all(ieee_is_finite(x%data)),'OneHotEncoder inverse rejects NaN or infinity')
  call require(x%n_cols==self%n_features_out_,'OneHotEncoder inverse encoded width mismatch')
  call result_init(self,x%n_rows,z)
  offset=0;allocate(sums(x%n_rows))
  do j=1,self%n_features_in_
   k=self%widths_(j);nzero=0;position=0
   do i=1,x%n_rows
    call sparse_row(x,i,offset,k,row);sums(i)=0
    do t=1,k
     sums(i)=sub(sums(i),-row(t))
    end do
    if(iszero(sums(i))) nzero=nzero+1
   end do
   do i=1,x%n_rows
    call sparse_row(x,i,offset,k,row)
    if(iszero(sums(i))) position=position+1
    call inverse_row(self,row,sums(i),i,j,z,position,nzero)
   end do
   offset=offset+k
  end do
 end function
 subroutine sparse_row(x,i,offset,k,row)
  type(csr_matrix),intent(in)::x
  integer,intent(in)::i,offset,k
  real(dp),allocatable,intent(out)::row(:)
  integer::t,col
  row=spread(0._dp,1,k)
  do t=x%indptr(i),x%indptr(i+1)-1
   col=x%indices(t)-offset
   if(col<1.or.col>k) cycle
   row(col)=sub(row(col),-x%data(t))
  end do
 end subroutine
 function inverse_values(self) result(values)
  class(onehot_inverse_result),intent(in)::self
  real(dp),allocatable::values(:,:)
  call require(allocated(self%values),'OneHotEncoder inverse result is unallocated')
  call require(.not.any(self%is_none).and..not.any(self%is_infrequent), &
   'OneHotEncoder inverse result contains None or infrequent category; inspect its masks')
  values=self%values
 end function
 logical function roundtrips(text,x) result(ok)
  use,intrinsic::ieee_exceptions
  character(*),intent(in)::text
  real(dp),intent(in)::x
  type(ieee_flag_type)::flags(3)
  logical::raised(3),halt(3)
  integer::i,status
  real(dp)::value
  flags=[ieee_invalid,ieee_divide_by_zero,ieee_overflow]
  do i=1,3
   call ieee_get_halting_mode(flags(i),halt(i));call ieee_get_flag(flags(i),raised(i))
  end do
  do i=1,3
   call ieee_set_halting_mode(flags(i),.false.)
  end do
  read(text,*,iostat=status) value
  ok=.false.;if(status==0) ok=equal(value,x)
  do i=1,3
   call ieee_set_halting_mode(flags(i),halt(i))
  end do
  do i=1,3
   call ieee_set_flag(flags(i),raised(i))
  end do
 end function
 function numeric_text(x) result(text)
  real(dp),intent(in)::x
  character(:),allocatable::text,digits,sign,exptext
  character(64)::buffer,fmt
  integer::precision,pos,exponent,status,point,n
  if(ieee_is_nan(x)) then
   text='nan';return
  else if(.not.ieee_is_finite(x)) then
   text='inf';if(x<0) text='-inf'
   return
  else if(x==0) then
   text='0.0';if(ieee_is_negative(x)) text='-0.0'
   return
  end if
  do precision=1,17
   write(fmt,'(a,i0,a)') '(es26.',precision-1,'e3)'
   write(buffer,fmt,round='nearest') x
   if(roundtrips(buffer,x)) exit
  end do
  text=trim(adjustl(buffer));pos=index(text,'E');read(text(pos+1:),*,iostat=status) exponent
  call require(status==0,'OneHotEncoder category formatting failed')
  digits=text(:pos-1);sign=''
  if(digits(1:1)=='-') then
   sign='-';digits=digits(2:)
  end if
  pos=index(digits,'.');digits=digits(:pos-1)//digits(pos+1:);n=len(digits)
  if(exponent>=-4.and.exponent<16) then
   point=exponent+1
   if(point<=0) then
    text=sign//'0.'//repeat('0',-point)//digits
   else if(point>=n) then
    text=sign//digits//repeat('0',point-n)//'.0'
   else
    text=sign//digits(:point)//'.'//digits(point+1:)
   end if
  else
   write(buffer,'(i0)') abs(exponent);exptext=trim(buffer)
   if(len(exptext)<2) exptext='0'//exptext
   if(exponent<0) then
    exptext='e-'//exptext
   else
    exptext='e+'//exptext
   end if
   if(n==1) then
    text=sign//digits//exptext
   else
    text=sign//digits(:1)//'.'//digits(2:)//exptext
   end if
  end if
 end function
 function onehot_names(self,input_features) result(names)
  class(one_hot_encoder),intent(in)::self
  type(feature_string),optional,intent(in)::input_features(:)
  type(feature_string),allocatable::names(:),input(:)
  real(dp)::value
  type(onehot_name_value)::category
  type(feature_string)::combined
  logical::infrequent
  character(:),allocatable::text
  integer::j,k,pos,dropindex,raw
  call fitted(self);input=input_names(self,input_features);allocate(names(self%n_features_out_));pos=0
  if(associated(self%feature_name_combiner)) then
   category%is_string=.true.;category%text='category'
   combined=self%feature_name_combiner('feature',category)
   call require(allocated(combined%text),'OneHotEncoder name callback returned unallocated text')
  end if
  do j=1,self%n_features_in_
   dropindex=0;if(allocated(self%drop_group_)) dropindex=self%drop_group_(j)
   do k=1,size(self%maps_(j)%representatives)
    if(k==dropindex) cycle
    raw=self%maps_(j)%representatives(k);infrequent=raw==0
    value=ieee_value(0._dp,ieee_quiet_nan)
    if(.not.infrequent) value=self%categories_(j)%values(raw)
    if(associated(self%feature_name_combiner)) then
     category%is_string=infrequent;category%number=value
     if(allocated(category%text)) deallocate(category%text)
     if(infrequent) category%text='infrequent_sklearn'
     combined=self%feature_name_combiner(input(j)%text,category)
     call require(allocated(combined%text),'OneHotEncoder name callback returned unallocated text')
     text=combined%text
    else
     if(infrequent) then
      text='infrequent_sklearn'
     else
      text=numeric_text(value)
     end if
     text=input(j)%text//'_'//text
    end if
    pos=pos+1;names(pos)%text=text
   end do
  end do
 end function
end module
