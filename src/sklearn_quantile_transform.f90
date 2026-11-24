! sklearn 1.6.1 QuantileTransformer/quantile_transform; NumPy 2.2.6 percentiles.
! BSD notices: LICENSES/scikit-learn-COPYING, LICENSES/numpy-* and THIRD_PARTY_NOTICES.md.
module sklearn_quantile_transform
 use sklearn_common,only:dp,require,argsort
 use sklearn_sparse,only:csr_matrix
 use sklearn_legacy_rng,only:legacy_random_state
 use sklearn_ieee_math,only:ieee_difference,ieee_product,ieee_quotient
 use,intrinsic::iso_fortran_env,only:int64
 use,intrinsic::ieee_arithmetic,only:ieee_is_finite,ieee_is_nan,ieee_value,ieee_quiet_nan
 implicit none
 private
 public::quantile_transformer,quantile_transform
 real(dp),parameter::bounds_threshold=1.e-7_dp
 type::quantile_transformer
  integer::n_quantiles=1000,subsample=10000,random_state=0,n_quantiles_=0,n_features_in_=0
  character(16)::output_distribution='uniform'
  real(dp),allocatable::quantiles_(:,:),references_(:)
  logical::ignore_implicit_zeros=.false.
 contains
  procedure,private::fit_dense,fit_csr,transform_dense,transform_csr,inverse_dense,inverse_csr
  procedure,private::fit_transform_dense,fit_transform_csr
  generic::fit=>fit_dense,fit_csr
  generic::transform=>transform_dense,transform_csr
  generic::inverse_transform=>inverse_dense,inverse_csr
  generic::fit_transform=>fit_transform_dense,fit_transform_csr
 end type
 interface quantile_transform
  module procedure free_dense,free_csr
 end interface
contains
 subroutine quantile_check(x)
  real(dp),intent(in)::x(:,:)
  call require(size(x,1)>0.and.size(x,2)>0,'QuantileTransformer X must be nonempty')
  call require(all(ieee_is_finite(x).or.ieee_is_nan(x)),'QuantileTransformer input contains infinity')
 end subroutine
 subroutine sparse_check(self,x,inverse)
  class(quantile_transformer),intent(in)::self
  type(csr_matrix),intent(in)::x
  logical,intent(in)::inverse
  integer::k
  call x%validate()
  call require(x%n_rows>0.and.x%n_cols>0,'QuantileTransformer X must be nonempty')
  call require(all(ieee_is_finite(x%data).or.ieee_is_nan(x%data)),'QuantileTransformer input contains infinity')
  if(inverse.or.self%ignore_implicit_zeros) return
  do k=1,size(x%data)
   if(ieee_is_nan(x%data(k))) cycle
   call require(x%data(k)>=0,'QuantileTransformer only accepts non-negative sparse matrices')
  end do
 end subroutine
 subroutine distribution_check(self)
  class(quantile_transformer),intent(in)::self
  call require(self%output_distribution=='uniform'.or.self%output_distribution=='normal', &
   'invalid QuantileTransformer output_distribution')
 end subroutine
 subroutine controls(self)
  class(quantile_transformer),intent(in)::self
  call require(self%n_quantiles>=1,'QuantileTransformer n_quantiles must be positive')
  call require(self%subsample>=0,'QuantileTransformer subsample must be nonnegative')
  if(self%subsample>0) call require(self%n_quantiles<=self%subsample, &
   'QuantileTransformer n_quantiles exceeds subsample')
  call require(self%random_state>=0,'QuantileTransformer random_state must be nonnegative')
  call distribution_check(self)
 end subroutine
 subroutine initialize(self,n,p)
  class(quantile_transformer),intent(inout)::self
  integer,intent(in)::n,p
  integer::i
  real(dp)::step
  self%n_quantiles_=min(self%n_quantiles,n);self%n_features_in_=p
  if(allocated(self%quantiles_)) deallocate(self%quantiles_)
  if(allocated(self%references_)) deallocate(self%references_)
  allocate(self%quantiles_(self%n_quantiles_,p),self%references_(self%n_quantiles_))
  self%references_=0
  if(self%n_quantiles_>1) then
   step=1._dp/real(self%n_quantiles_-1,dp)
   self%references_=[(real(i-1,dp)*step,i=1,self%n_quantiles_)]
   self%references_(self%n_quantiles_)=1
  end if
 end subroutine
 function permutation(n,rng) result(order)
  integer,intent(in)::n
  type(legacy_random_state),intent(inout)::rng
  integer::order(n),i,j,temp
  order=[(i,i=1,n)]
  do i=n,2,-1
   j=rng%integer_index(i);temp=order(i);order(i)=order(j);order(j)=temp
  end do
 end function
 real(dp) function padded_value(a,zeros,negative,k) result(value)
  real(dp),intent(in)::a(:)
  integer,intent(in)::zeros,negative,k
  if(k<=negative) then
   value=a(k)
  else if(k<=negative+zeros) then
   value=0
  else
   value=a(k-zeros)
  end if
 end function
 subroutine percentiles(self,j,column,zeros,empty_zero)
  class(quantile_transformer),intent(inout)::self
  integer,intent(in)::j,zeros
  real(dp),intent(in)::column(:)
  logical,intent(in)::empty_zero
  real(dp),allocatable::a(:)
  real(dp)::position,t,lo,hi,difference,prob,value
  integer::n,i,left,right,negative
  a=pack(column,.not.ieee_is_nan(column));a=a(argsort(a));n=size(a)+zeros
  if(n==0) then
   self%quantiles_(:,j)=ieee_value(0._dp,ieee_quiet_nan)
   if(empty_zero) self%quantiles_(:,j)=0
   return
  end if
  negative=count(a<0)
  do i=1,self%n_quantiles_
   ! Source first multiplies reference fractions by 100, then divides by 100.
   prob=(self%references_(i)*100._dp)/100._dp;position=real(n-1,dp)*prob
   left=floor(position)+1;right=min(left+1,n);t=position-floor(position)
   lo=padded_value(a,zeros,negative,left);hi=padded_value(a,zeros,negative,right)
   difference=ieee_difference(hi,lo)
   if(t>=.5_dp) then
    value=ieee_difference(hi,ieee_product(difference,1-t))
   else
    value=ieee_difference(lo,-ieee_product(difference,t))
   end if
   if(i>1) then
    if(ieee_is_nan(value).or.ieee_is_nan(self%quantiles_(i-1,j))) then
     value=ieee_value(0._dp,ieee_quiet_nan)
    else
     value=max(value,self%quantiles_(i-1,j))
    end if
   end if
   self%quantiles_(i,j)=value
  end do
 end subroutine
 subroutine fit_dense(self,x)
  class(quantile_transformer),intent(inout)::self
  real(dp),intent(in)::x(:,:)
  type(legacy_random_state)::rng
  integer,allocatable::rows(:)
  integer::n,m,j,i
  call controls(self);call quantile_check(x);n=size(x,1);m=n
  call rng%seed(self%random_state)
  if(self%subsample>0) m=min(n,self%subsample)
  if(m<n) then
   rows=permutation(n,rng)
  else
   rows=[(i,i=1,n)]
  end if
  call initialize(self,n,size(x,2))
  do j=1,size(x,2)
   call percentiles(self,j,x(rows(:m),j),0,.false.)
  end do
 end subroutine
 function transposed(x) result(z)
  type(csr_matrix),intent(in)::x
  type(csr_matrix)::z
  integer,allocatable::counts(:),next(:)
  integer::i,j,k,slot
  z%n_rows=x%n_cols;z%n_cols=x%n_rows
  call require(z%n_rows<huge(1),'QuantileTransformer sparse transpose capacity overflow')
  allocate(counts(z%n_rows),next(z%n_rows),z%indptr(z%n_rows+1),z%indices(size(x%data)),z%data(size(x%data)))
  counts=0
  do k=1,size(x%data)
   counts(x%indices(k))=counts(x%indices(k))+1
  end do
  z%indptr(1)=1
  do j=1,z%n_rows
   z%indptr(j+1)=z%indptr(j)+counts(j)
  end do
  next=z%indptr(:z%n_rows)
  do i=1,x%n_rows
   do k=x%indptr(i),x%indptr(i+1)-1
    j=x%indices(k);slot=next(j);next(j)=slot+1
    z%indices(slot)=i;z%data(slot)=x%data(k)
   end do
  end do
 end function
 subroutine fit_sparse_columns(self,columns)
  class(quantile_transformer),intent(inout)::self
  type(csr_matrix),intent(in)::columns ! Each CSR row represents one CSC column.
  type(legacy_random_state)::rng
  integer,allocatable::order(:)
  real(dp),allocatable::values(:)
  integer::j,first,last,d,k,length,zeros
  integer(int64)::sample_count
  call rng%seed(self%random_state);call initialize(self,columns%n_cols,columns%n_rows)
  do j=1,columns%n_rows
   first=columns%indptr(j);last=columns%indptr(j+1)-1;d=last-first+1
   if(self%subsample>0.and.d>self%subsample) then
    sample_count=int(self%subsample,int64)*int(d,int64)/int(columns%n_cols,int64)
    call require(sample_count<=int(huge(1),int64),'QuantileTransformer sparse sample capacity overflow')
    k=int(sample_count);length=self%subsample
    call require(k<=d,'QuantileTransformer sparse subsample exceeds stored population')
    if(self%ignore_implicit_zeros) length=k
    call require(k<=length,'QuantileTransformer stored column exceeds padded sample size')
    order=permutation(d,rng);values=columns%data(first+order(:k)-1)
   else
    k=d;length=columns%n_cols
    if(self%ignore_implicit_zeros) length=k
    call require(k<=length,'QuantileTransformer stored column exceeds padded sample size')
    values=columns%data(first:last)
   end if
   zeros=length-k;call percentiles(self,j,values,zeros,length==0)
  end do
 end subroutine
 subroutine fit_csr(self,x)
  class(quantile_transformer),intent(inout)::self
  type(csr_matrix),intent(in)::x
  type(csr_matrix)::columns
  call controls(self);call sparse_check(self,x,.false.);columns=transposed(x)
  call fit_sparse_columns(self,columns)
 end subroutine
 logical function less(a,b) result(value)
  real(dp),intent(in)::a,b
  value=.false.
  if(ieee_is_nan(a).or.ieee_is_nan(b)) return
  value=a<b
 end function
 logical function equal(a,b) result(value)
  real(dp),intent(in)::a,b
  value=.false.
  if(ieee_is_nan(a).or.ieee_is_nan(b)) return
  value=a==b
 end function
 ! NumPy 2.2.6 compiled_base.c binary_search_with_guess / arr_interp.
 integer function search_with_guess(key,a,guess) result(found)
  real(dp),intent(in)::key,a(0:)
  integer,intent(in)::guess
  integer::n,lo,hi,g,i,mid
  n=size(a);lo=0;hi=n
  if(less(a(n-1),key)) then
   found=n;return
  else if(less(key,a(0))) then
   found=-1;return
  end if
  if(n<=4) then
   i=1
   do while(i<n)
    if(.not.(less(a(i),key).or.equal(a(i),key))) exit
    i=i+1
   end do
   found=i-1;return
  end if
  g=max(1,min(guess,n-3))
  if(less(key,a(g))) then
   if(less(key,a(g-1))) then
    hi=g-1
    if(g>8) then
     if(less(a(g-8),key).or.equal(a(g-8),key)) lo=g-8
    end if
   else
    found=g-1;return
   end if
  else
   if(less(key,a(g+1))) then
    found=g;return
   else if(less(key,a(g+2))) then
    found=g+1;return
   else
    lo=g+2
    if(g<n-9) then
     if(less(key,a(g+8))) hi=g+8
    end if
   end if
  end if
  do while(lo<hi)
   mid=lo+(hi-lo)/2
   if(less(a(mid),key).or.equal(a(mid),key)) then
    lo=mid+1
   else
    hi=mid
   end if
  end do
  found=lo-1
 end function
 real(dp) function interpolate(x,xp,yp,guess) result(y)
  real(dp),intent(in)::x,xp(:),yp(:)
  integer,intent(inout)::guess
  integer::n,j
  real(dp)::slope
  n=size(xp)
  if(n==1) then
   y=yp(1);return
  end if
  guess=search_with_guess(x,xp,guess)
  if(guess==-1) then
   y=yp(1);return
  else if(guess==n.or.guess==n-1) then
   y=yp(n);return
  end if
  j=guess+1;y=yp(j)
  if(equal(x,xp(j))) return
  slope=ieee_quotient(ieee_difference(yp(j+1),yp(j)),ieee_difference(xp(j+1),xp(j)))
  y=ieee_difference(ieee_product(slope,ieee_difference(x,xp(j))),-yp(j))
  if(ieee_is_nan(y)) then
   y=ieee_difference(ieee_product(slope,ieee_difference(x,xp(j+1))),-yp(j+1))
   if(ieee_is_nan(y).and.equal(yp(j),yp(j+1))) y=yp(j)
  end if
 end function
 real(dp) function normal_cdf(x) result(p)
  real(dp),intent(in)::x
  p=.5_dp*erfc(-x/sqrt(2._dp))
 end function
 real(dp) function normal_quantile(p) result(x)
  real(dp),intent(in)::p
  real(dp)::tail,lo,hi,mid
  integer::i
  x=ieee_value(0._dp,ieee_quiet_nan)
  if(ieee_is_nan(p)) return
  tail=min(p,1-p);lo=-40;hi=0
  do i=1,80
   mid=.5_dp*(lo+hi)
   if(normal_cdf(mid)<tail) then
    lo=mid
   else
    hi=mid
   end if
  end do
  x=.5_dp*(lo+hi)
  if(p>.5_dp) x=-x
  if(p==.5_dp) x=0
 end function
 subroutine ready(self,p)
  class(quantile_transformer),intent(in)::self
  integer,intent(in)::p
  call require(allocated(self%quantiles_),'QuantileTransformer is not fitted')
  call require(p==self%n_features_in_,'QuantileTransformer feature count mismatch')
  call distribution_check(self)
 end subroutine
 real(dp) function mapped(self,x,j,inverse,ga,gb) result(y)
  class(quantile_transformer),intent(in)::self
  real(dp),intent(in)::x
  integer,intent(in)::j
  integer,intent(inout)::ga,gb
  logical,intent(in)::inverse
  real(dp)::v,lower,upper,low_y,high_y,clip_p
  logical::low,high,normal
  integer::n
  y=x;if(ieee_is_nan(x)) return
  n=size(self%quantiles_,1);normal=self%output_distribution=='normal'
  lower=self%quantiles_(1,j);upper=self%quantiles_(n,j);low_y=0;high_y=1;v=x
  if(inverse) then
   lower=0;upper=1;low_y=self%quantiles_(1,j);high_y=self%quantiles_(n,j)
   if(normal) v=normal_cdf(v)
  end if
  if(normal) then
   low=less(ieee_difference(v,bounds_threshold),lower)
   high=less(upper,ieee_difference(v,-bounds_threshold))
  else
   low=equal(v,lower);high=equal(v,upper)
  end if
  if(inverse) then
   y=interpolate(v,self%references_,self%quantiles_(:,j),ga)
  else
   y=.5_dp*ieee_difference(interpolate(v,self%quantiles_(:,j),self%references_,ga), &
    interpolate(-v,-self%quantiles_(n:1:-1,j),-self%references_(n:1:-1),gb))
  end if
  if(high) y=high_y
  if(low) y=low_y
  if(.not.inverse.and.normal.and..not.ieee_is_nan(y)) then
   clip_p=bounds_threshold-epsilon(1._dp)
   if(y<0.or.y>1) then
    y=ieee_value(0._dp,ieee_quiet_nan)
   else
    y=normal_quantile(max(clip_p,min(1-clip_p,y)))
   end if
  end if
 end function
 function apply_dense(self,x,inverse) result(z)
  class(quantile_transformer),intent(in)::self
  real(dp),intent(in)::x(:,:)
  logical,intent(in)::inverse
  real(dp),allocatable::z(:,:)
  integer::i,j,ga,gb
  call ready(self,size(x,2));call quantile_check(x);z=x
  do j=1,size(x,2)
   ga=0;gb=0
   do i=1,size(x,1)
    z(i,j)=mapped(self,x(i,j),j,inverse,ga,gb)
   end do
  end do
 end function
 function apply_csr(self,x,inverse) result(z)
  class(quantile_transformer),intent(in)::self
  type(csr_matrix),intent(in)::x
  logical,intent(in)::inverse
  type(csr_matrix)::z
  integer::k
  integer,allocatable::ga(:),gb(:)
  call ready(self,x%n_cols);call sparse_check(self,x,inverse);z=x
  allocate(ga(x%n_cols),gb(x%n_cols));ga=0;gb=0
  do k=1,size(z%data)
   z%data(k)=mapped(self,x%data(k),x%indices(k),inverse,ga(x%indices(k)),gb(x%indices(k)))
  end do
 end function
 function transform_dense(self,x) result(z)
  class(quantile_transformer),intent(in)::self
  real(dp),intent(in)::x(:,:)
  real(dp),allocatable::z(:,:)
  z=apply_dense(self,x,.false.)
 end function
 function inverse_dense(self,x) result(z)
  class(quantile_transformer),intent(in)::self
  real(dp),intent(in)::x(:,:)
  real(dp),allocatable::z(:,:)
  z=apply_dense(self,x,.true.)
 end function
 function transform_csr(self,x) result(z)
  class(quantile_transformer),intent(in)::self
  type(csr_matrix),intent(in)::x
  type(csr_matrix)::z
  z=apply_csr(self,x,.false.)
 end function
 function inverse_csr(self,x) result(z)
  class(quantile_transformer),intent(in)::self
  type(csr_matrix),intent(in)::x
  type(csr_matrix)::z
  z=apply_csr(self,x,.true.)
 end function
 function fit_transform_dense(self,x) result(z)
  class(quantile_transformer),intent(inout)::self
  real(dp),intent(in)::x(:,:)
  real(dp),allocatable::z(:,:)
  call self%fit(x);z=self%transform(x)
 end function
 function fit_transform_csr(self,x) result(z)
  class(quantile_transformer),intent(inout)::self
  type(csr_matrix),intent(in)::x
  type(csr_matrix)::z
  call self%fit(x);z=self%transform(x)
 end function
 function free_dense(x,axis,n_quantiles,output_distribution,ignore_implicit_zeros,subsample,random_state) result(z)
  real(dp),intent(in)::x(:,:)
  integer,intent(in),optional::axis,n_quantiles,subsample,random_state
  character(*),intent(in),optional::output_distribution
  logical,intent(in),optional::ignore_implicit_zeros
  real(dp),allocatable::z(:,:),temp(:,:)
  type(quantile_transformer)::model
  integer::ax
  call free_controls(model,axis,n_quantiles,output_distribution,ignore_implicit_zeros,subsample,random_state,ax)
  if(ax==0) then
   z=model%fit_transform(x)
  else
   temp=model%fit_transform(transpose(x));z=transpose(temp)
  end if
 end function
 function free_csr(x,axis,n_quantiles,output_distribution,ignore_implicit_zeros,subsample,random_state) result(z)
  type(csr_matrix),intent(in)::x
  integer,intent(in),optional::axis,n_quantiles,subsample,random_state
  character(*),intent(in),optional::output_distribution
  logical,intent(in),optional::ignore_implicit_zeros
  type(csr_matrix)::z
  type(quantile_transformer)::model
  integer::ax,i,k,ga,gb
  call free_controls(model,axis,n_quantiles,output_distribution,ignore_implicit_zeros,subsample,random_state,ax)
  if(ax==0) then
   z=model%fit_transform(x)
  else
   call controls(model);call sparse_check(model,x,.false.);call fit_sparse_columns(model,x);z=x
   do i=1,x%n_rows
    ga=0;gb=0
    do k=x%indptr(i),x%indptr(i+1)-1
     z%data(k)=mapped(model,x%data(k),i,.false.,ga,gb)
    end do
   end do
  end if
 end function
 subroutine free_controls(model,axis,n_quantiles,output_distribution,ignore_implicit_zeros,subsample,random_state,ax)
  type(quantile_transformer),intent(out)::model
  integer,intent(in),optional::axis,n_quantiles,subsample,random_state
  character(*),intent(in),optional::output_distribution
  logical,intent(in),optional::ignore_implicit_zeros
  integer,intent(out)::ax
  ax=0;if(present(axis)) ax=axis
  call require(ax==0.or.ax==1,'quantile_transform axis must be 0 or 1')
  model%subsample=100000
  if(present(n_quantiles)) model%n_quantiles=n_quantiles
  if(present(output_distribution)) then
   call require(output_distribution=='uniform'.or.output_distribution=='normal', &
    'invalid QuantileTransformer output_distribution')
   model%output_distribution=output_distribution
  end if
  if(present(ignore_implicit_zeros)) model%ignore_implicit_zeros=ignore_implicit_zeros
  if(present(subsample)) model%subsample=subsample
  if(present(random_state)) model%random_state=random_state
 end subroutine
end module
