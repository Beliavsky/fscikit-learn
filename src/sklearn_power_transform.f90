! sklearn 1.6.1 PowerTransformer/power_transform; SciPy 1.15.3 Brent and Box-Cox.
! Retained sklearn/NumPy/SciPy BSD notices: LICENSES/ and THIRD_PARTY_NOTICES.md.
module sklearn_power_transform
 use sklearn_common,only:dp,require
 use sklearn_standard_scaling,only:standard_scaler
 use sklearn_ieee_math,only:sub=>ieee_difference,mul=>ieee_product,div=>ieee_quotient
 use,intrinsic::iso_c_binding,only:c_double
 use,intrinsic::iso_fortran_env,only:int64
 use,intrinsic::ieee_arithmetic
 use,intrinsic::ieee_exceptions
 implicit none
 private
 public::power_transformer,power_transform
 real(dp),parameter::qnan=transfer(int(z'7FF8000000000000',int64),0._dp)
 type::power_transformer
  character(16)::method='yeo-johnson'
  logical::standardize=.true.
  real(dp),allocatable::lambdas_(:),mean_(:),scale_(:)
  integer::n_features_in_=0
  type(standard_scaler),allocatable,private::scaler_
 contains
  procedure::fit=>power_fit
  procedure::transform=>power_forward
  procedure::fit_transform=>power_fit_transform
  procedure::inverse_transform=>power_inverse
 end type
 interface power_transform
  module procedure free_power
 end interface
 interface
  function c_pow(x,y) bind(C,name='pow') result(v)
   import c_double
   real(c_double),value::x,y
   real(c_double)::v
  end function
  function c_exp(x) bind(C,name='exp') result(v)
   import c_double
   real(c_double),value::x
   real(c_double)::v
  end function
  function c_expm1(x) bind(C,name='expm1') result(v)
   import c_double
   real(c_double),value::x
   real(c_double)::v
  end function
  function c_log(x) bind(C,name='log') result(v)
   import c_double
   real(c_double),value::x
   real(c_double)::v
  end function
  function c_log1p(x) bind(C,name='log1p') result(v)
   import c_double
   real(c_double),value::x
   real(c_double)::v
  end function
 end interface
contains
 real(dp) function quiet_math(op,x,y) result(v)
  integer,intent(in)::op
  real(dp),intent(in)::x
  real(dp),optional,intent(in)::y
  type(ieee_flag_type)::flags(3)
  logical::halt(3),raised(3)
  integer::i
  ! Match NumPy/SciPy nonfinite math under checked Fortran FPE traps. Restore
  ! incoming exception flags and halting modes, rather than changing caller policy.
  flags=[ieee_invalid,ieee_divide_by_zero,ieee_overflow]
  do i=1,3
   call ieee_get_halting_mode(flags(i),halt(i));call ieee_get_flag(flags(i),raised(i))
  end do
  do i=1,3
   call ieee_set_halting_mode(flags(i),.false.)
  end do
  select case(op)
  case(1);v=c_pow(x,y)
  case(2);v=c_exp(x)
  case(3);v=c_expm1(x)
  case(4);v=c_log(x)
  case(5);v=c_log1p(x)
  end select
  do i=1,3
   call ieee_set_halting_mode(flags(i),halt(i))
  end do
  do i=1,3
   call ieee_set_flag(flags(i),raised(i))
  end do
 end function
 real(dp) function add(a,b) result(v)
  real(dp),intent(in)::a,b
  v=sub(a,-b)
 end function
 logical function lt(a,b) result(v)
  real(dp),intent(in)::a,b
  v=.false.;if(ieee_is_nan(a).or.ieee_is_nan(b)) return
  v=a<b
 end function
 logical function le(a,b) result(v)
  real(dp),intent(in)::a,b
  v=.false.;if(ieee_is_nan(a).or.ieee_is_nan(b)) return
  v=a<=b
 end function
 recursive real(dp) function total(a) result(v)
  real(dp),intent(in)::a(:)
  real(dp)::r(8)
  integer::i,j,n,last,split
  n=size(a);v=-0._dp
  if(n<8) then
   do i=1,n
    v=add(v,a(i))
   end do
  else if(n<=128) then
   r=a(:8);last=n-modulo(n,8)
   do i=9,last,8
    do j=1,8
     r(j)=add(r(j),a(i+j-1))
    end do
   end do
   v=add(add(add(r(1),r(2)),add(r(3),r(4))),add(add(r(5),r(6)),add(r(7),r(8))))
   do i=last+1,n
    v=add(v,a(i))
   end do
  else
   split=n/2;split=split-modulo(split,8);v=add(total(a(:split)),total(a(split+1:)))
  end if
 end function
 real(dp) function column_total(a,pairwise) result(v)
  real(dp),intent(in)::a(:)
  logical,intent(in)::pairwise
  integer::i
  v=0
  if(pairwise) then
   v=total(a)
  else
   do i=1,size(a)
    v=add(v,a(i))
   end do
  end if
 end function
 real(dp) function variance(a,pairwise) result(v)
  real(dp),intent(in)::a(:)
  logical,intent(in)::pairwise
  real(dp)::terms(size(a)),mean
  integer::i
  mean=div(column_total(a,pairwise),real(size(a),dp))
  do i=1,size(a)
   terms(i)=mul(sub(a(i),mean),sub(a(i),mean))
  end do
  v=div(column_total(terms,pairwise),real(size(a),dp))
 end function
 real(dp) function box_value(x,l,inverse) result(v)
  real(dp),intent(in)::x,l
  logical,intent(in)::inverse
  real(dp)::t,a
  v=qnan;if(ieee_is_nan(x).or.ieee_is_nan(l)) return
  if(inverse) then
   if(l==0) then
    v=quiet_math(2,x)
   else
    t=mul(l,x)
    if(lt(t,1.79e308_dp)) then
     v=quiet_math(2,div(quiet_math(5,t),l))
    else
     a=mul(sign(1._dp,l),add(x,div(1._dp,l)))
     v=quiet_math(2,div(add(quiet_math(4,a),quiet_math(4,abs(l))),l))
    end if
   end if
  else
   a=quiet_math(4,x)
   if(abs(l)<1.e-19_dp) then
    v=a
   else
    t=mul(l,a)
    if(lt(t,709.78_dp)) then
     v=div(quiet_math(3,t),l)
    else
     v=sub(mul(sign(1._dp,l),quiet_math(2,sub(t,quiet_math(4,abs(l))))),div(1._dp,l))
    end if
   end if
  end if
 end function
 real(dp) function yeo_value(x,l,inverse) result(v)
  real(dp),intent(in)::x,l
  logical,intent(in)::inverse
  real(dp)::base,exponent
  v=qnan
  if(le(0._dp,x)) then
   exponent=l
   if(inverse) then
    if(lt(abs(l),epsilon(1._dp))) then
     v=sub(quiet_math(2,x),1._dp)
    else
     v=sub(quiet_math(1,add(mul(x,l),1._dp),div(1._dp,l)),1._dp)
    end if
   else
    if(lt(abs(l),epsilon(1._dp))) then
     v=quiet_math(5,x)
    else
     v=div(sub(quiet_math(1,add(x,1._dp),l),1._dp),l)
    end if
   end if
  else
   exponent=sub(2._dp,l)
   if(inverse) then
    if(lt(epsilon(1._dp),abs(sub(l,2._dp)))) then
     base=add(-mul(exponent,x),1._dp);v=sub(1._dp,quiet_math(1,base,div(1._dp,exponent)))
    else
     v=sub(1._dp,quiet_math(2,-x))
    end if
   else
    if(lt(epsilon(1._dp),abs(sub(l,2._dp)))) then
     v=-div(sub(quiet_math(1,add(-x,1._dp),exponent),1._dp),exponent)
    else
     v=-quiet_math(5,-x)
    end if
   end if
  end if
 end function
 complex(dp) function complex_log1p(z) result(v)
  complex(dp),intent(in)::z
  real(dp)::x,y,r
  x=real(z,dp);y=aimag(z)
  if(abs(1+x)<.5_dp) then
   r=quiet_math(4,hypot(1+x,y))
  else
   r=.5_dp*quiet_math(5,2*x+x*x+y*y)
  end if
  v=cmplx(r,atan2(y,1+x),dp)
 end function
 real(dp) function log_variance(logx) result(value)
  real(dp),intent(in)::logx(:)
  real(dp)::amax,lmean,r(size(logx)),m,pi
  complex(dp)::mu(size(logx)),a,b,c,s
  integer::i,k
  value=qnan
  if(.not.all(ieee_is_finite(logx))) return
  amax=maxval(logx);m=real(count(logx==amax),dp)
  do i=1,size(logx)
   r(i)=0
   if(logx(i)/=amax) r(i)=quiet_math(2,logx(i)-amax)
  end do
  lmean=quiet_math(5,div(total(r),m))+quiet_math(4,m)+amax-quiet_math(4,real(size(logx),dp))
  pi=acos(-1._dp)
  do i=1,size(logx)
   a=cmplx(logx(i),0._dp,dp);b=cmplx(lmean,pi,dp)
   if(logx(i)>lmean) then
    c=a;a=b;b=c
   end if
   mu(i)=complex_log1p(exp(a-b))+b
  end do
  mu=2*mu;k=maxloc(real(mu,dp),dim=1,back=.true.);b=mu(k);s=cmplx(0._dp,0._dp,dp)
  ! Shift before complex exponentiation, as in scipy.special.logsumexp.
  do i=1,size(logx)
   if(i/=k) s=s+exp(mu(i)-b)
  end do
  value=real(complex_log1p(s)+b,dp)-quiet_math(4,real(size(logx),dp))
 end function
 real(dp) function likelihood(data,l,method) result(v)
  real(dp),intent(in)::data(:),l
  character(*),intent(in)::method
  real(dp)::z(size(data)),logs(size(data)),var,jac,logvar
  integer::i,n
  n=size(data);v=qnan
  do i=1,n
   if(method=='box-cox') then
    logs(i)=quiet_math(4,data(i));z(i)=mul(l,logs(i))
   else
    z(i)=yeo_value(data(i),l,.false.);logs(i)=mul(sign(1._dp,data(i)),quiet_math(5,abs(data(i))))
   end if
  end do
  jac=total(logs)
  if(method=='box-cox') then
   if(l==0) then
    logvar=quiet_math(4,variance(logs,.true.))
   else
    logvar=sub(log_variance(z),2*quiet_math(4,abs(l)))
   end if
  else
   var=variance(z,.true.)
   if(lt(var,tiny(1._dp))) then
    v=ieee_value(0._dp,ieee_positive_inf);return
   end if
   logvar=quiet_math(4,var)
  end if
  v=sub(mul(real(n,dp)/2,logvar),mul(sub(l,1._dp),jac))
 end function
 subroutine bracket(data,method,xa,xb,xc,fa,fb,fc)
  real(dp),intent(in)::data(:)
  character(*),intent(in)::method
  real(dp),intent(out)::xa,xb,xc,fa,fb,fc
  real(dp)::tmp1,tmp2,val,denom,w,wlim,fw,t
  integer::i
  xa=-2;xb=2;fa=likelihood(data,xa,method);fb=likelihood(data,xb,method)
  if(lt(fa,fb)) then
   t=xa;xa=xb;xb=t;t=fa;fa=fb;fb=t
  end if
  xc=add(xb,mul(1.618034_dp,sub(xb,xa)));fc=likelihood(data,xc,method);i=0
  do while(lt(fc,fb))
   tmp1=mul(sub(xb,xa),sub(fb,fc));tmp2=mul(sub(xb,xc),sub(fb,fa));val=sub(tmp2,tmp1)
   denom=2*1.e-21_dp
   if(.not.lt(abs(val),1.e-21_dp)) denom=mul(2._dp,val)
   w=sub(xb,div(sub(mul(sub(xb,xc),tmp2),mul(sub(xb,xa),tmp1)),denom))
   wlim=add(xb,mul(110._dp,sub(xc,xb)));call require(i<=1000,'PowerTransformer bracket iteration limit');i=i+1
   if(lt(0._dp,mul(sub(w,xc),sub(xb,w)))) then
    fw=likelihood(data,w,method)
    if(lt(fw,fc)) then
     xa=xb;xb=w;fa=fb;fb=fw;exit
    else if(lt(fb,fw)) then
     xc=w;fc=fw;exit
    end if
    w=add(xc,mul(1.618034_dp,sub(xc,xb)));fw=likelihood(data,w,method)
   else if(le(0._dp,mul(sub(w,wlim),sub(wlim,xc)))) then
    w=wlim;fw=likelihood(data,w,method)
   else if(lt(0._dp,mul(sub(w,wlim),sub(xc,w)))) then
    fw=likelihood(data,w,method)
    if(lt(fw,fc)) then
     xb=xc;xc=w;w=add(xc,mul(1.618034_dp,sub(xc,xb)));fb=fc;fc=fw;fw=likelihood(data,w,method)
    end if
   else
    w=add(xc,mul(1.618034_dp,sub(xc,xb)));fw=likelihood(data,w,method)
   end if
   xa=xb;xb=xc;xc=w;fa=fb;fb=fc;fc=fw
  end do
  call require(((lt(fb,fc).and.le(fb,fa)).or.(lt(fb,fa).and.le(fb,fc))).and. &
   ((lt(xa,xb).and.lt(xb,xc)).or.(lt(xc,xb).and.lt(xb,xa))).and. &
   ieee_is_finite(xa).and.ieee_is_finite(xb).and.ieee_is_finite(xc), &
   'PowerTransformer could not bracket likelihood optimum')
 end subroutine
 real(dp) function optimal_power(data,method) result(x)
  real(dp),intent(in)::data(:)
  character(*),intent(in)::method
  real(dp)::xa,xb,xc,fa,fb,fc,w,v,fx,fw,fv,a,b,deltax,tol1,tol2,xmid,rat,tmp1,tmp2,p,dx,u,fu
  real(dp)::extreme,bound,low,high,mid,val
  integer::i
  call bracket(data,method,xa,xb,xc,fa,fb,fc);x=xb;w=x;v=x;fx=fb;fw=fx;fv=fx
  a=min(xa,xc);b=max(xa,xc);deltax=0;rat=0
  do i=1,500
   tol1=1.48e-8_dp*abs(x)+1.e-11_dp;tol2=2*tol1;xmid=.5_dp*add(a,b)
   if(lt(abs(sub(x,xmid)),sub(tol2,.5_dp*sub(b,a)))) exit
   if(le(abs(deltax),tol1)) then
    if(x>=xmid) then
     deltax=sub(a,x)
    else
     deltax=sub(b,x)
    end if
    rat=mul(.3819660_dp,deltax)
   else
    tmp1=mul(sub(x,w),sub(fx,fv));tmp2=mul(sub(x,v),sub(fx,fw))
    p=sub(mul(sub(x,v),tmp2),mul(sub(x,w),tmp1));tmp2=mul(2._dp,sub(tmp2,tmp1))
    if(lt(0._dp,tmp2)) p=-p
    tmp2=abs(tmp2);dx=deltax;deltax=rat
    if(lt(mul(tmp2,sub(a,x)),p).and.lt(p,mul(tmp2,sub(b,x))).and. &
     lt(abs(p),abs(mul(mul(.5_dp,tmp2),dx)))) then
     rat=div(p,tmp2);u=add(x,rat)
     if(lt(sub(u,a),tol2).or.lt(sub(b,u),tol2)) then
      rat=tol1;if(xmid-x<0) rat=-tol1
     end if
    else
     if(x>=xmid) then
      deltax=sub(a,x)
     else
      deltax=sub(b,x)
     end if
     rat=mul(.3819660_dp,deltax)
    end if
   end if
   if(lt(abs(rat),tol1)) then
    u=add(x,sign(tol1,rat))
   else
    u=add(x,rat)
   end if
   fu=likelihood(data,u,method)
   if(lt(fx,fu)) then
    if(u<x) then
     a=u
    else
     b=u
    end if
    if(le(fu,fw).or.w==x) then
     v=w;w=u;fv=fw;fw=fu
    else if(le(fu,fv).or.v==x.or.v==w) then
     v=u;fv=fu
    end if
   else
    if(u>=x) then
     a=x
    else
     b=x
    end if
    v=w;w=x;x=u;fv=fw;fw=fx;fx=fu
   end if
  end do
  if(method=='box-cox'.and.ieee_is_finite(x)) then
   extreme=maxval(data)
   if(maxval(data)<=1) then
    extreme=minval(data)
   else if(minval(data)<1) then
    if(.not.lt(abs(box_value(minval(data),x,.false.)),box_value(maxval(data),x,.false.))) extreme=minval(data)
   end if
   bound=huge(1._dp)/10000
   if(lt(bound,abs(box_value(extreme,x,.false.)))) then
    low=0;high=x
    do i=1,160
     mid=.5_dp*(low+high);val=abs(box_value(extreme,mid,.false.))
     if(lt(bound,val)) then
      high=mid
     else
      low=mid
     end if
    end do
    x=.5_dp*(low+high)
   end if
  end if
 end function
 subroutine validate(x)
  real(dp),intent(in)::x(:,:)
  call require(size(x,1)>0.and.size(x,2)>0,'PowerTransformer needs a nonempty matrix')
  call require(all(ieee_is_finite(x).or.ieee_is_nan(x)),'PowerTransformer rejects infinity')
 end subroutine
 subroutine method_check(self)
  class(power_transformer),intent(in)::self
  call require(self%method=='yeo-johnson'.or.self%method=='box-cox','invalid PowerTransformer method')
 end subroutine
 subroutine positive_check(self,x)
  class(power_transformer),intent(in)::self
  real(dp),intent(in)::x(:,:)
  integer::i,j
  if(self%method/='box-cox') return
  do j=1,size(x,2)
   do i=1,size(x,1)
    if(ieee_is_nan(x(i,j))) cycle
    call require(x(i,j)>0,'Box-Cox requires strictly positive data')
   end do
  end do
 end subroutine
 function fit_core(self,x,force) result(z)
  class(power_transformer),intent(inout)::self
  real(dp),intent(in)::x(:,:)
  logical,intent(in)::force
  real(dp),allocatable::z(:,:),observed(:)
  real(dp)::mean,var,upper,n
  logical::constant
  integer::i,j
  call method_check(self);call validate(x);call positive_check(self,x);z=x
  self%n_features_in_=size(x,2)
  if(allocated(self%lambdas_)) deallocate(self%lambdas_)
  allocate(self%lambdas_(size(x,2)))
  do j=1,size(x,2)
   n=real(size(x,1),dp);mean=div(column_total(x(:,j),size(x,2)==1),n);var=variance(x(:,j),size(x,2)==1)
   upper=add(mul(mul(n,epsilon(1._dp)),var),mul(mul(mul(n,mean),epsilon(1._dp)),mul(mul(n,mean),epsilon(1._dp))))
   constant=le(var,upper)
   if(self%method=='yeo-johnson'.and.constant) then
    self%lambdas_(j)=1;cycle
   end if
   observed=pack(x(:,j),.not.ieee_is_nan(x(:,j)))
   call require(size(observed)>0,'PowerTransformer all-missing feature')
   if(self%method=='box-cox') call require(maxval(observed)>minval(observed),'Box-Cox requires nonconstant data')
   self%lambdas_(j)=optimal_power(observed,self%method)
   if(self%standardize.or.force) then
    do i=1,size(x,1)
     if(self%method=='box-cox') then
      z(i,j)=box_value(x(i,j),self%lambdas_(j),.false.)
     else
      z(i,j)=yeo_value(x(i,j),self%lambdas_(j),.false.)
     end if
    end do
   end if
  end do
  if(self%standardize) then
   if(.not.allocated(self%scaler_)) allocate(self%scaler_)
   call self%scaler_%fit(z);self%mean_=self%scaler_%mean_;self%scale_=self%scaler_%scale_
   if(force) z=self%scaler_%transform(z)
  end if
 end function
 subroutine power_fit(self,x)
  class(power_transformer),intent(inout)::self
  real(dp),intent(in)::x(:,:)
  real(dp),allocatable::unused(:,:)
  unused=fit_core(self,x,.false.)
 end subroutine
 function power_fit_transform(self,x) result(z)
  class(power_transformer),intent(inout)::self
  real(dp),intent(in)::x(:,:)
  real(dp),allocatable::z(:,:)
  z=fit_core(self,x,.true.)
 end function
 function apply(self,x,inverse) result(z)
  class(power_transformer),intent(in)::self
  real(dp),intent(in)::x(:,:)
  logical,intent(in)::inverse
  real(dp),allocatable::z(:,:),input(:,:)
  integer::i,j
  call require(allocated(self%lambdas_),'PowerTransformer is not fitted')
  call validate(x);call require(size(x,2)==size(self%lambdas_),'PowerTransformer feature count mismatch')
  call method_check(self);input=x
  if(inverse.and.self%standardize) then
   call require(allocated(self%scaler_),'PowerTransformer fitted scaler is absent')
   input=self%scaler_%inverse_transform(input)
  end if
  if(.not.inverse) call positive_check(self,input)
  z=input
  do j=1,size(x,2)
   do i=1,size(x,1)
    if(self%method=='box-cox') then
     z(i,j)=box_value(input(i,j),self%lambdas_(j),inverse)
    else
     z(i,j)=yeo_value(input(i,j),self%lambdas_(j),inverse)
    end if
   end do
  end do
  if(.not.inverse.and.self%standardize) then
   call require(allocated(self%scaler_),'PowerTransformer fitted scaler is absent')
   z=self%scaler_%transform(z)
  end if
 end function
 function power_forward(self,x) result(z)
  class(power_transformer),intent(in)::self
  real(dp),intent(in)::x(:,:)
  real(dp),allocatable::z(:,:)
  z=apply(self,x,.false.)
 end function
 function power_inverse(self,x) result(z)
  class(power_transformer),intent(in)::self
  real(dp),intent(in)::x(:,:)
  real(dp),allocatable::z(:,:)
  z=apply(self,x,.true.)
 end function
 function free_power(x,method,standardize) result(z)
  real(dp),intent(in)::x(:,:)
  character(*),intent(in),optional::method
  logical,intent(in),optional::standardize
  real(dp),allocatable::z(:,:)
  type(power_transformer)::model
  if(present(method)) then
   call require(method=='yeo-johnson'.or.method=='box-cox','invalid PowerTransformer method');model%method=method
  end if
  if(present(standardize)) model%standardize=standardize
  z=model%fit_transform(x)
 end function
end module
