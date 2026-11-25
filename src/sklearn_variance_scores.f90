! R2/explained variance translated from scikit-learn 1.6.1 (BSD-3-Clause).
! NumPy 2.2.6 pairwise reduction (BSD-3-Clause); retained licenses in LICENSES/.
! Typed adapters and IEEE-state handling are original MIT integration.
module sklearn_variance_scores
 use sklearn_common,only:dp,require
 use,intrinsic::ieee_arithmetic
 use,intrinsic::ieee_exceptions
 implicit none
 private
 public::r2_score,explained_variance_score,r2_score_raw,explained_variance_score_raw
 interface r2_score
  module procedure r2_score_11
  module procedure r2_score_weights_11
  module procedure r2_score_12
  module procedure r2_score_weights_12
  module procedure r2_score_21
  module procedure r2_score_weights_21
  module procedure r2_score_22
  module procedure r2_score_weights_22
 end interface
 interface r2_score_raw
  module procedure r2_score_raw_11
  module procedure r2_score_raw_12
  module procedure r2_score_raw_21
  module procedure r2_score_raw_22
 end interface
 interface explained_variance_score
  module procedure explained_variance_score_11
  module procedure explained_variance_score_weights_11
  module procedure explained_variance_score_12
  module procedure explained_variance_score_weights_12
  module procedure explained_variance_score_21
  module procedure explained_variance_score_weights_21
  module procedure explained_variance_score_22
  module procedure explained_variance_score_weights_22
 end interface
 interface explained_variance_score_raw
  module procedure explained_variance_score_raw_11
  module procedure explained_variance_score_raw_12
  module procedure explained_variance_score_raw_21
  module procedure explained_variance_score_raw_22
 end interface
contains
 recursive real(dp) function pairwise(a) result(total)
  real(dp),intent(in)::a(:)
  real(dp)::r(8)
  integer::n,i,j,whole,split
  n=size(a);total=-0._dp
  if(n<8) then
   do i=1,n
    total=subtract(total,-a(i))
   end do
  else if(n<=128) then
   r=a(:8);whole=n-mod(n,8)
   do i=9,whole,8
    do j=1,8
     r(j)=subtract(r(j),-a(i+j-1))
    end do
   end do
   total=subtract(subtract(subtract(r(1),-r(2)),-subtract(r(3),-r(4))),-subtract(subtract(r(5),-r(6)),-subtract(r(7),-r(8))))
   do i=whole+1,n
    total=subtract(total,-a(i))
   end do
  else
   split=n/2;split=split-mod(split,8);total=subtract(pairwise(a(:split)),-pairwise(a(split+1:)))
  end if
 end function
 real(dp) function summed(a) result(value)
  real(dp),intent(in)::a(:)
  integer::first,last
  value=0
  do first=1,size(a),8192
   last=min(size(a),first+8191);value=subtract(value,-pairwise(a(first:last)))
  end do
 end function
 real(dp) function subtract(a,b) result(c)
  real(dp),intent(in)::a,b
  real(dp),volatile::x,y
  x=a;y=b;c=x-y
 end function
 function matrix(a) result(b)
  real(dp),intent(in)::a(..)
  real(dp),allocatable::b(:,:)
  select rank(a)
  rank(1)
   b=reshape(a,[size(a),1])
  rank(2)
   b=a
  end select
 end function
 function columns(a) result(v)
  real(dp),intent(in)::a(:,:)
  real(dp)::v(size(a,2))
  integer::i,j
  v=0
  if(size(a,2)==1) then
   v(1)=summed(a(:,1))
  else
   ! Match the slow-axis reduction of the pinned C-contiguous NumPy oracle.
   do i=1,size(a,1)
    do j=1,size(a,2)
     v(j)=subtract(v(j),-a(i,j))
    end do
   end do
  end if
 end function
 subroutine suspend(halting,raised)
  logical,intent(out)::halting(size(ieee_all)),raised(size(ieee_all))
  integer::i
  do i=1,size(ieee_all)
   call ieee_get_flag(ieee_all(i),raised(i))
  end do
  do i=1,size(ieee_all)
   call ieee_get_halting_mode(ieee_all(i),halting(i))
  end do
  do i=1,size(ieee_all)
   call ieee_set_halting_mode(ieee_all(i),.false.)
  end do
 end subroutine
 subroutine restore(halting,raised)
  logical,intent(in)::halting(size(ieee_all)),raised(size(ieee_all))
  integer::i
  do i=1,size(ieee_all)
   call ieee_set_halting_mode(ieee_all(i),halting(i))
  end do
  do i=1,size(ieee_all)
   call ieee_set_flag(ieee_all(i),raised(i))
  end do
 end subroutine
 function evaluate(y,p,ev,sample_weight,multioutput,force_finite,output_weights,raw) result(value)
  real(dp),intent(in)::y(:,:),p(:,:)
  logical,intent(in)::ev
  real(dp),optional,intent(in)::sample_weight(:),output_weights(:)
  character(*),optional,intent(in)::multioutput
  logical,optional,intent(in)::force_finite,raw
  real(dp),allocatable::value(:),a(:,:),d(:,:),w(:),num(:),den(:),avg(:),scores(:),ow(:)
  real(dp)::mass,total
  logical::finite,halting(size(ieee_all)),raised(size(ieee_all)),want_raw
  integer::n,q,j
  character(:),allocatable::mode
  n=size(y,1);q=size(y,2)
  call require(n>0.and.q>0,'variance scores require nonempty targets')
  call require(all(shape(y)==shape(p)),'variance score target shapes differ')
  call require(all(ieee_is_finite(y)).and.all(ieee_is_finite(p)),'variance score targets must be finite')
  if(present(sample_weight)) call require(size(sample_weight)==n,'variance score sample weight length')
  mode='uniform_average'
  if(present(multioutput)) mode=multioutput
  call require(mode=='uniform_average'.or.mode=='variance_weighted','invalid variance multioutput mode; use *_raw')
  if(present(output_weights)) then
   call require(q>1,'custom output weights require multiple outputs')
   call require(size(output_weights)==q,'variance score output weight length')
   call require(all(ieee_is_finite(output_weights)),'variance score output weights must be finite')
  end if
  finite=.true.;if(present(force_finite)) finite=force_finite
  want_raw=.false.;if(present(raw)) want_raw=raw
  ! sklearn returns scalar NaN before any reduction when R2 has one sample.
  if(.not.ev.and.n<2) then
   value=[ieee_value(0._dp,ieee_quiet_nan)];return
  end if
  call suspend(halting,raised)
  allocate(w(n));w=1
  if(present(sample_weight)) w=sample_weight
  mass=real(n,dp)
  if(present(sample_weight)) mass=summed(w)
  call require(mass/=0,'variance score weights sum to zero')
  a=y
  if(present(sample_weight)) then
   do j=1,q
    a(:,j)=a(:,j)*w
   end do
  end if
  avg=columns(a)/mass
  a=y
  do j=1,q
   a(:,j)=a(:,j)-avg(j)
  end do
  a=a*a
  if(present(sample_weight)) then
   do j=1,q
    a(:,j)=a(:,j)*w
   end do
  end if
  den=columns(a)
  d=y-p
  if(ev) then
   a=d
   if(present(sample_weight)) then
    do j=1,q
     a(:,j)=a(:,j)*w
    end do
   end if
   avg=columns(a)/mass
   do j=1,q
    d(:,j)=d(:,j)-avg(j)
   end do
  end if
  d=d*d
  if(present(sample_weight)) then
   do j=1,q
    d(:,j)=d(:,j)*w
   end do
  end if
  num=columns(d)
  if(ev) then
   num=num/mass;den=den/mass
  end if
  allocate(scores(q));scores=1
  do j=1,q
   if(.not.finite) then
    scores(j)=1-num(j)/den(j)
   else if(num(j)/=0) then
    if(den(j)/=0) then
     scores(j)=1-num(j)/den(j)
    else
     scores(j)=0
    end if
   end if
  end do
  if(want_raw) then
   value=scores
  else
   allocate(ow(q));ow=1
   if(mode=='variance_weighted'.and.any(den/=0)) ow=den
   if(present(output_weights)) ow=output_weights
   total=summed(ow)
   call require(total/=0,'variance score output weights sum to zero')
   if(present(output_weights).or.mode=='variance_weighted') then
    value=[summed(scores*ow)/total]
   else
    value=[summed(scores)/real(q,dp)]
   end if
  end if
  call restore(halting,raised)
 end function
 function r2_score_11(y,p,sample_weight,multioutput,force_finite,raw_values) result(v)
  real(dp),intent(in)::y(:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::force_finite
  character(*),optional,intent(in)::multioutput
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),.false.,sample_weight,force_finite=force_finite,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),.false.,sample_weight, &
   force_finite=force_finite,multioutput=multioutput,raw=.true.)
 end function
 function r2_score_weights_11(y,p,sample_weight,multioutput,force_finite,raw_values) result(v)
  real(dp),intent(in)::y(:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::force_finite
  real(dp),intent(in)::multioutput(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),.false.,sample_weight,force_finite=force_finite,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),.false.,sample_weight, &
   force_finite=force_finite,output_weights=multioutput,raw=.true.)
 end function
 function r2_score_raw_11(y,p,sample_weight,force_finite) result(v)
  real(dp),intent(in)::y(:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::force_finite
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(p),.false.,sample_weight,force_finite=force_finite,raw=.true.)
 end function
 function r2_score_12(y,p,sample_weight,multioutput,force_finite,raw_values) result(v)
  real(dp),intent(in)::y(:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::force_finite
  character(*),optional,intent(in)::multioutput
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),.false.,sample_weight,force_finite=force_finite,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),.false.,sample_weight, &
   force_finite=force_finite,multioutput=multioutput,raw=.true.)
 end function
 function r2_score_weights_12(y,p,sample_weight,multioutput,force_finite,raw_values) result(v)
  real(dp),intent(in)::y(:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::force_finite
  real(dp),intent(in)::multioutput(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),.false.,sample_weight,force_finite=force_finite,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),.false.,sample_weight, &
   force_finite=force_finite,output_weights=multioutput,raw=.true.)
 end function
 function r2_score_raw_12(y,p,sample_weight,force_finite) result(v)
  real(dp),intent(in)::y(:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::force_finite
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(p),.false.,sample_weight,force_finite=force_finite,raw=.true.)
 end function
 function r2_score_21(y,p,sample_weight,multioutput,force_finite,raw_values) result(v)
  real(dp),intent(in)::y(:,:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::force_finite
  character(*),optional,intent(in)::multioutput
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),.false.,sample_weight,force_finite=force_finite,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),.false.,sample_weight, &
   force_finite=force_finite,multioutput=multioutput,raw=.true.)
 end function
 function r2_score_weights_21(y,p,sample_weight,multioutput,force_finite,raw_values) result(v)
  real(dp),intent(in)::y(:,:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::force_finite
  real(dp),intent(in)::multioutput(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),.false.,sample_weight,force_finite=force_finite,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),.false.,sample_weight, &
   force_finite=force_finite,output_weights=multioutput,raw=.true.)
 end function
 function r2_score_raw_21(y,p,sample_weight,force_finite) result(v)
  real(dp),intent(in)::y(:,:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::force_finite
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(p),.false.,sample_weight,force_finite=force_finite,raw=.true.)
 end function
 function r2_score_22(y,p,sample_weight,multioutput,force_finite,raw_values) result(v)
  real(dp),intent(in)::y(:,:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::force_finite
  character(*),optional,intent(in)::multioutput
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),.false.,sample_weight,force_finite=force_finite,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),.false.,sample_weight, &
   force_finite=force_finite,multioutput=multioutput,raw=.true.)
 end function
 function r2_score_weights_22(y,p,sample_weight,multioutput,force_finite,raw_values) result(v)
  real(dp),intent(in)::y(:,:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::force_finite
  real(dp),intent(in)::multioutput(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),.false.,sample_weight,force_finite=force_finite,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),.false.,sample_weight, &
   force_finite=force_finite,output_weights=multioutput,raw=.true.)
 end function
 function r2_score_raw_22(y,p,sample_weight,force_finite) result(v)
  real(dp),intent(in)::y(:,:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::force_finite
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(p),.false.,sample_weight,force_finite=force_finite,raw=.true.)
 end function
 function explained_variance_score_11(y,p,sample_weight,multioutput,force_finite,raw_values) result(v)
  real(dp),intent(in)::y(:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::force_finite
  character(*),optional,intent(in)::multioutput
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),.true.,sample_weight,force_finite=force_finite,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),.true.,sample_weight, &
   force_finite=force_finite,multioutput=multioutput,raw=.true.)
 end function
 function explained_variance_score_weights_11(y,p,sample_weight,multioutput,force_finite,raw_values) result(v)
  real(dp),intent(in)::y(:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::force_finite
  real(dp),intent(in)::multioutput(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),.true.,sample_weight,force_finite=force_finite,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),.true.,sample_weight, &
   force_finite=force_finite,output_weights=multioutput,raw=.true.)
 end function
 function explained_variance_score_raw_11(y,p,sample_weight,force_finite) result(v)
  real(dp),intent(in)::y(:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::force_finite
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(p),.true.,sample_weight,force_finite=force_finite,raw=.true.)
 end function
 function explained_variance_score_12(y,p,sample_weight,multioutput,force_finite,raw_values) result(v)
  real(dp),intent(in)::y(:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::force_finite
  character(*),optional,intent(in)::multioutput
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),.true.,sample_weight,force_finite=force_finite,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),.true.,sample_weight, &
   force_finite=force_finite,multioutput=multioutput,raw=.true.)
 end function
 function explained_variance_score_weights_12(y,p,sample_weight,multioutput,force_finite,raw_values) result(v)
  real(dp),intent(in)::y(:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::force_finite
  real(dp),intent(in)::multioutput(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),.true.,sample_weight,force_finite=force_finite,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),.true.,sample_weight, &
   force_finite=force_finite,output_weights=multioutput,raw=.true.)
 end function
 function explained_variance_score_raw_12(y,p,sample_weight,force_finite) result(v)
  real(dp),intent(in)::y(:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::force_finite
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(p),.true.,sample_weight,force_finite=force_finite,raw=.true.)
 end function
 function explained_variance_score_21(y,p,sample_weight,multioutput,force_finite,raw_values) result(v)
  real(dp),intent(in)::y(:,:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::force_finite
  character(*),optional,intent(in)::multioutput
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),.true.,sample_weight,force_finite=force_finite,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),.true.,sample_weight, &
   force_finite=force_finite,multioutput=multioutput,raw=.true.)
 end function
 function explained_variance_score_weights_21(y,p,sample_weight,multioutput,force_finite,raw_values) result(v)
  real(dp),intent(in)::y(:,:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::force_finite
  real(dp),intent(in)::multioutput(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),.true.,sample_weight,force_finite=force_finite,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),.true.,sample_weight, &
   force_finite=force_finite,output_weights=multioutput,raw=.true.)
 end function
 function explained_variance_score_raw_21(y,p,sample_weight,force_finite) result(v)
  real(dp),intent(in)::y(:,:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::force_finite
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(p),.true.,sample_weight,force_finite=force_finite,raw=.true.)
 end function
 function explained_variance_score_22(y,p,sample_weight,multioutput,force_finite,raw_values) result(v)
  real(dp),intent(in)::y(:,:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::force_finite
  character(*),optional,intent(in)::multioutput
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),.true.,sample_weight,force_finite=force_finite,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),.true.,sample_weight, &
   force_finite=force_finite,multioutput=multioutput,raw=.true.)
 end function
 function explained_variance_score_weights_22(y,p,sample_weight,multioutput,force_finite,raw_values) result(v)
  real(dp),intent(in)::y(:,:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::force_finite
  real(dp),intent(in)::multioutput(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),.true.,sample_weight,force_finite=force_finite,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),.true.,sample_weight, &
   force_finite=force_finite,output_weights=multioutput,raw=.true.)
 end function
 function explained_variance_score_raw_22(y,p,sample_weight,force_finite) result(v)
  real(dp),intent(in)::y(:,:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::force_finite
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(p),.true.,sample_weight,force_finite=force_finite,raw=.true.)
 end function
end module
