! Adapted from sklearn 1.6.1 metrics/_regression.py (BSD-3-Clause).
! NumPy 2.2.6 buffered reductions retain the BSD license in LICENSES/.
! Explicit rank adapters and IEEE-state handling are original MIT integration.
module sklearn_error_metrics
 use sklearn_common,only:dp,require
 use,intrinsic::ieee_arithmetic
 use,intrinsic::ieee_exceptions
 implicit none
 private
 public::mean_squared_error
 interface mean_squared_error
  module procedure mean_squared_error_11
  module procedure mean_squared_error_weighted_11
  module procedure mean_squared_error_12
  module procedure mean_squared_error_weighted_12
  module procedure mean_squared_error_21
  module procedure mean_squared_error_weighted_21
  module procedure mean_squared_error_22
  module procedure mean_squared_error_weighted_22
 end interface
 public::mean_squared_error_raw
 interface mean_squared_error_raw
  module procedure mean_squared_error_raw_11
  module procedure mean_squared_error_raw_12
  module procedure mean_squared_error_raw_21
  module procedure mean_squared_error_raw_22
 end interface
 public::root_mean_squared_error
 interface root_mean_squared_error
  module procedure root_mean_squared_error_11
  module procedure root_mean_squared_error_weighted_11
  module procedure root_mean_squared_error_12
  module procedure root_mean_squared_error_weighted_12
  module procedure root_mean_squared_error_21
  module procedure root_mean_squared_error_weighted_21
  module procedure root_mean_squared_error_22
  module procedure root_mean_squared_error_weighted_22
 end interface
 public::root_mean_squared_error_raw
 interface root_mean_squared_error_raw
  module procedure root_mean_squared_error_raw_11
  module procedure root_mean_squared_error_raw_12
  module procedure root_mean_squared_error_raw_21
  module procedure root_mean_squared_error_raw_22
 end interface
 public::mean_absolute_error
 interface mean_absolute_error
  module procedure mean_absolute_error_11
  module procedure mean_absolute_error_weighted_11
  module procedure mean_absolute_error_12
  module procedure mean_absolute_error_weighted_12
  module procedure mean_absolute_error_21
  module procedure mean_absolute_error_weighted_21
  module procedure mean_absolute_error_22
  module procedure mean_absolute_error_weighted_22
 end interface
 public::mean_absolute_error_raw
 interface mean_absolute_error_raw
  module procedure mean_absolute_error_raw_11
  module procedure mean_absolute_error_raw_12
  module procedure mean_absolute_error_raw_21
  module procedure mean_absolute_error_raw_22
 end interface
 public::mean_absolute_percentage_error
 interface mean_absolute_percentage_error
  module procedure mean_absolute_percentage_error_11
  module procedure mean_absolute_percentage_error_weighted_11
  module procedure mean_absolute_percentage_error_12
  module procedure mean_absolute_percentage_error_weighted_12
  module procedure mean_absolute_percentage_error_21
  module procedure mean_absolute_percentage_error_weighted_21
  module procedure mean_absolute_percentage_error_22
  module procedure mean_absolute_percentage_error_weighted_22
 end interface
 public::mean_absolute_percentage_error_raw
 interface mean_absolute_percentage_error_raw
  module procedure mean_absolute_percentage_error_raw_11
  module procedure mean_absolute_percentage_error_raw_12
  module procedure mean_absolute_percentage_error_raw_21
  module procedure mean_absolute_percentage_error_raw_22
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
 function evaluate(y,p,kind,sample_weight,multioutput,output_weights,raw) result(value)
  real(dp),intent(in)::y(:,:),p(:,:)
  integer,intent(in)::kind
  real(dp),optional,intent(in)::sample_weight(:),output_weights(:)
  character(*),optional,intent(in)::multioutput
  logical,optional,intent(in)::raw
  real(dp),allocatable::value(:),loss(:,:),errors(:),w(:)
  real(dp)::mass,total
  logical::halting(size(ieee_all)),raised(size(ieee_all)),want_raw
  integer::n,q,j
  n=size(y,1);q=size(y,2)
  call require(n>0.and.q>0,'error metrics require nonempty targets')
  call require(all(shape(y)==shape(p)),'error metric target shapes differ')
  call require(all(ieee_is_finite(y)).and.all(ieee_is_finite(p)),'error metric targets must be finite')
  if(present(sample_weight)) call require(size(sample_weight)==n,'error metric sample weight length')
  if(present(multioutput)) then
   call require(multioutput=='uniform_average','invalid error multioutput mode; use *_raw')
  end if
  if(present(output_weights)) then
   ! RMSE delegates target validation to MSE(raw_values), then directly averages
   ! the roots. Unlike the other three metrics it permits singleton/nonfinite ow.
   if(kind/=1) then
    call require(q>1,'custom error output weights require multiple outputs')
    call require(all(ieee_is_finite(output_weights)),'error metric output weights must be finite')
   end if
   call require(size(output_weights)==q,'error metric output weight length')
  end if
  want_raw=.false.;if(present(raw)) want_raw=raw
  call suspend(halting,raised)
  loss=y-p
  select case(kind)
  case(0,1)
   loss=loss*loss
  case(2)
   loss=abs(loss)
  case(3)
   loss=abs(loss)/max(abs(y),epsilon(1._dp))
  end select
  mass=real(n,dp)
  if(present(sample_weight)) then
   mass=summed(sample_weight)
   do j=1,q
    loss(:,j)=loss(:,j)*sample_weight
   end do
  end if
  call require(mass/=0,'error metric sample weights sum to zero')
  errors=columns(loss)/mass
  if(kind==1) errors=sqrt(errors)
  if(want_raw) then
   value=errors
  else if(present(output_weights)) then
   total=summed(output_weights)
   call require(total/=0,'error metric output weights sum to zero')
   value=[summed(errors*output_weights)/total]
  else
   value=[summed(errors)/real(q,dp)]
  end if
  call restore(halting,raised)
 end function
 function mean_squared_error_11(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  character(*),optional,intent(in)::multioutput
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),0,sample_weight,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),0,sample_weight,multioutput=multioutput,raw=.true.)
 end function
 function mean_squared_error_weighted_11(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),intent(in)::multioutput(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),0,sample_weight,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),0,sample_weight,output_weights=multioutput,raw=.true.)
 end function
 function mean_squared_error_raw_11(y,p,sample_weight) result(v)
  real(dp),intent(in)::y(:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(p),0,sample_weight,raw=.true.)
 end function
 function mean_squared_error_12(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  character(*),optional,intent(in)::multioutput
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),0,sample_weight,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),0,sample_weight,multioutput=multioutput,raw=.true.)
 end function
 function mean_squared_error_weighted_12(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),intent(in)::multioutput(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),0,sample_weight,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),0,sample_weight,output_weights=multioutput,raw=.true.)
 end function
 function mean_squared_error_raw_12(y,p,sample_weight) result(v)
  real(dp),intent(in)::y(:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(p),0,sample_weight,raw=.true.)
 end function
 function mean_squared_error_21(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:,:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  character(*),optional,intent(in)::multioutput
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),0,sample_weight,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),0,sample_weight,multioutput=multioutput,raw=.true.)
 end function
 function mean_squared_error_weighted_21(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:,:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),intent(in)::multioutput(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),0,sample_weight,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),0,sample_weight,output_weights=multioutput,raw=.true.)
 end function
 function mean_squared_error_raw_21(y,p,sample_weight) result(v)
  real(dp),intent(in)::y(:,:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(p),0,sample_weight,raw=.true.)
 end function
 function mean_squared_error_22(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:,:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  character(*),optional,intent(in)::multioutput
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),0,sample_weight,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),0,sample_weight,multioutput=multioutput,raw=.true.)
 end function
 function mean_squared_error_weighted_22(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:,:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),intent(in)::multioutput(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),0,sample_weight,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),0,sample_weight,output_weights=multioutput,raw=.true.)
 end function
 function mean_squared_error_raw_22(y,p,sample_weight) result(v)
  real(dp),intent(in)::y(:,:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(p),0,sample_weight,raw=.true.)
 end function
 function root_mean_squared_error_11(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  character(*),optional,intent(in)::multioutput
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),1,sample_weight,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),1,sample_weight,multioutput=multioutput,raw=.true.)
 end function
 function root_mean_squared_error_weighted_11(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),intent(in)::multioutput(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),1,sample_weight,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),1,sample_weight,output_weights=multioutput,raw=.true.)
 end function
 function root_mean_squared_error_raw_11(y,p,sample_weight) result(v)
  real(dp),intent(in)::y(:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(p),1,sample_weight,raw=.true.)
 end function
 function root_mean_squared_error_12(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  character(*),optional,intent(in)::multioutput
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),1,sample_weight,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),1,sample_weight,multioutput=multioutput,raw=.true.)
 end function
 function root_mean_squared_error_weighted_12(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),intent(in)::multioutput(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),1,sample_weight,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),1,sample_weight,output_weights=multioutput,raw=.true.)
 end function
 function root_mean_squared_error_raw_12(y,p,sample_weight) result(v)
  real(dp),intent(in)::y(:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(p),1,sample_weight,raw=.true.)
 end function
 function root_mean_squared_error_21(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:,:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  character(*),optional,intent(in)::multioutput
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),1,sample_weight,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),1,sample_weight,multioutput=multioutput,raw=.true.)
 end function
 function root_mean_squared_error_weighted_21(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:,:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),intent(in)::multioutput(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),1,sample_weight,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),1,sample_weight,output_weights=multioutput,raw=.true.)
 end function
 function root_mean_squared_error_raw_21(y,p,sample_weight) result(v)
  real(dp),intent(in)::y(:,:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(p),1,sample_weight,raw=.true.)
 end function
 function root_mean_squared_error_22(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:,:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  character(*),optional,intent(in)::multioutput
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),1,sample_weight,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),1,sample_weight,multioutput=multioutput,raw=.true.)
 end function
 function root_mean_squared_error_weighted_22(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:,:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),intent(in)::multioutput(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),1,sample_weight,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),1,sample_weight,output_weights=multioutput,raw=.true.)
 end function
 function root_mean_squared_error_raw_22(y,p,sample_weight) result(v)
  real(dp),intent(in)::y(:,:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(p),1,sample_weight,raw=.true.)
 end function
 function mean_absolute_error_11(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  character(*),optional,intent(in)::multioutput
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),2,sample_weight,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),2,sample_weight,multioutput=multioutput,raw=.true.)
 end function
 function mean_absolute_error_weighted_11(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),intent(in)::multioutput(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),2,sample_weight,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),2,sample_weight,output_weights=multioutput,raw=.true.)
 end function
 function mean_absolute_error_raw_11(y,p,sample_weight) result(v)
  real(dp),intent(in)::y(:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(p),2,sample_weight,raw=.true.)
 end function
 function mean_absolute_error_12(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  character(*),optional,intent(in)::multioutput
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),2,sample_weight,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),2,sample_weight,multioutput=multioutput,raw=.true.)
 end function
 function mean_absolute_error_weighted_12(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),intent(in)::multioutput(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),2,sample_weight,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),2,sample_weight,output_weights=multioutput,raw=.true.)
 end function
 function mean_absolute_error_raw_12(y,p,sample_weight) result(v)
  real(dp),intent(in)::y(:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(p),2,sample_weight,raw=.true.)
 end function
 function mean_absolute_error_21(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:,:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  character(*),optional,intent(in)::multioutput
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),2,sample_weight,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),2,sample_weight,multioutput=multioutput,raw=.true.)
 end function
 function mean_absolute_error_weighted_21(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:,:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),intent(in)::multioutput(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),2,sample_weight,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),2,sample_weight,output_weights=multioutput,raw=.true.)
 end function
 function mean_absolute_error_raw_21(y,p,sample_weight) result(v)
  real(dp),intent(in)::y(:,:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(p),2,sample_weight,raw=.true.)
 end function
 function mean_absolute_error_22(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:,:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  character(*),optional,intent(in)::multioutput
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),2,sample_weight,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),2,sample_weight,multioutput=multioutput,raw=.true.)
 end function
 function mean_absolute_error_weighted_22(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:,:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),intent(in)::multioutput(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),2,sample_weight,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),2,sample_weight,output_weights=multioutput,raw=.true.)
 end function
 function mean_absolute_error_raw_22(y,p,sample_weight) result(v)
  real(dp),intent(in)::y(:,:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(p),2,sample_weight,raw=.true.)
 end function
 function mean_absolute_percentage_error_11(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  character(*),optional,intent(in)::multioutput
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),3,sample_weight,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),3,sample_weight,multioutput=multioutput,raw=.true.)
 end function
 function mean_absolute_percentage_error_weighted_11(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),intent(in)::multioutput(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),3,sample_weight,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),3,sample_weight,output_weights=multioutput,raw=.true.)
 end function
 function mean_absolute_percentage_error_raw_11(y,p,sample_weight) result(v)
  real(dp),intent(in)::y(:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(p),3,sample_weight,raw=.true.)
 end function
 function mean_absolute_percentage_error_12(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  character(*),optional,intent(in)::multioutput
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),3,sample_weight,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),3,sample_weight,multioutput=multioutput,raw=.true.)
 end function
 function mean_absolute_percentage_error_weighted_12(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),intent(in)::multioutput(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),3,sample_weight,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),3,sample_weight,output_weights=multioutput,raw=.true.)
 end function
 function mean_absolute_percentage_error_raw_12(y,p,sample_weight) result(v)
  real(dp),intent(in)::y(:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(p),3,sample_weight,raw=.true.)
 end function
 function mean_absolute_percentage_error_21(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:,:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  character(*),optional,intent(in)::multioutput
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),3,sample_weight,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),3,sample_weight,multioutput=multioutput,raw=.true.)
 end function
 function mean_absolute_percentage_error_weighted_21(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:,:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),intent(in)::multioutput(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),3,sample_weight,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),3,sample_weight,output_weights=multioutput,raw=.true.)
 end function
 function mean_absolute_percentage_error_raw_21(y,p,sample_weight) result(v)
  real(dp),intent(in)::y(:,:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(p),3,sample_weight,raw=.true.)
 end function
 function mean_absolute_percentage_error_22(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:,:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  character(*),optional,intent(in)::multioutput
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),3,sample_weight,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),3,sample_weight,multioutput=multioutput,raw=.true.)
 end function
 function mean_absolute_percentage_error_weighted_22(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:,:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),intent(in)::multioutput(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),3,sample_weight,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),3,sample_weight,output_weights=multioutput,raw=.true.)
 end function
 function mean_absolute_percentage_error_raw_22(y,p,sample_weight) result(v)
  real(dp),intent(in)::y(:,:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(p),3,sample_weight,raw=.true.)
 end function
end module
