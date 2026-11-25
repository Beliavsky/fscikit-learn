! sklearn 1.6.1 logarithmic errors and Poisson/Gamma deviance (BSD-3-Clause).
! NumPy 2.2.6 buffered reductions retain their BSD notice under LICENSES/.
! Explicit rank/legacy adapters and quiet IEEE composition are MIT integration.
module sklearn_distribution_errors
 use sklearn_common,only:dp,require
 use,intrinsic::iso_c_binding,only:c_double
 use,intrinsic::ieee_arithmetic
 use,intrinsic::ieee_exceptions
 implicit none
 private
 public::mean_squared_log_error,root_mean_squared_log_error
 public::mean_squared_log_error_raw,root_mean_squared_log_error_raw
 public::mean_poisson_deviance,mean_gamma_deviance
 interface
  function c_log1p(x) bind(C,name='log1p') result(v)
   import c_double
   real(c_double),value::x
   real(c_double)::v
  end function
 end interface
 interface mean_squared_log_error
  module procedure mean_squared_log_error_11
  module procedure mean_squared_log_error_weighted_11
  module procedure mean_squared_log_error_12
  module procedure mean_squared_log_error_weighted_12
  module procedure mean_squared_log_error_21
  module procedure mean_squared_log_error_weighted_21
  module procedure mean_squared_log_error_22
  module procedure mean_squared_log_error_weighted_22
 end interface
 interface mean_squared_log_error_raw
  module procedure mean_squared_log_error_raw_11
  module procedure mean_squared_log_error_raw_12
  module procedure mean_squared_log_error_raw_21
  module procedure mean_squared_log_error_raw_22
 end interface
 interface root_mean_squared_log_error
  module procedure root_mean_squared_log_error_11
  module procedure root_mean_squared_log_error_weighted_11
  module procedure root_mean_squared_log_error_legacy_11
  module procedure root_mean_squared_log_error_12
  module procedure root_mean_squared_log_error_weighted_12
  module procedure root_mean_squared_log_error_legacy_12
  module procedure root_mean_squared_log_error_21
  module procedure root_mean_squared_log_error_weighted_21
  module procedure root_mean_squared_log_error_legacy_21
  module procedure root_mean_squared_log_error_22
  module procedure root_mean_squared_log_error_weighted_22
  module procedure root_mean_squared_log_error_legacy_22
 end interface
 interface root_mean_squared_log_error_raw
  module procedure root_mean_squared_log_error_raw_11
  module procedure root_mean_squared_log_error_raw_12
  module procedure root_mean_squared_log_error_raw_21
  module procedure root_mean_squared_log_error_raw_22
 end interface
 interface mean_poisson_deviance
  module procedure mean_poisson_deviance_11
  module procedure mean_poisson_deviance_12
  module procedure mean_poisson_deviance_21
  module procedure mean_poisson_deviance_22
 end interface
 interface mean_gamma_deviance
  module procedure mean_gamma_deviance_11
  module procedure mean_gamma_deviance_12
  module procedure mean_gamma_deviance_21
  module procedure mean_gamma_deviance_22
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
 subroutine check_targets(y,p)
  real(dp),intent(in)::y(:,:),p(:,:)
  call require(size(y,1)>0.and.size(y,2)>0,'distribution errors require nonempty targets')
  call require(all(shape(y)==shape(p)),'distribution error target shapes differ')
  call require(all(ieee_is_finite(y)).and.all(ieee_is_finite(p)),'distribution error targets must be finite')
 end subroutine
 function log_evaluate(y,p,root,sample_weight,multioutput,output_weights,raw) result(value)
  real(dp),intent(in)::y(:,:),p(:,:)
  logical,intent(in)::root
  real(dp),optional,intent(in)::sample_weight(:),output_weights(:)
  character(*),optional,intent(in)::multioutput
  logical,optional,intent(in)::raw
  real(dp),allocatable::value(:),loss(:,:),errors(:)
  real(dp)::mass,total,difference
  logical::halting(size(ieee_all)),raised(size(ieee_all)),want_raw
  integer::i,j,n,q
  call check_targets(y,p);n=size(y,1);q=size(y,2)
  call require(all(y>-1).and.all(p>-1),'logarithmic targets must be greater than -1')
  if(present(sample_weight)) call require(size(sample_weight)==n,'distribution error sample weight length')
  if(present(multioutput)) then
   call require(multioutput=='uniform_average','invalid logarithmic multioutput mode; use *_raw')
  end if
  if(present(output_weights)) then
   call require(q>1,'custom logarithmic output weights require multiple outputs')
   call require(size(output_weights)==q,'logarithmic output weight length')
   call require(all(ieee_is_finite(output_weights)),'logarithmic output weights must be finite')
  end if
  want_raw=.false.;if(present(raw)) want_raw=raw
  call suspend(halting,raised)
  allocate(loss(n,q))
  do j=1,q
   do i=1,n
    difference=c_log1p(y(i,j))-c_log1p(p(i,j));loss(i,j)=difference*difference
   end do
  end do
  mass=real(n,dp)
  if(present(sample_weight)) then
   mass=summed(sample_weight)
   do j=1,q
    loss(:,j)=loss(:,j)*sample_weight
   end do
  end if
  call require(mass/=0,'distribution error sample weights sum to zero')
  errors=columns(loss)/mass
  if(root) errors=sqrt(errors)
  if(want_raw) then
   value=errors
  else if(present(output_weights)) then
   total=summed(output_weights)
   call require(total/=0,'logarithmic output weights sum to zero')
   value=[summed(errors*output_weights)/total]
  else
   value=[summed(errors)/real(q,dp)]
  end if
  call restore(halting,raised)
 end function
 real(dp) function deviance(y,p,poisson,sample_weight) result(value)
  real(dp),intent(in)::y(:,:),p(:,:)
  logical,intent(in)::poisson
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::loss(size(y,1)),term,ratio,mass
  logical::halting(size(ieee_all)),raised(size(ieee_all))
  integer::i,n
  call check_targets(y,p);n=size(y,1)
  call require(size(y,2)==1,'deviance rejects multioutput targets')
  if(present(sample_weight)) call require(size(sample_weight)==n,'distribution error sample weight length')
  call require(all(p>0),'deviance predictions must be strictly positive')
  if(poisson) then
   call require(all(y>=0),'Poisson targets must be nonnegative')
  else
   call require(all(y>0),'Gamma targets must be strictly positive')
  end if
  call suspend(halting,raised)
  do i=1,n
   if(poisson) then
    term=0
    if(y(i,1)/=0) then
     ratio=y(i,1)/p(i,1);term=y(i,1)*log(ratio)
    end if
    term=subtract(subtract(term,y(i,1)),-p(i,1));loss(i)=2*term
   else
    ratio=p(i,1)/y(i,1);term=log(ratio)
    ratio=y(i,1)/p(i,1);term=subtract(term,-ratio)
    loss(i)=2*subtract(term,1._dp)
   end if
  end do
  mass=real(n,dp)
  if(present(sample_weight)) then
   mass=summed(sample_weight);loss=loss*sample_weight
  end if
  call require(mass/=0,'distribution error sample weights sum to zero')
  value=summed(loss)/mass
  call restore(halting,raised)
 end function
 function mean_squared_log_error_11(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  character(*),optional,intent(in)::multioutput
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=log_evaluate(matrix(y),matrix(p),.false.,sample_weight, &
   multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=log_evaluate(matrix(y),matrix(p),.false.,sample_weight, &
   multioutput=multioutput,raw=.true.)
 end function
 function mean_squared_log_error_weighted_11(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),intent(in)::multioutput(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=log_evaluate(matrix(y),matrix(p),.false.,sample_weight, &
   output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=log_evaluate(matrix(y),matrix(p),.false.,sample_weight, &
   output_weights=multioutput,raw=.true.)
 end function
 function mean_squared_log_error_raw_11(y,p,sample_weight) result(v)
  real(dp),intent(in)::y(:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),allocatable::v(:)
  v=log_evaluate(matrix(y),matrix(p),.false.,sample_weight,raw=.true.)
 end function
 function mean_squared_log_error_12(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  character(*),optional,intent(in)::multioutput
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=log_evaluate(matrix(y),matrix(p),.false.,sample_weight, &
   multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=log_evaluate(matrix(y),matrix(p),.false.,sample_weight, &
   multioutput=multioutput,raw=.true.)
 end function
 function mean_squared_log_error_weighted_12(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),intent(in)::multioutput(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=log_evaluate(matrix(y),matrix(p),.false.,sample_weight, &
   output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=log_evaluate(matrix(y),matrix(p),.false.,sample_weight, &
   output_weights=multioutput,raw=.true.)
 end function
 function mean_squared_log_error_raw_12(y,p,sample_weight) result(v)
  real(dp),intent(in)::y(:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),allocatable::v(:)
  v=log_evaluate(matrix(y),matrix(p),.false.,sample_weight,raw=.true.)
 end function
 function mean_squared_log_error_21(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:,:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  character(*),optional,intent(in)::multioutput
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=log_evaluate(matrix(y),matrix(p),.false.,sample_weight, &
   multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=log_evaluate(matrix(y),matrix(p),.false.,sample_weight, &
   multioutput=multioutput,raw=.true.)
 end function
 function mean_squared_log_error_weighted_21(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:,:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),intent(in)::multioutput(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=log_evaluate(matrix(y),matrix(p),.false.,sample_weight, &
   output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=log_evaluate(matrix(y),matrix(p),.false.,sample_weight, &
   output_weights=multioutput,raw=.true.)
 end function
 function mean_squared_log_error_raw_21(y,p,sample_weight) result(v)
  real(dp),intent(in)::y(:,:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),allocatable::v(:)
  v=log_evaluate(matrix(y),matrix(p),.false.,sample_weight,raw=.true.)
 end function
 function mean_squared_log_error_22(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:,:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  character(*),optional,intent(in)::multioutput
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=log_evaluate(matrix(y),matrix(p),.false.,sample_weight, &
   multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=log_evaluate(matrix(y),matrix(p),.false.,sample_weight, &
   multioutput=multioutput,raw=.true.)
 end function
 function mean_squared_log_error_weighted_22(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:,:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),intent(in)::multioutput(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=log_evaluate(matrix(y),matrix(p),.false.,sample_weight, &
   output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=log_evaluate(matrix(y),matrix(p),.false.,sample_weight, &
   output_weights=multioutput,raw=.true.)
 end function
 function mean_squared_log_error_raw_22(y,p,sample_weight) result(v)
  real(dp),intent(in)::y(:,:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),allocatable::v(:)
  v=log_evaluate(matrix(y),matrix(p),.false.,sample_weight,raw=.true.)
 end function
 function root_mean_squared_log_error_11(y,predicted,sample_weight,multioutput,raw_values,output_weights) result(v)
  real(dp),intent(in)::y(:),predicted(:)
  real(dp),optional,intent(in)::sample_weight(:)
  character(*),intent(in)::multioutput
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  real(dp),optional,intent(in)::output_weights(:)
  call require(.not.present(output_weights), &
   'RMSLE cannot combine multioutput and output_weights')
  tmp=log_evaluate(matrix(y),matrix(predicted),.true.,sample_weight, &
   multioutput=multioutput,output_weights=output_weights)
  v=tmp(1)
  if(present(raw_values)) raw_values=log_evaluate(matrix(y),matrix(predicted),.true.,sample_weight, &
   multioutput=multioutput,output_weights=output_weights,raw=.true.)
 end function
 function root_mean_squared_log_error_weighted_11(y,predicted,sample_weight,output_weights,raw_values,multioutput) result(v)
  real(dp),intent(in)::y(:),predicted(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),intent(in)::multioutput(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  real(dp),optional,intent(in)::output_weights(:)
  call require(.not.present(output_weights), &
   'RMSLE cannot combine multioutput and output_weights')
  tmp=log_evaluate(matrix(y),matrix(predicted),.true.,sample_weight, &
   output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=log_evaluate(matrix(y),matrix(predicted),.true.,sample_weight, &
   output_weights=multioutput,raw=.true.)
 end function
 function root_mean_squared_log_error_raw_11(y,predicted,sample_weight) result(v)
  real(dp),intent(in)::y(:),predicted(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),allocatable::v(:)
  v=log_evaluate(matrix(y),matrix(predicted),.true.,sample_weight,raw=.true.)
 end function
 function root_mean_squared_log_error_legacy_11(y,predicted,sample_weight,output_weights,raw_values) result(v)
  real(dp),intent(in)::y(:),predicted(:)
  real(dp),optional,intent(in)::sample_weight(:),output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=log_evaluate(matrix(y),matrix(predicted),.true.,sample_weight,output_weights=output_weights)
  v=tmp(1)
  if(present(raw_values)) raw_values=log_evaluate(matrix(y),matrix(predicted),.true.,sample_weight, &
   output_weights=output_weights,raw=.true.)
 end function
 function root_mean_squared_log_error_12(y,predicted,sample_weight,multioutput,raw_values,output_weights) result(v)
  real(dp),intent(in)::y(:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  character(*),intent(in)::multioutput
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  real(dp),optional,intent(in)::output_weights(:)
  call require(.not.present(output_weights), &
   'RMSLE cannot combine multioutput and output_weights')
  tmp=log_evaluate(matrix(y),matrix(predicted),.true.,sample_weight, &
   multioutput=multioutput,output_weights=output_weights)
  v=tmp(1)
  if(present(raw_values)) raw_values=log_evaluate(matrix(y),matrix(predicted),.true.,sample_weight, &
   multioutput=multioutput,output_weights=output_weights,raw=.true.)
 end function
 function root_mean_squared_log_error_weighted_12(y,predicted,sample_weight,output_weights,raw_values,multioutput) result(v)
  real(dp),intent(in)::y(:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),intent(in)::multioutput(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  real(dp),optional,intent(in)::output_weights(:)
  call require(.not.present(output_weights), &
   'RMSLE cannot combine multioutput and output_weights')
  tmp=log_evaluate(matrix(y),matrix(predicted),.true.,sample_weight, &
   output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=log_evaluate(matrix(y),matrix(predicted),.true.,sample_weight, &
   output_weights=multioutput,raw=.true.)
 end function
 function root_mean_squared_log_error_raw_12(y,predicted,sample_weight) result(v)
  real(dp),intent(in)::y(:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),allocatable::v(:)
  v=log_evaluate(matrix(y),matrix(predicted),.true.,sample_weight,raw=.true.)
 end function
 function root_mean_squared_log_error_legacy_12(y,predicted,sample_weight,output_weights,raw_values) result(v)
  real(dp),intent(in)::y(:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:),output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=log_evaluate(matrix(y),matrix(predicted),.true.,sample_weight,output_weights=output_weights)
  v=tmp(1)
  if(present(raw_values)) raw_values=log_evaluate(matrix(y),matrix(predicted),.true.,sample_weight, &
   output_weights=output_weights,raw=.true.)
 end function
 function root_mean_squared_log_error_21(y,predicted,sample_weight,multioutput,raw_values,output_weights) result(v)
  real(dp),intent(in)::y(:,:),predicted(:)
  real(dp),optional,intent(in)::sample_weight(:)
  character(*),intent(in)::multioutput
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  real(dp),optional,intent(in)::output_weights(:)
  call require(.not.present(output_weights), &
   'RMSLE cannot combine multioutput and output_weights')
  tmp=log_evaluate(matrix(y),matrix(predicted),.true.,sample_weight, &
   multioutput=multioutput,output_weights=output_weights)
  v=tmp(1)
  if(present(raw_values)) raw_values=log_evaluate(matrix(y),matrix(predicted),.true.,sample_weight, &
   multioutput=multioutput,output_weights=output_weights,raw=.true.)
 end function
 function root_mean_squared_log_error_weighted_21(y,predicted,sample_weight,output_weights,raw_values,multioutput) result(v)
  real(dp),intent(in)::y(:,:),predicted(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),intent(in)::multioutput(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  real(dp),optional,intent(in)::output_weights(:)
  call require(.not.present(output_weights), &
   'RMSLE cannot combine multioutput and output_weights')
  tmp=log_evaluate(matrix(y),matrix(predicted),.true.,sample_weight, &
   output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=log_evaluate(matrix(y),matrix(predicted),.true.,sample_weight, &
   output_weights=multioutput,raw=.true.)
 end function
 function root_mean_squared_log_error_raw_21(y,predicted,sample_weight) result(v)
  real(dp),intent(in)::y(:,:),predicted(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),allocatable::v(:)
  v=log_evaluate(matrix(y),matrix(predicted),.true.,sample_weight,raw=.true.)
 end function
 function root_mean_squared_log_error_legacy_21(y,predicted,sample_weight,output_weights,raw_values) result(v)
  real(dp),intent(in)::y(:,:),predicted(:)
  real(dp),optional,intent(in)::sample_weight(:),output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=log_evaluate(matrix(y),matrix(predicted),.true.,sample_weight,output_weights=output_weights)
  v=tmp(1)
  if(present(raw_values)) raw_values=log_evaluate(matrix(y),matrix(predicted),.true.,sample_weight, &
   output_weights=output_weights,raw=.true.)
 end function
 function root_mean_squared_log_error_22(y,predicted,sample_weight,multioutput,raw_values,output_weights) result(v)
  real(dp),intent(in)::y(:,:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  character(*),intent(in)::multioutput
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  real(dp),optional,intent(in)::output_weights(:)
  call require(.not.present(output_weights), &
   'RMSLE cannot combine multioutput and output_weights')
  tmp=log_evaluate(matrix(y),matrix(predicted),.true.,sample_weight, &
   multioutput=multioutput,output_weights=output_weights)
  v=tmp(1)
  if(present(raw_values)) raw_values=log_evaluate(matrix(y),matrix(predicted),.true.,sample_weight, &
   multioutput=multioutput,output_weights=output_weights,raw=.true.)
 end function
 function root_mean_squared_log_error_weighted_22(y,predicted,sample_weight,output_weights,raw_values,multioutput) result(v)
  real(dp),intent(in)::y(:,:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),intent(in)::multioutput(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  real(dp),optional,intent(in)::output_weights(:)
  call require(.not.present(output_weights), &
   'RMSLE cannot combine multioutput and output_weights')
  tmp=log_evaluate(matrix(y),matrix(predicted),.true.,sample_weight, &
   output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=log_evaluate(matrix(y),matrix(predicted),.true.,sample_weight, &
   output_weights=multioutput,raw=.true.)
 end function
 function root_mean_squared_log_error_raw_22(y,predicted,sample_weight) result(v)
  real(dp),intent(in)::y(:,:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),allocatable::v(:)
  v=log_evaluate(matrix(y),matrix(predicted),.true.,sample_weight,raw=.true.)
 end function
 function root_mean_squared_log_error_legacy_22(y,predicted,sample_weight,output_weights,raw_values) result(v)
  real(dp),intent(in)::y(:,:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:),output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=log_evaluate(matrix(y),matrix(predicted),.true.,sample_weight,output_weights=output_weights)
  v=tmp(1)
  if(present(raw_values)) raw_values=log_evaluate(matrix(y),matrix(predicted),.true.,sample_weight, &
   output_weights=output_weights,raw=.true.)
 end function
 real(dp) function mean_poisson_deviance_11(y,p,sample_weight) result(v)
  real(dp),intent(in)::y(:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  v=deviance(matrix(y),matrix(p),.true.,sample_weight)
 end function
 real(dp) function mean_poisson_deviance_12(y,p,sample_weight) result(v)
  real(dp),intent(in)::y(:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  v=deviance(matrix(y),matrix(p),.true.,sample_weight)
 end function
 real(dp) function mean_poisson_deviance_21(y,p,sample_weight) result(v)
  real(dp),intent(in)::y(:,:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  v=deviance(matrix(y),matrix(p),.true.,sample_weight)
 end function
 real(dp) function mean_poisson_deviance_22(y,p,sample_weight) result(v)
  real(dp),intent(in)::y(:,:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  v=deviance(matrix(y),matrix(p),.true.,sample_weight)
 end function
 real(dp) function mean_gamma_deviance_11(y,p,sample_weight) result(v)
  real(dp),intent(in)::y(:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  v=deviance(matrix(y),matrix(p),.false.,sample_weight)
 end function
 real(dp) function mean_gamma_deviance_12(y,p,sample_weight) result(v)
  real(dp),intent(in)::y(:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  v=deviance(matrix(y),matrix(p),.false.,sample_weight)
 end function
 real(dp) function mean_gamma_deviance_21(y,p,sample_weight) result(v)
  real(dp),intent(in)::y(:,:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  v=deviance(matrix(y),matrix(p),.false.,sample_weight)
 end function
 real(dp) function mean_gamma_deviance_22(y,p,sample_weight) result(v)
  real(dp),intent(in)::y(:,:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  v=deviance(matrix(y),matrix(p),.false.,sample_weight)
 end function
end module
