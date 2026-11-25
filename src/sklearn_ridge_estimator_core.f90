! Ridge estimator preprocessing: sklearn 1.6.1 linear_model/_base.py/_ridge.py (BSD).
! Retain sklearn and NumPy notices. Native typed integration is MIT.
module sklearn_ridge_estimator_core
 use sklearn_common,only:dp,require,check_x
 use sklearn_sparse,only:csr_matrix
 use sklearn_standard_scaling,only:standard_scaler
 use sklearn_ridge_solver,only:ridge_regression,ridge_centered_sparse,ridge_solver_options,ridge_solution
 use,intrinsic::ieee_arithmetic
 use,intrinsic::ieee_exceptions
 implicit none
 private
 public::fit_ridge_estimator
 interface fit_ridge_estimator
  module procedure dense_fit,sparse_fit
 end interface
contains
 recursive real(dp)function numpy_total(a)result(v)
  real(dp),intent(in)::a(:)
  real(dp)::lane(8)
  integer::n,i,j,whole,split
  n=size(a);v=-0._dp
  if(n<8)then
   do i=1,n
    v=v+a(i)
   end do
  else if(n<=128)then
   lane=a(:8);whole=n-mod(n,8)
   do i=9,whole,8
    do j=1,8
     lane(j)=lane(j)+a(i+j-1)
    end do
   end do
   v=((lane(1)+lane(2))+(lane(3)+lane(4)))+((lane(5)+lane(6))+(lane(7)+lane(8)))
   do i=whole+1,n
    v=v+a(i)
   end do
  else
   split=n/2;split=split-mod(split,8);v=numpy_total(a(:split))+numpy_total(a(split+1:))
  end if
  if(ieee_is_finite(v))then
   if(v==0)v=0
  end if
 end function
 function column_means(a,w)result(mean_)
  real(dp),intent(in)::a(:,:)
  real(dp),optional,intent(in)::w(:)
  real(dp)::mean_(size(a,2)),mass,total
  integer::i,j
  mass=real(size(a,1),dp);if(present(w))mass=numpy_total(w)
  call require(mass/=0,'Ridge centering weights must have nonzero total')
  do j=1,size(a,2)
   if(size(a,2)==1)then
    if(present(w))then
     total=numpy_total(a(:,j)*w)
    else
     total=numpy_total(a(:,j))
    end if
   else
    total=0
    do i=1,size(a,1)
     if(present(w))then
      total=total+a(i,j)*w(i)
     else
      total=total+a(i,j)
     end if
    end do
   end if
   mean_(j)=total/mass
  end do
 end function
 subroutine controls(o)
  type(ridge_solver_options),intent(in)::o
  call require(o%max_iter==-1.or.o%max_iter>=1,'Ridge max_iter must be positive or -1 for None')
  call require(ieee_is_finite(o%tol),'Ridge tolerance must be finite')
  call require(o%tol>=0,'Ridge tolerance must be nonnegative')
  call require(o%random_state>=0,'Ridge random_state must be nonnegative')
  if(o%positive)call require(o%solver=='auto'.or.o%solver=='lbfgs','positive Ridge requires auto or lbfgs')
  if(o%solver=='lbfgs')call require(o%positive,'Ridge lbfgs requires positive coefficients')
 end subroutine
 function dense_fit(x,y,alpha,options,fit_intercept,sample_weight)result(fit)
  real(dp),intent(in)::x(:,:),y(:,:),alpha(:)
  type(ridge_solver_options),intent(in)::options
  logical,intent(in)::fit_intercept
  real(dp),optional,intent(in)::sample_weight(:)
  type(ridge_solution)::fit
  real(dp),allocatable::a(:,:),b(:,:),xm(:),ym(:)
  logical::modes(size(ieee_all)),flags(size(ieee_all))
  integer::j
  call ieee_get_halting_mode(ieee_all,modes);call ieee_get_flag(ieee_all,flags)
  call ieee_set_halting_mode(ieee_all,.false.)
  call controls(options);call check_x(x);call check_x(y)
  call require(size(x,1)==size(y,1),'Ridge target sample count mismatch')
  if(present(sample_weight))then
   call require(size(sample_weight)==size(x,1),'Ridge sample weight length mismatch')
   call require(all(ieee_is_finite(sample_weight)),'Ridge sample weights must be finite')
  end if
  a=x;b=y;allocate(xm(size(x,2)),ym(size(y,2)));xm=0;ym=0
  if(fit_intercept)then
   xm=column_means(x,sample_weight);ym=column_means(y,sample_weight)
   a=a-spread(xm,1,size(x,1));b=b-spread(ym,1,size(y,1))
  end if
  fit=ridge_regression(a,b,alpha,options,sample_weight)
  if(fit_intercept)then
   fit%intercept=ym-matmul(fit%coefficients,xm)
  else
   fit%intercept=0
  end if
  call ieee_set_flag(ieee_all,flags);call ieee_set_halting_mode(ieee_all,modes)
 end function
 function sparse_fit(x,y,alpha,options,fit_intercept,sample_weight)result(fit)
  type(csr_matrix),intent(in)::x
  real(dp),intent(in)::y(:,:),alpha(:)
  type(ridge_solver_options),intent(in)::options
  logical,intent(in)::fit_intercept
  real(dp),optional,intent(in)::sample_weight(:)
  type(ridge_solution)::fit
  type(ridge_solver_options)::o
  type(standard_scaler)::scaler
  real(dp),allocatable::b(:,:),xm(:),ym(:)
  logical::modes(size(ieee_all)),flags(size(ieee_all))
  call ieee_get_halting_mode(ieee_all,modes);call ieee_get_flag(ieee_all,flags)
  call ieee_set_halting_mode(ieee_all,.false.)
  call controls(options);call x%validate();call check_x(y)
  call require(x%n_rows==size(y,1),'Ridge target sample count mismatch')
  if(present(sample_weight))then
   call require(size(sample_weight)==x%n_rows,'Ridge sample weight length mismatch')
   call require(all(ieee_is_finite(sample_weight)),'Ridge sample weights must be finite')
  end if
  o=options
  if(fit_intercept.and..not.o%positive)then
   call require(any(o%solver==[character(16)::'auto','sparse_cg','lsqr','sag','lbfgs']), &
    'this Ridge solver cannot fit an intercept on sparse inputs')
   if(o%solver/='lsqr'.and.o%solver/='lbfgs')then
    if(.not.(o%solver=='sag'.and.o%max_iter==-1.and.o%tol>1.e-4_dp))o%solver='sparse_cg'
   end if
  end if
  b=y;allocate(xm(x%n_cols),ym(size(y,2)));xm=0;ym=0
  if(fit_intercept)then
   ym=column_means(y,sample_weight)
   scaler%with_mean=.false.;call scaler%fit(x,sample_weight)
   xm=scaler%mean_;b=b-spread(ym,1,size(y,1))
  end if
  if(fit_intercept.and.o%solver=='sag')then
   o%return_intercept=.true.;fit=ridge_regression(x,b,alpha,o,sample_weight)
   fit%intercept=fit%intercept+ym
  else if(fit_intercept)then
   fit=ridge_centered_sparse(x,b,alpha,xm,o,sample_weight)
   fit%intercept=ym-matmul(fit%coefficients,xm)
  else
   fit=ridge_regression(x,b,alpha,o,sample_weight)
  end if
  call ieee_set_flag(ieee_all,flags);call ieee_set_halting_mode(ieee_all,modes)
 end function
end module
