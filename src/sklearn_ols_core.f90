! LinearRegression preprocessing and least squares: sklearn 1.6.1
! linear_model/_base.py (BSD). Native typed storage is MIT.
module sklearn_ols_core
 use sklearn_common,only:dp,require,check_x,dense_svd
 use sklearn_sparse,only:csr_matrix
 use sklearn_ridge_estimator_core,only:column_means
 use sklearn_standard_scaling,only:standard_scaler
 use sklearn_ridge_solver,only:ridge_centered_sparse,ridge_solver_options,ridge_solution
 use sklearn_nnls,only:positive_least_squares
 use,intrinsic::ieee_arithmetic
 use,intrinsic::ieee_exceptions
 implicit none
 private
 public::fit_ols_estimator,ols_solution
 type::ols_solution
  real(dp),allocatable::coefficients(:,:),intercept(:),singular(:)
  integer,allocatable::iterations(:),stop_code(:)
  integer::rank=0
  character(8)::solver=''
 end type
 interface fit_ols_estimator
  module procedure dense_fit,sparse_fit
 end interface
contains
 subroutine weights_check(w,n)
  real(dp),optional,intent(in)::w(:)
  integer,intent(in)::n
  if(.not.present(w))return
  call require(size(w)==n,'LinearRegression sample weight length mismatch')
  call require(all(ieee_is_finite(w)),'LinearRegression sample weights must be finite')
  call require(all(w>=0),'LinearRegression sample weights must be nonnegative')
 end subroutine
 function dense_fit(x,y,fit_intercept,positive,sample_weight)result(fit)
  real(dp),intent(in)::x(:,:),y(:,:)
  logical,intent(in)::fit_intercept,positive
  real(dp),optional,intent(in)::sample_weight(:)
  type(ols_solution)::fit
  real(dp),allocatable::a(:,:),b(:,:),xm(:),ym(:),u(:,:),s(:),v(:,:),coefs(:),projection(:)
  real(dp)::cutoff
  logical::modes(size(ieee_all)),flags(size(ieee_all))
  integer::n,p,t,j
  call ieee_get_halting_mode(ieee_all,modes);call ieee_get_flag(ieee_all,flags)
  call ieee_set_halting_mode(ieee_all,.false.)
  call check_x(x);call check_x(y);n=size(x,1);p=size(x,2);t=size(y,2)
  call require(size(y,1)==n,'LinearRegression target sample count mismatch')
  call weights_check(sample_weight,n)
  allocate(xm(p),ym(t));xm=0;ym=0;a=x;b=y
  if(fit_intercept)then
   xm=column_means(x,sample_weight);ym=column_means(y,sample_weight)
   a=a-spread(xm,1,n);b=b-spread(ym,1,n)
  end if
  if(present(sample_weight))then
   a=a*spread(sqrt(sample_weight),2,p);b=b*spread(sqrt(sample_weight),2,t)
  end if
  call check_x(a);call check_x(b)
  allocate(fit%coefficients(t,p))
  if(positive)then
   fit%solver='nnls'
   do j=1,t
    call positive_least_squares(a,b(:,j),0._dp,coefs)
    fit%coefficients(j,:)=coefs
   end do
  else
   fit%solver='svd';allocate(u(n,p),s(p),v(p,p),projection(p))
   call dense_svd(a,s,u,v)
   cutoff=max(n,p)*epsilon(1._dp)*maxval(s);fit%rank=count(s>cutoff)
   fit%singular=s(:min(n,p))
   do j=1,t
    projection=matmul(transpose(u),b(:,j))
    where(s>cutoff)
     projection=projection/s
    elsewhere
     projection=0
    end where
    fit%coefficients(j,:)=matmul(v,projection)
   end do
  end if
  fit%intercept=ym-matmul(fit%coefficients,xm)
  if(.not.fit_intercept)fit%intercept=0
  call ieee_set_flag(ieee_all,flags);call ieee_set_halting_mode(ieee_all,modes)
 end function
 function sparse_fit(x,y,fit_intercept,positive,sample_weight)result(fit)
  type(csr_matrix),intent(in)::x
  real(dp),intent(in)::y(:,:)
  logical,intent(in)::fit_intercept,positive
  real(dp),optional,intent(in)::sample_weight(:)
  type(ols_solution)::fit
  type(ridge_solution)::solved
  type(ridge_solver_options)::options
  type(standard_scaler)::scaler
  real(dp),allocatable::xm(:),ym(:),b(:,:)
  logical::modes(size(ieee_all)),flags(size(ieee_all))
  call ieee_get_halting_mode(ieee_all,modes);call ieee_get_flag(ieee_all,flags)
  call ieee_set_halting_mode(ieee_all,.false.)
  call require(.not.positive,'positive LinearRegression requires dense feature input')
  call x%validate();call check_x(y)
  call require(x%n_rows>0.and.x%n_cols>0,'LinearRegression sparse design must be nonempty')
  call require(all(ieee_is_finite(x%data)),'LinearRegression sparse data must be finite')
  call require(size(y,1)==x%n_rows,'LinearRegression target sample count mismatch')
  call weights_check(sample_weight,x%n_rows)
  allocate(xm(x%n_cols),ym(size(y,2)));xm=0;ym=0;b=y
  if(fit_intercept)then
   ym=column_means(y,sample_weight)
   scaler%with_mean=.false.;call scaler%fit(x,sample_weight);xm=scaler%mean_
   b=b-spread(ym,1,x%n_rows)
  end if
  options%solver='lsqr';options%tol=1.e-6_dp
  solved=ridge_centered_sparse(x,b,[0._dp],xm,options,sample_weight,unweighted_sum=.not.present(sample_weight))
  fit%solver='lsqr';fit%coefficients=solved%coefficients
  fit%iterations=solved%n_iter;fit%stop_code=solved%stop_code
  fit%intercept=ym-matmul(fit%coefficients,xm)
  if(.not.fit_intercept)fit%intercept=0
  call ieee_set_flag(ieee_all,flags);call ieee_set_halting_mode(ieee_all,modes)
 end function
end module
