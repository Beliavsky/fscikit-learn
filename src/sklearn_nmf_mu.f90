! Dense sklearn 1.6.1 decomposition/_nmf.py multiplicative-update NMF.
! BSD attribution: THIRD_PARTY_NOTICES.md and LICENSES/scikit-learn-COPYING.
module sklearn_nmf_mu
  use sklearn_common, only: dp,require
  use sklearn_sparse,only:csr_matrix
  use sklearn_nmf_sparse_data,only:nmf_sparse_cross
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_value,ieee_quiet_nan
  implicit none
  private
  public :: nmf_mu_solve,nmf_sparse_error
  interface nmf_mu_solve
    module procedure nmf_mu_solve_dense,nmf_mu_solve_sparse
  end interface
  real(dp), parameter :: eps32=1.1920928955078125e-7_dp,limit=huge(1._dp)/256
contains
  impure elemental real(dp) function multiply(a,b) result(c)
    real(dp), intent(in) :: a,b
    c=0
    if(a==0.or.b==0) return
    call require(log(abs(a))+log(abs(b))<=log(limit),'NMF MU product exceeds working range')
    c=a*b
  end function
  impure elemental real(dp) function divide(a,b) result(c)
    real(dp), intent(in) :: a,b
    call require(b/=0,'NMF MU denominator is zero')
    if(abs(b)<1) call require(abs(a)<=limit*abs(b),'NMF MU ratio exceeds working range')
    c=a/b
  end function
  impure elemental real(dp) function power(a,b) result(c)
    real(dp), intent(in) :: a,b
    real(dp) :: exponent,log_a
    if(a==0) then
      call require(b>=0,'NMF MU negative power of zero exceeds working range')
      c=0;if(b==0) c=1
      return
    end if
    log_a=log(a)
    if(abs(b)>1) call require(abs(log_a)<=huge(1._dp)/abs(b),'NMF MU exponent exceeds working range')
    exponent=b*log_a
    call require(exponent<=log(limit),'NMF MU power exceeds working range')
    ! Let pow round subnormal results; clipping at the smallest positive value
    ! would erase values that round *up* from below that threshold.
    c=a**b
  end function
  real(dp) function total(a) result(c)
    real(dp), intent(in) :: a(:,:)
    call require(maxval(abs(a))<=limit/real(size(a),dp),'NMF MU sum exceeds working range')
    c=sum(a)
  end function
  function dot(a,b) result(c)
    real(dp), intent(in) :: a(:,:),b(:,:)
    real(dp), allocatable :: c(:,:)
    real(dp) :: ma,mb
    ma=maxval(a);mb=maxval(b)
    if(ma>0.and.mb>0) call require(log(ma)+log(mb)+log(real(size(a,2),dp))<=log(limit), &
      'NMF MU matrix product exceeds working range')
    c=matmul(a,b)
  end function
  function add(a,b) result(c)
    real(dp), intent(in) :: a(:,:),b(:,:)
    real(dp), allocatable :: c(:,:)
    call require(maxval(abs(a))<=limit-maxval(abs(b)),'NMF MU addition exceeds working range')
    c=a+b
  end function
  subroutine update(x,w,h,beta,gamma,l1,l2,hidden,cache)
    real(dp), intent(in) :: x(:,:),beta,gamma,l1,l2
    real(dp), intent(inout) :: w(:,:),h(:,:)
    logical, intent(in) :: hidden
    real(dp), allocatable, intent(inout) :: cache(:,:)
    real(dp), allocatable :: numerator(:,:),denominator(:,:),prediction(:,:),safe(:,:),base(:,:),ratio(:,:),factor(:,:)
    integer :: n,p,k
    n=size(x,1);p=size(x,2);k=size(h,1)
    call require(maxval(w)<=limit/real(size(w),dp).and.maxval(h)<=limit/real(size(h),dp), &
      'NMF MU factor sums exceed working range')
    if(beta==2) then
      if(hidden) then
        numerator=dot(transpose(w),x)
        if(real(k,dp)*k*(real(n,dp)+p)<2._dp*n*k*p) then
          denominator=dot(dot(transpose(w),w),h)
        else
          denominator=dot(transpose(w),dot(w,h))
        end if
      else
        numerator=dot(x,transpose(h));denominator=dot(w,dot(h,transpose(h)))
      end if
    else
      prediction=dot(w,h);safe=prediction;base=prediction
      if(beta<2) safe=max(safe,eps32)
      if(beta<1) base=max(base,eps32)
      if(beta==1) then
        ratio=divide(x,safe)
      else if(beta==0) then
        ratio=multiply(power(divide(1._dp,safe),2._dp),x)
      else
        ratio=multiply(power(safe,beta-2),x)
      end if
      if(hidden) then
        numerator=dot(transpose(w),ratio)
        if(beta==1) then
          denominator=spread(sum(w,dim=1),2,p)
          where(denominator==0) denominator=1
        else
          denominator=dot(transpose(w),power(base,beta-1))
        end if
      else
        numerator=dot(ratio,transpose(h))
        if(beta==1) then
          if(allocated(cache)) then
            denominator=cache
          else
            denominator=spread(sum(h,dim=2),1,n)
          end if
        else
          denominator=dot(power(base,beta-1),transpose(h))
        end if
      end if
    end if
    if(hidden) then
      factor=h
    else
      factor=w
    end if
    if(l1>0) denominator=add(denominator,spread(spread(l1,1,size(denominator,1)),2,size(denominator,2)))
    ! Source's fixed-H KL denominator is a view of its cached H_sum. In-place
    ! L1 addition therefore accumulates across transform sweeps in sklearn 1.6.1.
    if(beta==1.and..not.hidden) cache=denominator
    if(l2>0) denominator=add(denominator,multiply(l2,factor))
    where(denominator==0) denominator=eps32
    if(beta==1.and..not.hidden.and.l2==0) cache=denominator
    ratio=divide(numerator,denominator)
    if(gamma/=1) ratio=power(ratio,gamma)
    factor=multiply(factor,ratio)
    call require(maxval(factor)<=limit/real(size(factor),dp),'NMF MU factor sums exceed working range')
    if(hidden) then
      h=factor
      if(beta<=1) where(h<epsilon(1._dp)) h=0
    else
      w=factor
      if(beta<1) where(w<epsilon(1._dp)) w=0
    end if
  end subroutine
  real(dp) function error(x,w,h,beta) result(value)
    real(dp), intent(in) :: x(:,:),w(:,:),h(:,:),beta
    real(dp), allocatable :: prediction(:,:),a(:,:),b(:,:),c(:,:),terms(:,:)
    real(dp) :: estimated,ratio,divergence,mixed,sum_x,sum_prediction
    integer :: i,j
    call require(maxval(w)<=limit/real(size(w),dp).and.maxval(h)<=limit/real(size(h),dp), &
      'NMF MU factor sums exceed working range')
    prediction=dot(w,h)
    if(beta==2) then
      terms=x-prediction;value=sqrt(total(multiply(terms,terms)));return
    end if
    allocate(a(size(x,1),size(x,2)),b(size(x,1),size(x,2)),c(size(x,1),size(x,2)));a=0;b=0;c=0
    do j=1,size(x,2)
      do i=1,size(x,1)
        if(x(i,j)<=eps32) cycle
        estimated=max(prediction(i,j),eps32)
        if(beta==0.or.beta==1) then
          ratio=divide(x(i,j),estimated)
          call require(ratio>0,'NMF MU divergence ratio underflows working range')
          if(beta==1) then
            a(i,j)=multiply(x(i,j),log(ratio));b(i,j)=x(i,j)
          else
            a(i,j)=ratio;b(i,j)=log(ratio)
          end if
        else
          a(i,j)=power(x(i,j),beta);b(i,j)=multiply(x(i,j),power(estimated,beta-1))
        end if
      end do
    end do
    if(beta==1) then
      sum_prediction=total(multiply(reshape(sum(w,dim=1),[size(w,2),1]),reshape(sum(h,dim=2),[size(h,1),1])))
      divergence=total(reshape([total(a),sum_prediction,-total(b)],[3,1]))
    else if(beta==0) then
      divergence=total(reshape([total(a),-real(size(x),dp),-total(b)],[3,1]))
    else
      sum_x=total(a);mixed=total(b);sum_prediction=total(power(prediction,beta))
      divergence=total(reshape([sum_x,-multiply(beta,mixed)],[2,1]))
      divergence=total(reshape([divergence,multiply(beta-1,sum_prediction)],[2,1]))
      divergence=divide(divergence,multiply(beta,beta-1))
    end if
    value=sqrt(multiply(2._dp,max(divergence,0._dp)))
  end function
  subroutine nmf_mu_solve_dense(x,w,h,beta,steps,tol,l1w,l2w,l1h,l2h,update_h,n_iter,converged,initial,last,previous)
    real(dp), intent(in) :: x(:,:),beta,tol,l1w,l2w,l1h,l2h
    real(dp), intent(inout) :: w(:,:),h(:,:)
    integer, intent(in) :: steps
    logical, intent(in) :: update_h
    integer, intent(out) :: n_iter
    logical, intent(out) :: converged
    real(dp), intent(out) :: initial,last,previous
    real(dp) :: gamma,change
    real(dp), allocatable :: cache(:,:)
    integer :: it
    call require(all(ieee_is_finite(w)).and.all(ieee_is_finite(h)),'NMF MU factors must be finite')
    call require(all(w>=0).and.all(h>=0),'NMF MU factors must be nonnegative')
    gamma=1
    if(beta<1) gamma=1/(2-beta)
    if(beta>2) gamma=1/(beta-1)
    initial=error(x,w,h,beta);previous=initial;converged=.false.
    do it=1,steps
      call update(x,w,h,beta,gamma,l1w,l2w,.false.,cache)
      if(update_h) then
        call update(x,w,h,beta,gamma,l1h,l2h,.true.,cache)
        if(allocated(cache)) deallocate(cache)
      end if
      if(tol>0.and.mod(it,10)==0) then
        last=error(x,w,h,beta);change=previous-last
        if(initial>0) then
          converged=divide(change,initial)<tol
        else
          converged=change<0 ! IEEE: negative/0 is -Inf; 0/0 is NaN.
        end if
        if(converged) exit
        previous=last
      end if
    end do
    n_iter=min(it,steps);last=error(x,w,h,beta)
  end subroutine
  function sparse_h_denominator(w,h)result(value)
    real(dp),intent(in)::w(:,:),h(:,:)
    real(dp),allocatable::value(:,:),column(:)
    integer::n,p,k,j
    n=size(w,1);p=size(h,2);k=size(h,1)
    if(real(k,dp)*k*(real(n,dp)+p)<2._dp*n*k*p)then
      value=dot(dot(transpose(w),w),h)
    else
      ! Source's right-associated chain would materialize n-by-p. Stream its
      ! columns instead; all persistent arrays have factor/Gram/CSR dimensions.
      allocate(value(k,p))
      do j=1,p
        column=matmul(w,h(:,j));value(:,j)=matmul(transpose(w),column)
      end do
    end if
  end function
  subroutine sparse_update(x,w,h,beta,gamma,l1,l2,hidden,cache)
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::beta,gamma,l1,l2
    real(dp),intent(inout)::w(:,:),h(:,:)
    logical,intent(in)::hidden
    real(dp),allocatable,intent(inout)::cache(:,:)
    real(dp),allocatable::numerator(:,:),denominator(:,:),factor(:,:),row(:),column(:),ratio(:,:)
    real(dp)::estimate,weighted
    integer::n,p,k,i,j,e
    n=x%n_rows;p=x%n_cols;k=size(h,1)
    call require(maxval(w)<=limit/real(size(w),dp).and.maxval(h)<=limit/real(size(h),dp), &
      'NMF MU factor sums exceed working range')
    if(beta==2)then
      if(hidden)then
        numerator=transpose(nmf_sparse_cross(x,w,.true.));denominator=sparse_h_denominator(w,h)
      else
        numerator=nmf_sparse_cross(x,transpose(h),.false.);denominator=dot(w,dot(h,transpose(h)))
      end if
    else
      if(hidden)then
        allocate(numerator(k,p));numerator=0
      else
        allocate(numerator(n,k));numerator=0
      end if
      do i=1,n
        do e=x%indptr(i),x%indptr(i+1)-1
          j=x%indices(e);estimate=sum(multiply(w(i,:),h(:,j)))
          if(beta<2)estimate=max(estimate,eps32)
          if(beta==1)then
            weighted=divide(x%data(e),estimate)
          else if(beta==0)then
            weighted=multiply(power(divide(1._dp,estimate),2._dp),x%data(e))
          else
            weighted=multiply(power(estimate,beta-2),x%data(e))
          end if
          if(hidden)then
            numerator(:,j)=numerator(:,j)+multiply(weighted,w(i,:))
          else
            numerator(i,:)=numerator(i,:)+multiply(weighted,h(:,j))
          end if
        end do
      end do
      if(beta==1)then
        if(hidden)then
          column=sum(w,dim=1);where(column==0)column=1
          denominator=spread(column,2,p)
        else
          if(allocated(cache))then
            denominator=cache
          else
            denominator=spread(sum(h,dim=2),1,n)
          end if
        end if
      else if(hidden)then
        allocate(denominator(k,p))
        do j=1,p
          column=matmul(w,h(:,j));if(beta<1)column=max(column,eps32)
          denominator(:,j)=matmul(transpose(w),power(column,beta-1))
        end do
      else
        allocate(denominator(n,k))
        do i=1,n
          row=matmul(w(i,:),h);if(beta<1)row=max(row,eps32)
          denominator(i,:)=matmul(power(row,beta-1),transpose(h))
        end do
      end if
    end if
    if(hidden)then
      factor=h
    else
      factor=w
    end if
    if(l1>0)denominator=add(denominator,spread(spread(l1,1,size(denominator,1)),2,size(denominator,2)))
    if(beta==1.and..not.hidden)cache=denominator
    if(l2>0)denominator=add(denominator,multiply(l2,factor))
    where(denominator==0)denominator=eps32
    if(beta==1.and..not.hidden.and.l2==0)cache=denominator
    ratio=divide(numerator,denominator)
    if(gamma/=1)ratio=power(ratio,gamma)
    factor=multiply(factor,ratio)
    call require(maxval(factor)<=limit/real(size(factor),dp),'NMF MU factor sums exceed working range')
    if(hidden)then
      h=factor
      if(beta<=1)where(h<epsilon(1._dp))h=0
    else
      w=factor
      if(beta<1)where(w<epsilon(1._dp))w=0
    end if
  end subroutine
  real(dp)function nmf_sparse_error(x,w,h,beta)result(value)
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::w(:,:),h(:,:),beta
    real(dp),allocatable::cross(:,:),column(:),wh(:,:)
    real(dp)::estimate,ratio,first,mixed,prediction,divergence
    integer::i,j,e
    if(beta==2)then
      cross=nmf_sparse_cross(x,transpose(h),.false.);wh=sparse_h_denominator(w,h)
      divergence=sum(multiply(x%data,x%data))+total(multiply(wh,h))-2*total(multiply(cross,w))
      if(divergence<0)then
        value=ieee_value(0._dp,ieee_quiet_nan)
      else
        value=sqrt(divergence)
      end if
      return
    end if
    first=0;mixed=0
    do i=1,x%n_rows
      do e=x%indptr(i),x%indptr(i+1)-1
        if(x%data(e)<=eps32)cycle
        j=x%indices(e);estimate=max(sum(multiply(w(i,:),h(:,j))),eps32)
        if(beta==0.or.beta==1)then
          ratio=divide(x%data(e),estimate)
          call require(ratio>0,'NMF MU divergence ratio underflows working range')
          if(beta==1)then
            first=first+multiply(x%data(e),log(ratio));mixed=mixed+x%data(e)
          else
            first=first+ratio;mixed=mixed+log(ratio)
          end if
        else
          first=first+power(x%data(e),beta);mixed=mixed+multiply(x%data(e),power(estimate,beta-1))
        end if
      end do
    end do
    if(beta==1)then
      prediction=sum(multiply(sum(w,dim=1),sum(h,dim=2)));divergence=first+prediction-mixed
    else if(beta==0)then
      divergence=first-real(x%n_rows,dp)*x%n_cols-mixed
    else
      prediction=0
      do j=1,x%n_cols
        column=matmul(w,h(:,j));prediction=prediction+sum(power(column,beta))
      end do
      divergence=divide(first-multiply(beta,mixed)+multiply(beta-1,prediction),multiply(beta,beta-1))
    end if
    value=sqrt(multiply(2._dp,max(divergence,0._dp)))
  end function
  subroutine nmf_mu_solve_sparse(x,w,h,beta,steps,tol,l1w,l2w,l1h,l2h,update_h,n_iter,converged,initial,last,previous)
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::beta,tol,l1w,l2w,l1h,l2h
    real(dp),intent(inout)::w(:,:),h(:,:)
    integer,intent(in)::steps
    logical,intent(in)::update_h
    integer,intent(out)::n_iter
    logical,intent(out)::converged
    real(dp),intent(out)::initial,last,previous
    real(dp),allocatable::cache(:,:)
    real(dp)::gamma,change
    integer::it
    call require(all(ieee_is_finite(w)).and.all(ieee_is_finite(h)),'NMF MU factors must be finite')
    call require(all(w>=0).and.all(h>=0),'NMF MU factors must be nonnegative')
    gamma=1
    if(beta<1)gamma=1/(2-beta)
    if(beta>2)gamma=1/(beta-1)
    initial=nmf_sparse_error(x,w,h,beta);previous=initial;converged=.false.
    do it=1,steps
      call sparse_update(x,w,h,beta,gamma,l1w,l2w,.false.,cache)
      if(update_h)then
        call sparse_update(x,w,h,beta,gamma,l1h,l2h,.true.,cache)
        if(allocated(cache))deallocate(cache)
      end if
      if(tol>0.and.mod(it,10)==0)then
        last=nmf_sparse_error(x,w,h,beta);change=previous-last
        if(initial>0)then
          converged=divide(change,initial)<tol
        else
          converged=change<0
        end if
        if(converged)exit
        previous=last
      end if
    end do
    n_iter=min(it,steps);last=nmf_sparse_error(x,w,h,beta)
  end subroutine
end module
