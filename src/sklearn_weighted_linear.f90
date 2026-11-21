! Internal weighted linear fitting; sklearn linear_model/_base.py and _ridge.py (BSD).
! See THIRD_PARTY_NOTICES.md and LICENSES/scikit-learn-COPYING.
module sklearn_weighted_linear
  use sklearn_common, only: dp,require,dense_svd,check_xy,check_x
  use sklearn_linear_model, only: linear_regression
  use sklearn_ieee_math, only: ieee_difference,ieee_product,ieee_quotient,ieee_total,ieee_mean
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan
  implicit none
  private
  public :: weighted_linear,weighted_ridge_loo
contains
  subroutine weighted_linear(model,x,y,w,alpha)
    class(linear_regression), intent(inout) :: model
    real(dp), intent(in) :: x(:,:),y(:),w(:),alpha
    real(dp) :: xc(size(x,1),size(x,2)),yc(size(y)),xm(size(x,2)),ym
    real(dp) :: u(size(x,1),size(x,2)),s(size(x,2)),v(size(x,2),size(x,2)),projection(size(x,2))
    real(dp) :: mass,scale,cutoff,limit
    integer :: i,j
    call require(ieee_is_finite(alpha),'bagging ridge alpha must be finite')
    call require(alpha>=0,'bagging ridge alpha must be nonnegative')
    call require(all(w>=0),'bagging linear weights must be nonnegative')
    mass=ieee_total(w)
    call require(ieee_is_finite(mass),'bagging linear weight sum must be finite')
    xm=0; ym=0
    if(model%fit_intercept) then
      call require(mass>0,'bagging linear weights must have a positive sum')
      do j=1,size(x,2)
        xm(j)=ieee_mean(x(:,j),w)
      end do
      ym=ieee_mean(y,w)
    end if
    call require(all(ieee_is_finite(xm)).and.ieee_is_finite(ym),'bagging linear weighted centering overflow')
    do i=1,size(y)
      scale=sqrt(w(i)); yc(i)=ieee_product(ieee_difference(y(i),ym),scale)
      do j=1,size(x,2)
        xc(i,j)=ieee_product(ieee_difference(x(i,j),xm(j)),scale)
      end do
    end do
    call require(all(ieee_is_finite(xc)).and.all(ieee_is_finite(yc)),'bagging linear rescaling overflow')
    limit=sqrt(huge(1._dp))/sqrt(real(max(size(x,1),size(x,2)),dp))/4
    call require(maxval(abs(xc))<=limit.and.maxval(abs(yc))<=limit,'bagging linear design exceeds working range')
    call dense_svd(xc,s,u,v)
    cutoff=epsilon(1._dp)*max(size(x,1),size(x,2))*maxval(s)
    model%rank_=count(s>cutoff); projection=matmul(transpose(u),yc)
    do j=1,size(s)
      if(s(j)>cutoff.or.alpha>0) then
        projection(j)=ieee_quotient(ieee_product(projection(j),s(j)), &
          ieee_difference(ieee_product(s(j),s(j)),-alpha))
      else
        projection(j)=0
      end if
    end do
    model%coef_=matmul(v,projection); model%intercept_=ym-dot_product(xm,model%coef_)
  end subroutine

  subroutine weighted_ridge_loo(model,x,y,alphas,alpha_best,best_score,scores,sample_weight)
    class(linear_regression), intent(inout) :: model
    real(dp), intent(in) :: x(:,:),y(:),alphas(:)
    real(dp), intent(out) :: alpha_best,best_score
    real(dp), allocatable, intent(out) :: scores(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp) :: w(size(y)),a(size(x,1),size(x,2)),b(size(y)),xm(size(x,2)),ym,total,cutoff,score,r
    real(dp) :: s(size(x,2)),u(size(x,1),size(x,2)),v(size(x,2),size(x,2)),projection(size(x,2))
    real(dp) :: outside(size(y)),remainder(size(y)),inverse(size(x,2)),dual(size(y)),diagonal(size(y)),errors(size(y))
    integer :: i,j,best,rank_,positive
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    call require(size(alphas)>0,'RidgeCV needs candidate alphas')
    call require(all(ieee_is_finite(alphas)),'RidgeCV alphas must be finite')
    call require(all(alphas>0),'RidgeCV LOO alphas must be positive')
    w=1
    if(present(sample_weight)) then
      call require(size(sample_weight)==size(y),'RidgeCV sample weight length mismatch')
      call require(all(ieee_is_finite(sample_weight)),'RidgeCV sample weights must be finite')
      call require(all(sample_weight>=0),'RidgeCV sample weights must be nonnegative')
      w=sample_weight
    end if
    total=ieee_total(w); positive=count(w>0)
    call require(ieee_is_finite(total),'RidgeCV weight sum must be finite')
    xm=0; ym=0
    if(model%fit_intercept) then
      call require(total>0,'RidgeCV needs positive total weight')
      do j=1,size(x,2)
        xm(j)=ieee_mean(x(:,j),w)
      end do
      ym=ieee_mean(y,w)
    end if
    do i=1,size(y)
      b(i)=ieee_product(ieee_difference(y(i),ym),sqrt(w(i)))
      do j=1,size(x,2)
        a(i,j)=ieee_product(ieee_difference(x(i,j),xm(j)),sqrt(w(i)))
      end do
    end do
    call require(all(ieee_is_finite(a)).and.all(ieee_is_finite(b)),'RidgeCV weighted data overflow')
    call require(maxval(abs(a))<=sqrt(huge(1._dp))/sqrt(real(max(size(x,1),size(x,2)),dp))/4, &
      'RidgeCV design exceeds working range')
    call dense_svd(a,s,u,v); cutoff=epsilon(1._dp)*max(size(x,1),size(x,2))*maxval(s)
    rank_=count(s>cutoff)
    where(s<=cutoff) s=0
    projection=matmul(transpose(u),b); outside=b-matmul(u,projection); remainder=1-sum(u*u,dim=2)
    if(model%fit_intercept) remainder=remainder-w/total
    if(rank_==positive-merge(1,0,model%fit_intercept)) then
      outside=0; where(w>0) remainder=0
    end if
    remainder=max(0._dp,remainder); allocate(scores(size(alphas))); best=1
    do j=1,size(alphas)
      inverse=0
      where(s>0) inverse=1/(s*s+alphas(j))
      dual=outside/alphas(j)+matmul(u,projection*inverse)
      diagonal=remainder/alphas(j)+matmul(u*u,inverse)
      errors=0
      do i=1,size(y)
        if(w(i)==0) cycle
        r=ieee_quotient(dual(i),diagonal(i)); errors(i)=ieee_product(r,r)
      end do
      score=-ieee_mean(errors); scores(j)=score
      if(j==1) then
        best=1
      else if(.not.ieee_is_nan(score).and..not.ieee_is_nan(scores(best))) then
        if(score>scores(best)) best=j
      end if
    end do
    alpha_best=alphas(best); best_score=scores(best)
    call weighted_linear(model,x,y,w,alpha_best)
  end subroutine
end module
