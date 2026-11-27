! ARDRegression update order, pruning, scores and Woodbury path follow
! sklearn 1.6.1 linear_model/_bayes.py (BSD-3-Clause).
! Native Hermitian pseudoinverse and numerical integration are MIT.
module sklearn_ard_core
  use sklearn_common,only:dp,require,check_xy,check_x,symmetric_eigen
  use sklearn_boosting_loss,only:numpy_sum=>boosting_sum
  use,intrinsic::ieee_arithmetic
  use,intrinsic::ieee_exceptions
  implicit none
  private
  public::ard_solve
contains
  function hermitian_inverse(a)result(b)
    real(dp),intent(in)::a(:,:)
    real(dp)::b(size(a,1),size(a,1)),w(size(a,1),size(a,1)),v(size(a,1),size(a,1)),e(size(a,1)),cut
    integer::i,j,n
    n=size(a,1);b=0
    if(n==0)return
    call require(all(ieee_is_finite(a)),'ARD posterior precision must be finite')
    ! scipy.linalg.pinvh uses the lower triangle and n*eps relative cutoff.
    do j=1,n
      do i=j,n
        w(i,j)=a(i,j);w(j,i)=a(i,j)
      end do
    end do
    call symmetric_eigen(w,e,v);cut=n*epsilon(1._dp)*maxval(abs(e))
    do i=1,n
      if(abs(e(i))>cut)b=b+spread(v(:,i),2,n)*spread(v(:,i),1,n)/e(i)
    end do
  end function
  real(dp)function logdet(a)result(value)
    real(dp),intent(in)::a(:,:)
    real(dp)::w(size(a,1),size(a,1)),row(size(a,1)),pivot
    integer::i,j,k,n,sign_det
    n=size(a,1);w=a;value=0;sign_det=1
    do k=1,n
      i=k-1+maxloc(abs(w(k:,k)),dim=1)
      if(w(i,k)==0)then
        value=ieee_value(0._dp,ieee_negative_inf);return
      end if
      if(i/=k)then
        row=w(i,:);w(i,:)=w(k,:);w(k,:)=row;sign_det=-sign_det
      end if
      pivot=w(k,k);if(pivot<0)sign_det=-sign_det
      value=value+log(abs(pivot))
      do j=k+1,n
        w(j,k+1:)=w(j,k+1:)-w(j,k)/pivot*w(k,k+1:)
      end do
    end do
    if(sign_det<0)value=ieee_value(0._dp,ieee_negative_inf)
  end function
  subroutine ard_solve(x,y,fit_intercept,max_iter,tol,a1,a2,l1,l2,threshold,compute_score, &
    coef,intercept,alpha,lambda,sigma,scores,offset,n_iter,rank,converged)
    real(dp),intent(in)::x(:,:),y(:),tol,a1,a2,l1,l2,threshold
    logical,intent(in)::fit_intercept,compute_score
    integer,intent(in)::max_iter
    real(dp),allocatable,intent(out)::coef(:),lambda(:),sigma(:,:),scores(:),offset(:)
    real(dp),intent(out)::intercept,alpha
    integer,intent(out)::n_iter,rank
    logical,intent(out)::converged
    real(dp),allocatable::xc(:,:),yc(:),old(:),gamma(:),history(:),a(:,:),z(:,:),inverse_lambda(:)
    integer,allocatable::keep(:)
    real(dp)::mean_y,rss,value
    integer::n,p,i,j,it
    logical::halting(size(ieee_all)),raised(size(ieee_all))
    call check_xy(x,size(y));call check_x(reshape(y,[size(y),1]))
    call require(size(y)>=2,'ARDRegression requires at least two samples')
    call require(all(ieee_is_finite([tol,a1,a2,l1,l2,threshold])),'ARD controls must be finite')
    call require(max_iter>0.and.tol>=0.and.threshold>=0,'invalid ARD parameters')
    call require(min(a1,a2,l1,l2)>=0,'ARD hyperpriors must be nonnegative')
    n=size(y);p=size(x,2);allocate(xc(n,p),yc(n),coef(p),lambda(p),offset(p),old(p),history(max_iter))
    offset=0;mean_y=0
    if(fit_intercept)then
      do j=1,p
        offset(j)=numpy_sum(x(:,j))/n
      end do
      mean_y=numpy_sum(y)/n
    end if
    xc=x-spread(offset,1,n);yc=y-mean_y;coef=0;lambda=1;old=0;keep=[(j,j=1,p)]
    do i=1,size(ieee_all)
      call ieee_get_flag(ieee_all(i),raised(i));call ieee_get_halting_mode(ieee_all(i),halting(i))
    end do
    do i=1,size(ieee_all)
      call ieee_set_halting_mode(ieee_all(i),.false.)
    end do
    alpha=1/(numpy_sum((yc-numpy_sum(yc)/n)**2)/n+epsilon(1._dp));converged=.false.
    do it=1,max_iter
      call posterior()
      rss=numpy_sum((yc-matmul(xc,coef))**2)
      gamma=1-lambda(keep)*[(sigma(j,j),j=1,size(keep))]
      lambda(keep)=(gamma+2*l1)/(coef(keep)**2+2*l2)
      alpha=(n-numpy_sum(gamma)+2*a1)/(rss+2*a2)
      keep=pack([(j,j=1,p)],lambda<threshold)
      where(.not.(lambda<threshold))coef=0
      if(compute_score)then
        call require(alpha>0,'ARD log score requires positive noise precision')
        value=numpy_sum(l1*log(lambda)-l2*lambda)+a1*log(alpha)-a2*alpha
        value=value+0.5_dp*(logdet(sigma)+n*log(alpha)+numpy_sum(log(lambda)))
        history(it)=value-0.5_dp*(alpha*rss+numpy_sum(lambda*coef**2))
      end if
      n_iter=it
      if(it>1.and.numpy_sum(abs(old-coef))<tol)then
        converged=.true.;exit
      end if
      old=coef
      if(size(keep)==0)then
        converged=.true.;exit
      end if
    end do
    if(size(keep)>0)then
      call posterior()
    else
      sigma=reshape([real(dp)::],[0,0])
    end if
    if(compute_score)then
      scores=history(:n_iter)
    else
      allocate(scores(0))
    end if
    intercept=mean_y-dot_product(offset,coef);rank=size(keep)
    do i=1,size(ieee_all)
      call ieee_set_halting_mode(ieee_all(i),halting(i))
    end do
    do i=1,size(ieee_all)
      call ieee_set_flag(ieee_all(i),raised(i))
    end do
  contains
    subroutine posterior()
      integer::q
      q=size(keep)
      if(n>=p)then
        a=alpha*matmul(transpose(xc(:,keep)),xc(:,keep))
        do j=1,q
          a(j,j)=a(j,j)+lambda(keep(j))
        end do
        sigma=hermitian_inverse(a)
      else
        inverse_lambda=1/lambda(keep);z=xc(:,keep)*spread(inverse_lambda,1,n)
        a=matmul(z,transpose(xc(:,keep)))
        do j=1,n
          a(j,j)=a(j,j)+1/alpha
        end do
        sigma=-matmul(transpose(z),matmul(hermitian_inverse(a),z))
        do j=1,q
          sigma(j,j)=sigma(j,j)+inverse_lambda(j)
        end do
      end if
      coef(keep)=alpha*matmul(sigma,matmul(transpose(xc(:,keep)),yc))
    end subroutine
  end subroutine
end module
