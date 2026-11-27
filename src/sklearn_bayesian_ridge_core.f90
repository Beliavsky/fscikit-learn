! sklearn 1.6.1 linear_model/_bayes.py and _base.py (BSD-3-Clause).
! Native SVD integration, rank adapters and IEEE preservation are MIT.
module sklearn_bayesian_ridge_core
  use sklearn_common,only:dp,require,check_xy,check_x,dense_svd
  use sklearn_boosting_loss,only:numpy_sum=>boosting_sum
  use,intrinsic::ieee_arithmetic
  use,intrinsic::ieee_exceptions
  implicit none
  private
  public::bayesian_ridge_solve,bayesian_posterior_std
contains
  function bayesian_posterior_std(x,sigma,alpha)result(value)
    real(dp),intent(in)::x(:,:),sigma(:,:),alpha
    real(dp)::value(size(x,1))
    logical::halting(size(ieee_all)),raised(size(ieee_all))
    integer::i
    do i=1,size(ieee_all)
      call ieee_get_flag(ieee_all(i),raised(i));call ieee_get_halting_mode(ieee_all(i),halting(i))
    end do
    do i=1,size(ieee_all)
      call ieee_set_halting_mode(ieee_all(i),.false.)
    end do
    value=sqrt(sum(matmul(x,sigma)*x,dim=2)+1/alpha)
    do i=1,size(ieee_all)
      call ieee_set_halting_mode(ieee_all(i),halting(i))
    end do
    do i=1,size(ieee_all)
      call ieee_set_flag(ieee_all(i),raised(i))
    end do
  end function
  subroutine bayesian_ridge_solve(x,y,fit_intercept,max_iter,tol,a1,a2,l1,l2,alpha_init,lambda_init,compute_score, &
    coef,intercept,alpha,lambda,sigma,scores,x_offset,n_iter,rank,converged,sample_weight)
    real(dp),intent(in)::x(:,:),y(:),tol,a1,a2,l1,l2,alpha_init,lambda_init
    logical,intent(in)::fit_intercept,compute_score
    integer,intent(in)::max_iter
    real(dp),allocatable,intent(out)::coef(:),sigma(:,:),scores(:),x_offset(:)
    real(dp),intent(out)::intercept,alpha,lambda
    integer,intent(out)::n_iter,rank
    logical,intent(out)::converged
    real(dp),optional,intent(in)::sample_weight(:)
    real(dp),allocatable::xc(:,:),yc(:),s(:),u(:,:),v(:,:),eigen(:),old(:),xt_y(:),temporary(:),history(:)
    real(dp)::weight(size(y)),root_weight(size(y)),mean_y,mass,gamma,rss,variance,logdet,score
    integer::n,p,k,i,j,iteration
    logical::halting(size(ieee_all)),raised(size(ieee_all))
    call check_xy(x,size(y));call check_x(reshape(y,[size(y),1]))
    call require(all(ieee_is_finite([tol,a1,a2,l1,l2,alpha_init,lambda_init])),'BayesianRidge controls must be finite')
    call require(max_iter>0.and.tol>0,'invalid BayesianRidge iteration parameters')
    call require(min(a1,a2,l1,l2)>=0,'BayesianRidge hyperpriors must be nonnegative')
    call require((alpha_init==-1.or.alpha_init>=0).and.(lambda_init==-1.or.lambda_init>=0), &
      'invalid BayesianRidge initial precision')
    weight=1
    if(present(sample_weight))then
      call require(size(sample_weight)==size(y),'BayesianRidge sample_weight length mismatch');weight=sample_weight
    end if
    call require(all(ieee_is_finite(weight)),'BayesianRidge sample weights must be finite')
    call require(all(weight>=0),'BayesianRidge sample weights must be nonnegative')
    n=size(y);p=size(x,2);k=min(n,p);mass=numpy_sum(weight)
    if(fit_intercept)call require(mass>0,'BayesianRidge centering weights must have positive sum')
    allocate(xc(n,p),yc(n),x_offset(p),s(p),u(n,p),v(p,p),eigen(k),coef(p),old(p),xt_y(p))
    x_offset=0;mean_y=0
    if(fit_intercept)then
      do j=1,p
        x_offset(j)=numpy_sum(weight*x(:,j))/mass
      end do
      mean_y=numpy_sum(weight*y)/mass
    end if
    root_weight=sqrt(weight);xc=(x-spread(x_offset,1,n))*spread(root_weight,2,p);yc=(y-mean_y)*root_weight
    call dense_svd(xc,s,u,v);eigen=s(:k)**2
    rank=count(s>epsilon(1._dp)*max(n,p)*maxval(s));xt_y=matmul(transpose(xc),yc)
    do i=1,size(ieee_all)
      call ieee_get_flag(ieee_all(i),raised(i));call ieee_get_halting_mode(ieee_all(i),halting(i))
    end do
    do i=1,size(ieee_all)
      call ieee_set_halting_mode(ieee_all(i),.false.)
    end do
    variance=numpy_sum((yc-numpy_sum(yc)/n)**2)/n;alpha=alpha_init
    if(alpha_init==-1)alpha=1/(variance+epsilon(1._dp))
    lambda=lambda_init;if(lambda_init==-1)lambda=1
    allocate(history(merge(max_iter+1,0,compute_score)));old=0;converged=.false.;n_iter=0
    do iteration=1,max_iter
      call posterior()
      if(compute_score)history(iteration)=marginal()
      gamma=numpy_sum(alpha*eigen/(lambda+alpha*eigen))
      lambda=(gamma+2*l1)/(numpy_sum(coef**2)+2*l2)
      alpha=(n-gamma+2*a1)/(rss+2*a2)
      n_iter=iteration
      if(iteration>1.and.numpy_sum(abs(old-coef))<tol)then
        converged=.true.;exit
      end if
      old=coef
    end do
    call posterior()
    if(compute_score)then
      history(n_iter+1)=marginal();scores=history(:n_iter+1)
    else
      allocate(scores(0))
    end if
    sigma=matmul(v(:,:k)*spread((1/alpha)/(eigen+lambda/alpha),1,p),transpose(v(:,:k)))
    intercept=mean_y-dot_product(x_offset,coef)
    do i=1,size(ieee_all)
      call ieee_set_halting_mode(ieee_all(i),halting(i))
    end do
    do i=1,size(ieee_all)
      call ieee_set_flag(ieee_all(i),raised(i))
    end do
  contains
    subroutine posterior()
      if(n>p)then
        temporary=matmul(transpose(v),xt_y)/(eigen+lambda/alpha)
        coef=matmul(v,temporary)
      else
        temporary=matmul(transpose(u(:,:k)),yc)/(eigen+lambda/alpha)
        coef=matmul(transpose(xc),matmul(u(:,:k),temporary))
      end if
      rss=numpy_sum((yc-matmul(xc,coef))**2)
    end subroutine
    real(dp)function marginal()result(value)
      call require(alpha/=0.and.lambda/=0,'BayesianRidge log score requires nonzero precisions')
      logdet=-numpy_sum(log(lambda+alpha*eigen))
      if(p>n)logdet=logdet-(p-n)*log(lambda)
      value=l1*log(lambda)-l2*lambda+a1*log(alpha)-a2*alpha
      value=value+0.5_dp*(p*log(lambda)+n*log(alpha)-alpha*rss-lambda*numpy_sum(coef**2)+logdet &
        -n*log(2*acos(-1._dp)))
    end function
  end subroutine
end module
