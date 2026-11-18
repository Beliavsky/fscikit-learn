module sklearn_linear_model
  use sklearn_common
  use sklearn_metrics, only: r2_score,accuracy_score
  implicit none
  private
  public :: linear_regression, ridge, lasso, elastic_net, logistic_regression, perceptron
  public :: bayesian_ridge, ard_regression, orthogonal_matching_pursuit
  type :: linear_regression
    logical :: fit_intercept=.true.
    real(dp), allocatable :: coef_(:)
    real(dp) :: intercept_=0
    integer :: rank_=0
  contains
    procedure :: fit=>ols_fit
    procedure :: predict=>linear_predict
    procedure :: score=>linear_score
  end type
  type, extends(linear_regression) :: ridge
    real(dp) :: alpha=1
  contains
    procedure :: fit=>ridge_fit
  end type
  type, extends(linear_regression) :: elastic_net
    real(dp) :: alpha=1, l1_ratio=0.5_dp,tol=1.e-8_dp
    integer :: max_iter=10000,n_iter_=0
    logical :: converged_=.false.
  contains
    procedure :: fit=>elastic_fit
  end type
  type, extends(elastic_net) :: lasso
  contains
    procedure :: fit=>lasso_fit
  end type
  type :: logistic_regression
    logical :: fit_intercept=.true., converged_=.false.
    real(dp) :: c=1, tol=1.e-8_dp
    integer :: max_iter=200,n_iter_=0
    integer, allocatable :: classes_(:)
    real(dp), allocatable :: coef_(:,:),intercept_(:)
  contains
    procedure :: fit=>logistic_fit
    procedure :: decision_function=>logistic_decision
    procedure :: predict_proba=>logistic_proba
    procedure :: predict=>logistic_predict
    procedure :: score=>logistic_score
  end type
  type :: perceptron
    logical :: fit_intercept=.true.
    integer :: max_iter=1000, n_iter_=0
    real(dp) :: eta0=1
    integer, allocatable :: classes_(:)
    real(dp), allocatable :: coef_(:,:),intercept_(:)
  contains
    procedure :: fit=>perceptron_fit
    procedure :: decision_function=>perceptron_decision
    procedure :: predict=>perceptron_predict
    procedure :: score=>perceptron_score
  end type
  type, extends(linear_regression) :: bayesian_ridge
    integer :: max_iter=300,n_iter_=0
    real(dp) :: tol=1.e-3_dp,alpha_1=1.e-6_dp,alpha_2=1.e-6_dp
    real(dp) :: lambda_1=1.e-6_dp,lambda_2=1.e-6_dp,alpha_init=-1,lambda_init=1
    real(dp) :: alpha_=0,lambda_=0
    logical :: converged_=.false.
    real(dp), allocatable :: sigma_(:,:)
  contains
    procedure :: fit=>bayesian_fit
    procedure :: predict_std=>bayesian_std
  end type
  type, extends(linear_regression) :: ard_regression
    integer :: max_iter=300,n_iter_=0
    real(dp) :: tol=1.e-3_dp,alpha_1=1.e-6_dp,alpha_2=1.e-6_dp
    real(dp) :: lambda_1=1.e-6_dp,lambda_2=1.e-6_dp,threshold_lambda=1.e4_dp,alpha_=0
    logical :: converged_=.false.
    real(dp), allocatable :: lambda_(:),sigma_(:,:)
  contains
    procedure :: fit=>ard_fit
    procedure :: predict_std=>ard_std
  end type
  type, extends(linear_regression) :: orthogonal_matching_pursuit
    integer :: n_nonzero_coefs=0,n_iter_=0
    real(dp) :: tol=-1
  contains
    procedure :: fit=>omp_fit
  end type
contains
  ! Bayesian precision updates and OMP selection follow sklearn 1.6.1
  ! linear_model/_bayes.py and _omp.py. See THIRD_PARTY_NOTICES.md.
  subroutine bayesian_fit(self,x,y)
    class(bayesian_ridge), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp) :: xc(size(x,1),size(x,2)),yc(size(y)),xm(size(x,2)),ym
    real(dp) :: u(size(x,1),size(x,2)),s(size(x,2)),v(size(x,2),size(x,2))
    real(dp) :: rhs(size(x,2)),old(size(x,2)),weights(size(x,2)),gamma,rss
    integer :: it,j,p,k
    call require(self%max_iter>0.and.self%tol>=0,'invalid BayesianRidge iteration parameters')
    call require(min(self%alpha_1,self%alpha_2,self%lambda_1,self%lambda_2)>0, &
      'BayesianRidge hyperpriors must be positive')
    call require((self%alpha_init==-1.or.self%alpha_init>0).and.self%lambda_init>0, &
      'invalid BayesianRidge initial precision')
    call centered(x,y,self%fit_intercept,xc,yc,xm,ym)
    p=size(x,2); k=min(size(x,1),p)
    call dense_svd(xc,s,u,v); rhs=s*matmul(transpose(u),yc)
    self%rank_=count(s>epsilon(1._dp)*maxval(s)*max(size(x,1),p))
    self%alpha_=self%alpha_init
    if(self%alpha_init==-1) self%alpha_=1/(sum((yc-sum(yc)/size(yc))**2)/size(yc)+epsilon(1._dp))
    self%lambda_=self%lambda_init; old=0; self%converged_=.false.
    do it=1,self%max_iter
      self%coef_=matmul(v,rhs/(s*s+self%lambda_/self%alpha_))
      rss=sum((yc-matmul(xc,self%coef_))**2)
      gamma=sum(self%alpha_*s*s/(self%lambda_+self%alpha_*s*s))
      self%lambda_=(gamma+2*self%lambda_1)/(sum(self%coef_**2)+2*self%lambda_2)
      self%alpha_=(size(y)-gamma+2*self%alpha_1)/(rss+2*self%alpha_2)
      self%n_iter_=it
      if(it>1.and.sum(abs(old-self%coef_))<self%tol) then
        self%converged_=.true.; exit
      end if
      old=self%coef_
    end do
    self%coef_=matmul(v,rhs/(s*s+self%lambda_/self%alpha_))
    self%intercept_=ym-dot_product(xm,self%coef_)
    weights=0; weights(:k)=1/(self%alpha_*s(:k)**2+self%lambda_)
    self%sigma_=matmul(v*spread(weights,1,p),transpose(v))
  end subroutine
  function bayesian_std(self,x) result(std)
    class(bayesian_ridge), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: std(size(x,1))
    call require(allocated(self%sigma_),'BayesianRidge is not fitted')
    call check_xy(transpose(x),size(self%coef_))
    std=sqrt(max(0._dp,sum(matmul(x,self%sigma_)*x,dim=2))+1/self%alpha_)
  end function
  function positive_inverse(a) result(inv)
    real(dp), intent(in) :: a(:,:)
    real(dp) :: inv(size(a,1),size(a,2)),l(size(a,1),size(a,2)),e(size(a,1))
    integer :: j
    l=cholesky_factor(a)
    do j=1,size(a,1)
      e=0; e(j)=1; inv(:,j)=solve_upper(transpose(l),solve_lower(l,e))
    end do
  end function
  subroutine ard_posterior(x,y,alpha,lambda,keep,sigma,coef)
    real(dp), intent(in) :: x(:,:),y(:),alpha,lambda(:)
    integer, intent(in) :: keep(:)
    real(dp), allocatable, intent(out) :: sigma(:,:)
    real(dp), intent(out) :: coef(:)
    real(dp) :: a(size(keep),size(keep))
    integer :: j
    coef=0
    a=alpha*matmul(transpose(x(:,keep)),x(:,keep))
    do j=1,size(keep)
      a(j,j)=a(j,j)+lambda(keep(j))
    end do
    sigma=positive_inverse(a)
    coef(keep)=alpha*matmul(sigma,matmul(transpose(x(:,keep)),y))
  end subroutine
  subroutine ard_fit(self,x,y)
    class(ard_regression), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp) :: xc(size(x,1),size(x,2)),yc(size(y)),xm(size(x,2)),ym
    real(dp) :: old(size(x,2)),gamma(size(x,2)),rss
    integer, allocatable :: keep(:)
    integer :: it,j,p
    call require(self%max_iter>0.and.self%tol>=0.and.self%threshold_lambda>0,'invalid ARD parameters')
    call require(min(self%alpha_1,self%alpha_2,self%lambda_1,self%lambda_2)>0,'ARD hyperpriors must be positive')
    call centered(x,y,self%fit_intercept,xc,yc,xm,ym); p=size(x,2)
    self%lambda_=spread(1._dp,1,p); self%coef_=spread(0._dp,1,p)
    self%alpha_=1/(sum((yc-sum(yc)/size(yc))**2)/size(yc)+epsilon(1._dp))
    keep=[(j,j=1,p)]; old=0; self%converged_=.false.
    do it=1,self%max_iter
      call ard_posterior(xc,yc,self%alpha_,self%lambda_,keep,self%sigma_,self%coef_)
      rss=sum((yc-matmul(xc,self%coef_))**2)
      do j=1,size(keep)
        gamma(j)=1-self%lambda_(keep(j))*self%sigma_(j,j)
      end do
      self%lambda_(keep)=(gamma(:size(keep))+2*self%lambda_1)/(self%coef_(keep)**2+2*self%lambda_2)
      self%alpha_=(size(y)-sum(gamma(:size(keep)))+2*self%alpha_1)/(rss+2*self%alpha_2)
      keep=pack([(j,j=1,p)],self%lambda_<self%threshold_lambda)
      where(self%lambda_>=self%threshold_lambda) self%coef_=0
      self%n_iter_=it
      if((it>1.and.sum(abs(old-self%coef_))<self%tol).or.size(keep)==0) then
        self%converged_=.true.; exit
      end if
      old=self%coef_
    end do
    call ard_posterior(xc,yc,self%alpha_,self%lambda_,keep,self%sigma_,self%coef_)
    self%intercept_=ym-dot_product(xm,self%coef_); self%rank_=size(keep)
  end subroutine
  function ard_std(self,x) result(std)
    class(ard_regression), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: std(size(x,1))
    integer, allocatable :: keep(:)
    integer :: j
    call require(allocated(self%sigma_),'ARDRegression is not fitted')
    call check_xy(transpose(x),size(self%coef_))
    keep=pack([(j,j=1,size(self%coef_))],self%lambda_<self%threshold_lambda)
    std=sqrt(max(0._dp,sum(matmul(x(:,keep),self%sigma_)*x(:,keep),dim=2))+1/self%alpha_)
  end function
  subroutine omp_fit(self,x,y)
    class(orthogonal_matching_pursuit), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp) :: xc(size(x,1),size(x,2)),yc(size(y)),xm(size(x,2)),ym
    real(dp) :: residual(size(y)),corr(size(x,2))
    real(dp), allocatable :: s(:),u(:,:),v(:,:),c(:)
    integer :: p,k,j,limit,idx(size(x,2))
    call centered(x,y,self%fit_intercept,xc,yc,xm,ym); p=size(x,2)
    call require(self%n_nonzero_coefs>=0.and.self%n_nonzero_coefs<=p,'invalid OMP sparsity')
    call require(self%tol>=0.or.self%tol==-1,'OMP tol must be nonnegative or -1')
    limit=self%n_nonzero_coefs
    if(limit==0) limit=max(1,p/10)
    if(self%tol>=0) limit=p
    residual=yc; self%coef_=spread(0._dp,1,p); self%n_iter_=0
    do k=1,limit
      corr=matmul(transpose(xc),residual); j=maxloc(abs(corr),dim=1)
      if(any(idx(:k-1)==j).or.corr(j)**2<epsilon(1._dp)) exit
      idx(k)=j
      if(allocated(s)) deallocate(s,u,v,c)
      allocate(s(k),u(size(y),k),v(k,k),c(k))
      call dense_svd(xc(:,idx(:k)),s,u,v)
      if(minval(s)<=sqrt(epsilon(1._dp))*maxval(s)) exit
      c=matmul(v,matmul(transpose(u),yc)/s)
      self%coef_=0; self%coef_(idx(:k))=c
      residual=yc-matmul(xc,self%coef_); self%n_iter_=k
      if(self%tol>=0.and.sum(residual**2)<=self%tol) exit
    end do
    self%rank_=self%n_iter_; self%intercept_=ym-dot_product(xm,self%coef_)
  end subroutine
  subroutine centered(x,y,fit_intercept,xc,yc,xmean,ymean)
    real(dp), intent(in) :: x(:,:),y(:)
    logical, intent(in) :: fit_intercept
    real(dp), intent(out) :: xc(:,:),yc(:),xmean(:),ymean
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    xmean=0; ymean=0
    if(fit_intercept) then
      xmean=sum(x,dim=1)/size(x,1); ymean=sum(y)/size(y)
    end if
    xc=x-spread(xmean,1,size(x,1)); yc=y-ymean
  end subroutine
  subroutine spectral_fit(self,x,y,alpha)
    class(linear_regression), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:),alpha
    real(dp) :: xc(size(x,1),size(x,2)),yc(size(y)),xm(size(x,2)),ym
    real(dp) :: u(size(x,1),size(x,2)),vals(size(x,2)),vecs(size(x,2),size(x,2))
    real(dp) :: rhs(size(x,2)),projection(size(x,2)),cutoff
    integer :: j,p
    call require(alpha>=0,'alpha must be nonnegative'); p=size(x,2)
    call centered(x,y,self%fit_intercept,xc,yc,xm,ym)
    call dense_svd(xc,vals,u,vecs)
    cutoff=epsilon(1._dp)*max(size(x,1),p)*maxval(vals)
    self%rank_=count(vals>cutoff); projection=matmul(transpose(u),yc)
    do j=1,p
      if(vals(j)>cutoff.or.alpha>0) then
        projection(j)=projection(j)*vals(j)/(vals(j)**2+alpha)
      else
        projection(j)=0
      end if
    end do
    self%coef_=matmul(vecs,projection); self%intercept_=ym-dot_product(xm,self%coef_)
  end subroutine
  subroutine ols_fit(self,x,y)
    class(linear_regression), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    call spectral_fit(self,x,y,0._dp)
  end subroutine
  subroutine ridge_fit(self,x,y)
    class(ridge), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    call spectral_fit(self,x,y,self%alpha)
  end subroutine
  function linear_predict(self,x) result(y)
    class(linear_regression), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: y(size(x,1))
    call require(allocated(self%coef_),'linear model is not fitted')
    call check_xy(transpose(x),size(self%coef_))
    y=matmul(x,self%coef_)+self%intercept_
  end function
  real(dp) function linear_score(self,x,y) result(s)
    class(linear_regression), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    s=r2_score(y,self%predict(x))
  end function
  subroutine lasso_fit(self,x,y)
    class(lasso), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    self%l1_ratio=1
    call elastic_fit(self,x,y)
  end subroutine
  subroutine elastic_fit(self,x,y)
    class(elastic_net), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp) :: xc(size(x,1),size(x,2)),yc(size(y)),xm(size(x,2)),ym
    real(dp) :: residual(size(y)),old,update,rho,den,change
    integer :: it,j,n
    call require(self%alpha>=0.and.self%l1_ratio>=0.and.self%l1_ratio<=1,'invalid elastic net penalties')
    call require(self%max_iter>0.and.self%tol>0,'invalid elastic net iteration controls')
    call centered(x,y,self%fit_intercept,xc,yc,xm,ym); n=size(y)
    self%coef_=spread(0._dp,1,size(x,2)); residual=yc; self%converged_=.false.
    do it=1,self%max_iter
      change=0
      do j=1,size(x,2)
        old=self%coef_(j); residual=residual+xc(:,j)*old
        rho=dot_product(xc(:,j),residual)/n
        den=sum(xc(:,j)**2)/n+self%alpha*(1-self%l1_ratio)
        update=0
        if(den>0) update=sign(max(abs(rho)-self%alpha*self%l1_ratio,0._dp),rho)/den
        self%coef_(j)=update; residual=residual-xc(:,j)*update
        change=max(change,abs(update-old))
      end do
      if(change<self%tol*max(1._dp,maxval(abs(self%coef_)))) then
        self%converged_=.true.; exit
      end if
    end do
    self%n_iter_=min(it,self%max_iter); self%intercept_=ym-dot_product(xm,self%coef_)
  end subroutine
  subroutine logistic_fit(self,x,y)
    class(logistic_regression), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), allocatable :: design(:,:),w(:,:),prob(:,:),gradient(:,:),hess(:,:),step(:),trial(:,:)
    real(dp) :: objective,new_objective,rate,penalty,v
    integer :: n,p,k,q,it,i,j,a,b,ia,ib,target
    call check_xy(x,size(y)); self%classes_=unique_labels(y); k=size(self%classes_)
    call require(k>=2,'logistic regression requires at least two classes')
    call require(self%c>0.and.self%tol>0.and.self%max_iter>0,'invalid logistic controls')
    n=size(x,1); p=size(x,2); q=p+merge(1,0,self%fit_intercept)
    ! Binary has one logit; multiclass uses one logit per class and softmax.
    if(k==2) k=1
    allocate(design(n,q),w(q,k),prob(n,k),gradient(q,k),hess(q*k,q*k),step(q*k),trial(q,k))
    design(:,:p)=x; if(q>p) design(:,q)=1
    w=0; penalty=1/self%c; self%converged_=.false.
    do it=1,self%max_iter
      call evaluate(w,objective,prob)
      gradient=0; hess=0
      do i=1,n
        target=label_index(self%classes_,y(i))
        do a=1,k
          v=prob(i,a)
          if((k==1.and.target==2).or.(k>1.and.target==a)) v=v-1
          gradient(:,a)=gradient(:,a)+design(i,:)*v
          do b=1,k
            v=prob(i,a)*(merge(1._dp,0._dp,a==b)-prob(i,b))
            do j=1,q
              ia=(a-1)*q+j
              hess(ia,(b-1)*q+1:b*q)=hess(ia,(b-1)*q+1:b*q)+v*design(i,j)*design(i,:)
            end do
          end do
        end do
      end do
      gradient(:p,:)=gradient(:p,:)+penalty*w(:p,:)
      do a=1,k
        do j=1,q
          ia=(a-1)*q+j
          if(j<=p) hess(ia,ia)=hess(ia,ia)+penalty
          hess(ia,ia)=hess(ia,ia)+1.e-10_dp
        end do
      end do
      if(maxval(abs(gradient))<self%tol) then
        self%converged_=.true.; exit
      end if
      step=solve_system(hess,reshape(gradient,[q*k])); rate=1
      do j=1,50
        trial=w-rate*reshape(step,[q,k]); call evaluate(trial,new_objective,prob)
        if(new_objective<=objective-1.e-4_dp*rate*sum(reshape(gradient,[q*k])*step)) exit
        rate=rate/2
      end do
      w=trial
      if(maxval(abs(rate*step))<self%tol) then
        self%converged_=.true.; exit
      end if
    end do
    self%n_iter_=min(it,self%max_iter)
    self%coef_=transpose(w(:p,:)); self%intercept_=spread(0._dp,1,k)
    if(q>p) self%intercept_=w(q,:)
  contains
    subroutine evaluate(weights,loss,probabilities)
      real(dp), intent(in) :: weights(:,:)
      real(dp), intent(out) :: loss,probabilities(:,:)
      real(dp) :: logits(k),normalizer,z
      integer :: row,cl
      loss=0.5_dp*penalty*sum(weights(:p,:)**2)
      do row=1,n
        logits=matmul(design(row,:),weights)
        cl=label_index(self%classes_,y(row))
        if(k==1) then
          z=logits(1); probabilities(row,1)=sigmoid(z)
          loss=loss+max(z,0._dp)+log(1+exp(-abs(z)))
          if(cl==2) loss=loss-z
        else
          normalizer=logsumexp(logits); probabilities(row,:)=exp(logits-normalizer)
          loss=loss+normalizer-logits(cl)
        end if
      end do
    end subroutine
  end subroutine
  function logistic_decision(self,x) result(z)
    class(logistic_regression), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call require(allocated(self%coef_),'logistic_regression is not fitted')
    call check_xy(transpose(x),size(self%coef_,2))
    z=matmul(x,transpose(self%coef_))+spread(self%intercept_,1,size(x,1))
  end function
  function logistic_proba(self,x) result(prob)
    class(logistic_regression), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: prob(:,:),z(:,:)
    integer :: i,k
    z=self%decision_function(x); k=size(self%classes_); allocate(prob(size(x,1),k))
    if(k==2) then
      prob(:,2)=sigmoid(z(:,1)); prob(:,1)=1-prob(:,2)
    else
      do i=1,size(x,1)
        prob(i,:)=exp(z(i,:)-logsumexp(z(i,:)))
      end do
    end if
  end function
  function logistic_predict(self,x) result(y)
    class(logistic_regression), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: y(size(x,1)),i
    real(dp), allocatable :: prob(:,:)
    prob=self%predict_proba(x)
    do i=1,size(y)
      y(i)=self%classes_(maxloc(prob(i,:),dim=1))
    end do
  end function
  real(dp) function logistic_score(self,x,y) result(s)
    class(logistic_regression), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    s=accuracy_score(y,self%predict(x))
  end function
  subroutine perceptron_fit(self,x,y)
    class(perceptron), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp) :: target,margin
    integer :: it,i,c,k,errors
    call check_xy(x,size(y)); self%classes_=unique_labels(y); k=size(self%classes_)
    call require(k>=2.and.self%max_iter>0.and.self%eta0>0,'invalid perceptron controls')
    if(allocated(self%coef_)) deallocate(self%coef_,self%intercept_)
    allocate(self%coef_(k,size(x,2)),self%intercept_(k)); self%coef_=0; self%intercept_=0
    do it=1,self%max_iter
      errors=0
      do i=1,size(y)
        do c=1,k
          target=merge(1._dp,-1._dp,y(i)==self%classes_(c))
          margin=dot_product(self%coef_(c,:),x(i,:))+self%intercept_(c)
          if(target*margin<=0) then
            self%coef_(c,:)=self%coef_(c,:)+self%eta0*target*x(i,:)
            if(self%fit_intercept) self%intercept_(c)=self%intercept_(c)+self%eta0*target
            errors=errors+1
          end if
        end do
      end do
      if(errors==0) exit
    end do
    self%n_iter_=min(it,self%max_iter)
  end subroutine
  function perceptron_decision(self,x) result(z)
    class(perceptron), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call require(allocated(self%coef_),'perceptron is not fitted')
    call check_xy(transpose(x),size(self%coef_,2))
    z=matmul(x,transpose(self%coef_))+spread(self%intercept_,1,size(x,1))
  end function
  function perceptron_predict(self,x) result(y)
    class(perceptron), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: y(size(x,1)),i
    real(dp), allocatable :: z(:,:)
    z=self%decision_function(x)
    do i=1,size(y)
      y(i)=self%classes_(maxloc(z(i,:),dim=1))
    end do
  end function
  real(dp) function perceptron_score(self,x,y) result(s)
    class(perceptron), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    s=accuracy_score(y,self%predict(x))
  end function
end module
