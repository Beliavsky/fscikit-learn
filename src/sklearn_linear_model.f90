module sklearn_linear_model
  use sklearn_common
  use sklearn_metrics, only: r2_score,accuracy_score
  implicit none
  private
  public :: linear_regression, ridge, lasso, elastic_net, logistic_regression, perceptron
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
contains
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
