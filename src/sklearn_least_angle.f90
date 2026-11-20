! Native dense LARS homotopy, adapted from sklearn 1.6.1 _least_angle.py.
! Upstream BSD-3-Clause terms are retained in THIRD_PARTY_NOTICES.md and LICENSES/.
module sklearn_least_angle
  use sklearn_common
  use sklearn_metrics, only: r2_score
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: lars,lasso_lars,lars_path,lars_path_result,lars_cv,lasso_lars_cv,lasso_lars_ic
  type :: lars_path_result
    real(dp), allocatable :: alphas(:),coefficients(:,:)
    integer, allocatable :: active(:)
    integer :: n_iter=0,n_skipped=0
  end type
  type :: lars
    integer :: n_nonzero_coefs=500,n_iter_=0,n_features_in_=0
    real(dp) :: eps=epsilon(1._dp),intercept_=0
    logical :: fit_intercept=.true.,fit_path=.true.
    real(dp), allocatable :: coef_(:),alphas_(:),coef_path_(:,:)
    integer, allocatable :: active_(:)
  contains
    procedure :: fit=>lars_fit
    procedure :: predict=>least_angle_predict
    procedure :: score=>least_angle_score
  end type
  type, extends(lars) :: lasso_lars
    real(dp) :: alpha=1
    integer :: max_iter=500
    logical :: positive=.false.
  contains
    procedure :: fit=>lasso_lars_fit
  end type
  type, extends(lars) :: lars_cv
    integer :: max_iter=500,cv=5,max_n_alphas=1000
    real(dp) :: alpha_=0
    real(dp), allocatable :: cv_alphas_(:),mse_path_(:,:)
  contains
    procedure :: fit=>lars_cv_fit
  end type
  type, extends(lars_cv) :: lasso_lars_cv
    logical :: positive=.false.
  contains
    procedure :: fit=>lasso_lars_cv_fit
  end type
  type, extends(lasso_lars) :: lasso_lars_ic
    character(3) :: criterion='aic'
    real(dp) :: noise_variance=-1,noise_variance_=0,alpha_=0
    real(dp), allocatable :: criterion_(:)
  contains
    procedure :: fit=>lasso_lars_ic_fit
  end type
  type :: cv_residual_path
    real(dp), allocatable :: alphas(:),residuals(:,:)
  end type

contains
  subroutine active_factor(gram,active,eps,l,okay)
    real(dp), intent(in) :: gram(:,:),eps
    integer, intent(in) :: active(:)
    real(dp), allocatable, intent(out) :: l(:,:)
    logical, intent(out) :: okay
    integer :: i,j,k
    real(dp) :: diagonal
    k=size(active); allocate(l(k,k)); l=0; okay=.true.
    do i=1,k
      do j=1,i-1
        l(i,j)=(gram(active(i),active(j))-dot_product(l(i,:j-1),l(j,:j-1)))/l(j,j)
      end do
      diagonal=gram(active(i),active(i))-sum(l(i,:i-1)**2)
      l(i,i)=max(sqrt(abs(diagonal)),eps)
      if(l(i,i)<1.e-7_dp) then
        okay=.false.; return
      end if
    end do
  end subroutine
  function lars_path(x,y,method,max_iter,alpha_min,eps,positive) result(path)
    real(dp), intent(in) :: x(:,:),y(:)
    character(*), intent(in), optional :: method
    integer, intent(in), optional :: max_iter
    real(dp), intent(in), optional :: alpha_min,eps
    logical, intent(in), optional :: positive
    type(lars_path_result) :: path
    real(dp), allocatable :: gram(:,:),original(:),cov(:),signs(:),l(:,:),w(:),corr(:),z(:)
    real(dp), allocatable :: coefficients(:,:),alphas(:),beta(:),old_beta(:)
    integer, allocatable :: active(:),inactive(:),dropped(:)
    logical, allocatable :: blocked(:),remove(:)
    integer :: n,p,limit,iteration,i,j,pick,key,k
    real(dp) :: target,regularizer,c,candidate,a,gamma,ratio,zmin,alpha,fraction,denominator
    real(dp), parameter :: tiny32=1.1754943508222875e-38_dp,equality_tol=1.1920928955078125e-7_dp
    character(8) :: kind
    logical :: constrained,drop,okay
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    n=size(x,1); p=size(x,2); limit=500; if(present(max_iter)) limit=max_iter
    target=0; if(present(alpha_min)) target=alpha_min
    regularizer=epsilon(1._dp); if(present(eps)) regularizer=eps
    kind='lar'
    if(present(method)) then
      call require(method=='lar'.or.method=='lasso','unknown LARS method')
      kind=method
    end if
    constrained=.false.; if(present(positive)) constrained=positive
    call require(limit>=0.and.limit<huge(limit),'invalid LARS max_iter')
    call require(all(ieee_is_finite([target,regularizer])),'nonfinite LARS controls')
    call require(target>=0.and.regularizer>=0,'LARS alpha_min and eps must be nonnegative')
    call require(kind=='lar'.or.kind=='lasso','unknown LARS method')
    call require(.not.(kind=='lar'.and.constrained),'positive constraint requires LARS method lasso')
    gram=matmul(transpose(x),x); original=matmul(transpose(x),y); cov=original
    allocate(coefficients(p,limit+1),alphas(limit+1),beta(p),old_beta(p),blocked(p))
    coefficients=0; alphas=0; beta=0; blocked=.false.
    active=[integer::]; signs=[real(dp)::]; inactive=[(i,i=1,p)]
    iteration=0; drop=.false.
    do
      pick=0; c=0
      do i=1,size(inactive)
        key=inactive(i); if(blocked(key)) cycle
        candidate=abs(cov(key)); if(constrained) candidate=cov(key)
        if(pick==0.or.candidate>c) then
          pick=i; c=candidate
        end if
      end do
      alpha=c/n; alphas(iteration+1)=alpha; coefficients(:,iteration+1)=beta
      if(alpha<=target+equality_tol) then
        if(abs(alpha-target)>equality_tol) then
          if(iteration>0) then
            fraction=(alphas(iteration)-target)/(alphas(iteration)-alpha)
            beta=coefficients(:,iteration)+fraction*(beta-coefficients(:,iteration))
            coefficients(:,iteration+1)=beta
          end if
          alphas(iteration+1)=target
        end if
        exit
      end if
      if(iteration>=limit.or.size(active)>=p.or.pick==0) exit
      if(.not.drop) then
        key=inactive(pick); inactive(pick)=inactive(1); inactive(1)=key
        call active_factor(gram,[active,key],regularizer,l,okay)
        if(.not.okay) then
          blocked(key)=.true.; cov(key)=0; path%n_skipped=path%n_skipped+1; cycle
        end if
        active=[active,key]; signs=[signs,merge(1._dp,sign(1._dp,cov(key)),constrained)]
        inactive=inactive(2:)
      else
        call active_factor(gram,active,regularizer,l,okay)
        call require(okay,'LARS active factor became degenerate')
      end if
      if(kind=='lasso'.and.iteration>0) then
        if(alphas(iteration)<alpha) exit
      end if
      k=size(active); w=solve_upper(transpose(l),solve_lower(l,signs))
      denominator=dot_product(w,signs)
      call require(ieee_is_finite(denominator).and.denominator>0,'ill-conditioned LARS equiangular direction')
      a=1/sqrt(denominator); w=w*a
      corr=matmul(gram(:,active),w)
      ! sklearn rounds inactive correlations to float64 decimal precision (15).
      corr=anint(corr*1.e15_dp)/1.e15_dp
      gamma=c/a
      do i=1,size(inactive)
        key=inactive(i); if(blocked(key)) cycle
        ratio=(c-cov(key))/(a-corr(key)+tiny32)
        if(ratio>0) gamma=min(gamma,ratio)
        if(.not.constrained) then
          ratio=(c+cov(key))/(a+corr(key)+tiny32)
          if(ratio>0) gamma=min(gamma,ratio)
        end if
      end do
      z=-beta(active)/(w+tiny32); zmin=huge(1._dp)
      do i=1,k
        if(z(i)>0) zmin=min(zmin,z(i))
      end do
      drop=zmin<gamma
      remove=z==zmin
      if(drop) then
        where(remove) signs=-signs
        if(kind=='lasso') gamma=zmin
      end if
      old_beta=beta; beta(active)=old_beta(active)+gamma*w
      cov(inactive)=cov(inactive)-gamma*corr(inactive)
      iteration=iteration+1
      if(drop.and.kind=='lasso') then
        dropped=pack(active,remove)
        beta(dropped)=0
        do j=1,size(dropped)
          key=dropped(j); cov(key)=original(key)-dot_product(gram(key,:),beta)
        end do
        active=pack(active,.not.remove); signs=pack(signs,.not.remove)
        inactive=[dropped,inactive]
        if(size(active)==0) drop=.false.
      end if
    end do
    path%alphas=alphas(:iteration+1); path%coefficients=coefficients(:,:iteration+1)
    path%active=active; path%n_iter=iteration
  end function
  subroutine fit_least_angle(self,x,y,limit,target,kind,positive)
    class(lars), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:),target
    integer, intent(in) :: limit
    character(*), intent(in) :: kind
    logical, intent(in) :: positive
    real(dp), allocatable :: center(:),centered(:,:),response(:)
    real(dp) :: mean_y
    type(lars_path_result) :: path
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    center=sum(x,dim=1)/size(x,1); mean_y=sum(y)/size(y)
    if(.not.self%fit_intercept) then
      center=0; mean_y=0
    end if
    centered=x-spread(center,1,size(x,1)); response=y-mean_y
    path=lars_path(centered,response,method=kind,max_iter=limit,alpha_min=target,eps=self%eps,positive=positive)
    self%coef_=path%coefficients(:,size(path%alphas)); self%intercept_=mean_y-dot_product(center,self%coef_)
    self%n_iter_=path%n_iter; self%n_features_in_=size(x,2)
    if(allocated(self%coef_path_)) deallocate(self%coef_path_)
    if(allocated(self%active_)) deallocate(self%active_)
    if(self%fit_path) then
      self%alphas_=path%alphas; self%coef_path_=path%coefficients; self%active_=path%active
    else
      self%alphas_=path%alphas(size(path%alphas):)
    end if
  end subroutine
  subroutine lars_fit(self,x,y)
    class(lars), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    call require(self%n_nonzero_coefs>=1,'Lars n_nonzero_coefs must be positive')
    call fit_least_angle(self,x,y,self%n_nonzero_coefs,0._dp,'lar',.false.)
  end subroutine
  subroutine lasso_lars_fit(self,x,y)
    class(lasso_lars), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    call fit_least_angle(self,x,y,self%max_iter,self%alpha,'lasso',self%positive)
  end subroutine
  subroutine lars_cv_fit(self,x,y)
    class(lars_cv), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    call fit_path_cv(self,x,y,'lar',.false.)
  end subroutine
  subroutine lasso_lars_cv_fit(self,x,y)
    class(lasso_lars_cv), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    call fit_path_cv(self,x,y,'lasso',self%positive)
  end subroutine
  subroutine fit_path_cv(self,x,y,kind,positive)
    class(lars_cv), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    character(*), intent(in) :: kind
    logical, intent(in) :: positive
    type(cv_residual_path), allocatable :: folds(:)
    type(lars_path_result) :: path
    real(dp), allocatable :: train_x(:,:),train_y(:),test_x(:,:),test_y(:),center(:),grid(:),sorted(:),unique(:)
    real(dp), allocatable :: interpolated(:),errors(:,:)
    integer, allocatable :: ids(:),train(:)
    real(dp) :: center_y
    integer :: n,p,c,first,last,width,i,j,used,stride,k,best
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    n=size(x,1); p=size(x,2)
    call require(self%cv>=2.and.self%cv<=n,'invalid least-angle CV fold count')
    call require(self%max_n_alphas>=1,'least-angle max_n_alphas must be positive')
    call require(self%max_iter>=0.and.self%max_iter<huge(self%max_iter),'invalid LARS max_iter')
    call require(ieee_is_finite(self%eps),'nonfinite LARS controls')
    call require(self%eps>=0,'LARS alpha_min and eps must be nonnegative')
    allocate(folds(self%cv)); ids=[(i,i=1,n)]; grid=[real(dp)::]; first=1
    do c=1,self%cv
      width=n/self%cv+merge(1,0,c<=mod(n,self%cv)); last=first+width-1
      train=pack(ids,ids<first.or.ids>last)
      train_x=x(train,:); train_y=y(train); test_x=x(first:last,:); test_y=y(first:last)
      center=sum(train_x,dim=1)/size(train); center_y=sum(train_y)/size(train)
      if(.not.self%fit_intercept) then
        center=0; center_y=0
      end if
      train_x=train_x-spread(center,1,size(train)); train_y=train_y-center_y
      test_x=test_x-spread(center,1,width); test_y=test_y-center_y
      path=lars_path(train_x,train_y,method=kind,max_iter=self%max_iter,eps=self%eps,positive=positive)
      k=size(path%alphas)
      folds(c)%alphas=path%alphas(k:1:-1)
      folds(c)%residuals=transpose(matmul(test_x,path%coefficients(:,k:1:-1))-spread(test_y,2,k))
      grid=[grid,path%alphas]; first=last+1
    end do
    sorted=grid(argsort(grid)); allocate(unique(size(sorted))); used=1; unique(1)=sorted(1)
    do i=2,size(sorted)
      if(sorted(i)==unique(used)) cycle
      used=used+1; unique(used)=sorted(i)
    end do
    ! sklearn's max_n_alphas is a stride target, not a strict upper bound.
    stride=max(1,int(real(used,dp)/self%max_n_alphas)); grid=unique(:used:stride)
    allocate(errors(size(grid),self%cv))
    do c=1,self%cv
      call prepare_cv_interpolation(folds(c),grid(size(grid)),size(grid)>1)
      do i=1,size(grid)
        call interpolate_residuals(folds(c),grid(i),interpolated)
        errors(i,c)=sum(interpolated**2)/size(interpolated)
      end do
    end do
    call require(all(ieee_is_finite(errors)),'nonfinite least-angle CV errors')
    best=minloc(sum(errors,dim=2),dim=1)
    self%alpha_=grid(best); self%cv_alphas_=grid; self%mse_path_=errors
    self%fit_path=.true.
    call fit_least_angle(self,x,y,self%max_iter,self%alpha_,kind,positive)
  end subroutine
  subroutine prepare_cv_interpolation(fold,upper,append_upper)
    type(cv_residual_path), intent(inout) :: fold
    real(dp), intent(in) :: upper
    logical, intent(in) :: append_upper
    real(dp), allocatable :: values(:,:)
    integer, allocatable :: order(:)
    integer :: k,m
    k=size(fold%alphas); m=size(fold%residuals,2)
    if(fold%alphas(1)/=0) then
      values=reshape([fold%residuals(1,:),reshape(transpose(fold%residuals),[k*m])],[m,k+1])
      fold%residuals=transpose(values); fold%alphas=[0._dp,fold%alphas]
    end if
    k=size(fold%alphas)
    ! Preserve sklearn's appended boundary even when a subsampled grid ends
    ! below a fold's initial alpha. A single-point grid avoids its duplicate
    ! zero-endpoint interpolation failure and is evaluated by endpoint clamping.
    if(append_upper.and.fold%alphas(k)/=upper) then
      values=reshape([reshape(transpose(fold%residuals),[k*m]),fold%residuals(k,:)],[m,k+1])
      fold%residuals=transpose(values); fold%alphas=[fold%alphas,upper]
    end if
    ! scipy interp1d sorts knots stably; LAR alphas need not be monotonic.
    order=argsort(fold%alphas); fold%alphas=fold%alphas(order); fold%residuals=fold%residuals(order,:)
  end subroutine
  subroutine interpolate_residuals(fold,alpha,residual)
    type(cv_residual_path), intent(in) :: fold
    real(dp), intent(in) :: alpha
    real(dp), allocatable, intent(out) :: residual(:)
    real(dp) :: fraction
    integer :: i,k
    k=size(fold%alphas)
    if(alpha<=fold%alphas(1).or.k==1) then
      residual=fold%residuals(1,:); return
    end if
    if(alpha>=fold%alphas(k)) then
      residual=fold%residuals(k,:); return
    end if
    do i=1,k-1
      if(alpha>fold%alphas(i+1)) cycle
      fraction=(alpha-fold%alphas(i))/(fold%alphas(i+1)-fold%alphas(i))
      residual=fold%residuals(i,:)+fraction*(fold%residuals(i+1,:)-fold%residuals(i,:)); return
    end do
  end subroutine
  function svd_least_squares(x,y) result(beta)
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp) :: beta(size(x,2)),s(size(x,2)),u(size(x,1),size(x,2)),v(size(x,2),size(x,2))
    real(dp) :: projection(size(x,2)),cutoff
    call dense_svd(x,s,u,v); projection=matmul(transpose(u),y)
    cutoff=epsilon(1._dp)*maxval(s)*max(size(x,1),size(x,2))
    where(s>cutoff)
      projection=projection/max(s,tiny(1._dp))
    elsewhere
      projection=0
    end where
    beta=matmul(v,projection)
  end function
  function nonnegative_least_squares(x,y) result(beta)
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp) :: beta(size(x,2)),trial(size(x,2)),gradient(size(x,2)),tol,fraction
    logical :: passive(size(x,2)),converged
    integer :: p,outer,inner,i,pick
    integer, allocatable :: indices(:),selected(:)
    p=size(x,2); beta=0; passive=.false.; indices=[(i,i=1,p)]; converged=.false.; inner=0
    tol=10*epsilon(1._dp)*max(size(x,1),p)*max(1._dp,maxval(abs(matmul(transpose(x),y))))
    do outer=1,5*p+1
      gradient=matmul(transpose(x),y-matmul(x,beta)); pick=0
      do i=1,p
        if(passive(i).or.gradient(i)<=tol) cycle
        if(pick==0) then
          pick=i
        else if(gradient(i)>gradient(pick)) then
          pick=i
        end if
      end do
      if(pick==0) then
        converged=.true.; exit
      end if
      passive(pick)=.true.
      do
        inner=inner+1
        call require(inner<=10*p+10,'NNLS noise-variance fit did not converge')
        selected=pack(indices,passive); trial=0; trial(selected)=svd_least_squares(x(:,selected),y)
        if(all(trial(selected)>0)) exit
        fraction=1
        do i=1,p
          if(passive(i).and.trial(i)<=0) fraction=min(fraction,beta(i)/(beta(i)-trial(i)))
        end do
        beta=beta+fraction*(trial-beta)
        do i=1,p
          if(passive(i).and.beta(i)<=tol) then
            passive(i)=.false.; beta(i)=0
          end if
        end do
      end do
      beta=trial
    end do
    call require(converged,'NNLS noise-variance fit did not converge')
  end function
  subroutine lasso_lars_ic_fit(self,x,y)
    class(lasso_lars_ic), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    type(lars_path_result) :: path
    real(dp), allocatable :: center(:),centered(:,:),response(:),noise_beta(:),residuals(:,:)
    real(dp) :: mean_y,factor
    integer :: n,p,best,j
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    call require(self%criterion=='aic'.or.self%criterion=='bic','unknown least-angle information criterion')
    call require(ieee_is_finite(self%noise_variance),'nonfinite least-angle noise variance')
    call require(self%noise_variance==-1.or.self%noise_variance>0, &
      'least-angle noise variance must be -1 (auto) or positive')
    n=size(x,1); p=size(x,2); center=sum(x,dim=1)/n; mean_y=sum(y)/n
    if(.not.self%fit_intercept) then
      center=0; mean_y=0
    end if
    centered=x-spread(center,1,n); response=y-mean_y
    path=lars_path(centered,response,method='lasso',max_iter=self%max_iter,eps=self%eps,positive=self%positive)
    self%noise_variance_=self%noise_variance
    if(self%noise_variance==-1) then
      call require(n>p+merge(1,0,self%fit_intercept),'automatic least-angle noise variance requires n > p + intercept')
      if(self%positive) then
        noise_beta=nonnegative_least_squares(centered,response)
      else
        noise_beta=svd_least_squares(centered,response)
      end if
      self%noise_variance_=sum((response-matmul(centered,noise_beta))**2)/(n-p-merge(1,0,self%fit_intercept))
    end if
    call require(self%noise_variance_>0.and.ieee_is_finite(self%noise_variance_), &
      'estimated least-angle noise variance must be positive and finite')
    factor=2; if(self%criterion=='bic') factor=log(real(n,dp))
    residuals=spread(response,2,size(path%alphas))-matmul(centered,path%coefficients)
    self%criterion_=n*log(2*acos(-1._dp)*self%noise_variance_)+sum(residuals**2,dim=1)/self%noise_variance_ &
      +factor*[(count(abs(path%coefficients(:,j))>epsilon(1._dp)),j=1,size(path%alphas))]
    best=minloc(self%criterion_,dim=1); self%alpha_=path%alphas(best)
    self%coef_=path%coefficients(:,best); self%intercept_=mean_y-dot_product(center,self%coef_)
    self%alphas_=path%alphas; self%n_iter_=path%n_iter; self%n_features_in_=p
    if(allocated(self%coef_path_)) deallocate(self%coef_path_)
    if(allocated(self%active_)) deallocate(self%active_)
  end subroutine
  function least_angle_predict(self,x) result(y)
    class(lars), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:)
    call require(allocated(self%coef_),'least-angle model is not fitted'); call check_x(x)
    call require(size(x,2)==size(self%coef_),'least-angle feature count mismatch')
    y=matmul(x,self%coef_)+self%intercept_
  end function
  real(dp) function least_angle_score(self,x,y) result(s)
    class(lars), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    s=r2_score(y,self%predict(x))
  end function
end module
