! Native dense LARS homotopy, adapted from sklearn 1.6.1 _least_angle.py.
! Upstream BSD-3-Clause terms are retained in THIRD_PARTY_NOTICES.md and LICENSES/.
module sklearn_least_angle
  use sklearn_common
  use sklearn_metrics, only: r2_score
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: lars,lasso_lars,lars_path,lars_path_result
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
