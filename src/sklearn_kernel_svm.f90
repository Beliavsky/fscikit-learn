! Dense SVR, NuSVR and one-class SVM contracts from sklearn/svm/_base.py,
! _classes.py and its bundled LIBSVM. Native most-violating-pair optimization.
! See THIRD_PARTY_NOTICES.md and LICENSES/libsvm-COPYRIGHT for attribution.
module sklearn_kernel_svm
  use sklearn_common
  use sklearn_metrics, only: rbf_kernel,linear_kernel,polynomial_kernel,r2_score
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: svr,nu_svr,one_class_svm
  type :: kernel_model
    character(16) :: kernel='rbf',gamma_rule='scale'
    real(dp) :: gamma=-1,gamma_=0,coef0=0,tol=1.e-3_dp,intercept_=0,kkt_violation_=0
    integer :: degree=3,max_iter=100000,n_iter_=0,fit_status_=1,n_support_=0
    logical :: converged_=.false.
    integer, allocatable :: support_(:)
    real(dp), allocatable :: support_vectors_(:,:),dual_coef_(:),coef_(:),x_fit_(:,:)
    real(dp), allocatable, private :: dual_full_(:)
    character(16), private :: kernel_fit_='rbf'
    integer, private :: degree_fit_=3
    real(dp), private :: coef0_fit_=0
  contains
    procedure :: decision_function=>kernel_decision
  end type
  type, extends(kernel_model) :: regression_model
    real(dp) :: c=1
  contains
    procedure :: predict=>regression_predict
    procedure :: score=>regression_score
  end type
  type, extends(regression_model) :: svr
    real(dp) :: epsilon=0.1_dp
  contains
    procedure :: fit=>svr_fit
  end type
  type, extends(regression_model) :: nu_svr
    real(dp) :: nu=0.5_dp,epsilon_=0
  contains
    procedure :: fit=>nusvr_fit
  end type
  type, extends(kernel_model) :: one_class_svm
    real(dp) :: nu=0.5_dp,offset_=0
  contains
    procedure :: fit=>one_fit
    procedure :: predict=>one_predict
    procedure :: fit_predict=>one_fit_predict
    procedure :: score_samples=>one_score_samples
  end type
contains
  function gram_matrix(self,x,y) result(k)
    class(kernel_model), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp), allocatable :: k(:,:)
    select case(self%kernel_fit_)
    case('rbf'); k=rbf_kernel(x,y,self%gamma_)
    case('linear'); k=linear_kernel(x,y)
    case('poly','polynomial'); k=polynomial_kernel(x,y,self%degree_fit_,self%gamma_,self%coef0_fit_)
    case('precomputed'); k=x
    end select
  end function
  subroutine setup(self,x,gram)
    class(kernel_model), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable, intent(out) :: gram(:,:)
    real(dp), allocatable :: values(:),vectors(:,:)
    real(dp) :: variance,mean
    integer :: n
    call check_x(x); n=size(x,1)
    call require(self%max_iter>0,'kernel SVM max_iter must be positive')
    call require(ieee_is_finite(self%tol),'invalid kernel SVM tolerance')
    call require(self%tol>0,'invalid kernel SVM tolerance')
    call require(self%kernel=='rbf'.or.self%kernel=='linear'.or.self%kernel=='poly'.or. &
      self%kernel=='polynomial'.or.self%kernel=='precomputed','unsupported kernel SVM kernel')
    call require(ieee_is_finite(self%gamma).and.ieee_is_finite(self%coef0),'invalid kernel SVM parameters')
    call require((self%gamma>=0.or.self%gamma==-1).and.self%degree>=0.and.self%coef0>=0, &
      'invalid kernel SVM parameters')
    call require(self%gamma_rule=='scale'.or.self%gamma_rule=='auto','invalid kernel SVM gamma rule')
    self%kernel_fit_=self%kernel; self%degree_fit_=self%degree; self%coef0_fit_=self%coef0
    self%x_fit_=x; self%gamma_=self%gamma
    if(self%gamma_==-1) then
      self%gamma_=1._dp/size(x,2)
      if(self%gamma_rule=='scale') then
        mean=sum(x)/size(x); variance=sum((x-mean)**2)/size(x); self%gamma_=1
        if(variance>0) self%gamma_=1/(size(x,2)*variance)
      end if
    end if
    if(self%kernel_fit_=='precomputed') then
      self%gamma_=0
      call require(size(x,2)==n,'kernel SVM precomputed matrix must be square')
      call require(all(abs(x-transpose(x))<=1.e-10_dp*max(1._dp,abs(x),abs(transpose(x)))), &
        'kernel SVM precomputed matrix must be symmetric')
      allocate(values(n),vectors(n,n)); call symmetric_eigen(x,values,vectors)
      call require(minval(values)>=-1.e-10_dp*max(1._dp,maxval(abs(values))), &
        'kernel SVM precomputed matrix must be positive semidefinite')
    end if
    gram=gram_matrix(self,x,x)
    call require(all(ieee_is_finite(gram)),'kernel SVM nonfinite Gram matrix')
    if(allocated(self%coef_)) deallocate(self%coef_)
  end subroutine
  function bounds(n,c,sample_weight) result(cap)
    integer, intent(in) :: n
    real(dp), intent(in) :: c
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp) :: cap(n)
    call require(ieee_is_finite(c),'kernel SVM C must be positive')
    call require(c>0,'kernel SVM C must be positive'); cap=c
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'kernel SVM weight count mismatch')
      call require(all(ieee_is_finite(sample_weight)),'invalid kernel SVM sample weights')
      call require(all(sample_weight>=0).and.any(sample_weight>0),'invalid kernel SVM sample weights')
      cap=c*sample_weight
    end if
    call require(all(ieee_is_finite(cap)),'kernel SVM capacity overflow')
  end function
  subroutine select_pair(g,alpha,cap,signs,index,group,epsilon,groups,first,last,gap)
    real(dp), intent(in) :: g(:),alpha(:),cap(:),signs(:),epsilon
    integer, intent(in) :: index(:),group(:),groups
    integer, intent(out) :: first,last
    real(dp), intent(out) :: gap
    real(dp) :: low(groups),high(groups),value
    integer :: left(groups),right(groups),j,h
    low=huge(1._dp); high=-huge(1._dp); left=0; right=0
    do j=1,size(alpha)
      if(cap(j)==0) cycle
      h=group(j); value=g(index(j))+signs(j)*epsilon
      if((signs(j)>0.and.alpha(j)<cap(j)).or.(signs(j)<0.and.alpha(j)>0)) then
        if(value<low(h)) then
          low(h)=value; left(h)=j
        end if
      end if
      if((signs(j)>0.and.alpha(j)>0).or.(signs(j)<0.and.alpha(j)<cap(j))) then
        if(value>high(h)) then
          high(h)=value; right(h)=j
        end if
      end if
    end do
    gap=0; first=0; last=0
    do h=1,groups
      if(left(h)==0.or.right(h)==0) cycle
      if(high(h)-low(h)>gap) then
        gap=high(h)-low(h); first=left(h); last=right(h)
      end if
    end do
  end subroutine
  subroutine working_pair(gram,g,alpha,cap,signs,index,group,epsilon,groups,first,last,best_step)
    real(dp), intent(in) :: gram(:,:),g(:),alpha(:),cap(:),signs(:),epsilon
    integer, intent(in) :: index(:),group(:),groups
    integer, intent(out) :: first,last
    real(dp), intent(out) :: best_step
    real(dp) :: low,value,inc_limit,dec_limit,eta,diff,step,gain,best_gain
    integer :: h,i,j,left
    first=0; last=0; best_step=0; best_gain=-1
    do h=1,groups
      low=huge(1._dp); left=0
      do i=1,size(alpha)
        if(group(i)/=h.or.cap(i)==0) cycle
        if(.not.((signs(i)>0.and.alpha(i)<cap(i)).or.(signs(i)<0.and.alpha(i)>0))) cycle
        value=g(index(i))+signs(i)*epsilon
        if(value<low) then
          low=value; left=i
        end if
      end do
      if(left==0) cycle
      inc_limit=alpha(left)
      if(signs(left)>0) inc_limit=cap(left)-alpha(left)
      do j=1,size(alpha)
        if(j==left.or.group(j)/=h.or.cap(j)==0) cycle
        if(.not.((signs(j)>0.and.alpha(j)>0).or.(signs(j)<0.and.alpha(j)<cap(j)))) cycle
        diff=g(index(j))+signs(j)*epsilon-low
        if(diff<=0) cycle
        dec_limit=cap(j)-alpha(j)
        if(signs(j)>0) dec_limit=alpha(j)
        step=min(inc_limit,dec_limit)
        eta=max(0._dp,gram(index(left),index(left))+gram(index(j),index(j))-2*gram(index(left),index(j)))
        if(eta>0) step=min(step,diff/eta)
        gain=step*(diff-eta*step/2)
        if(gain>best_gain) then
          first=left; last=j; best_step=step; best_gain=gain
        end if
      end do
    end do
  end subroutine
  subroutine optimize(self,gram,y,capacity,mode,epsilon,nu,levels)
    class(kernel_model), intent(inout) :: self
    real(dp), intent(in) :: gram(:,:),y(:),capacity(:),epsilon,nu
    integer, intent(in) :: mode ! 1: epsilon SVR; 2: nu SVR; 3: one-class
    real(dp), intent(out) :: levels(2)
    real(dp), allocatable :: alpha(:),cap(:),signs(:),g(:),beta(:)
    integer, allocatable :: index(:),group(:)
    real(dp) :: remaining,target,gap,step,low,high,free_sum,value,bound_tol,fallback_sum
    integer :: n,m,groups,i,j,a,b,it,h,count_free,fallback_count
    n=size(y); m=2*n; groups=1
    if(mode==3) m=n
    if(mode==2) groups=2
    allocate(alpha(m),cap(m),signs(m),g(n),beta(n),index(m),group(m)); alpha=0; signs=1; group=1
    index=[(1+mod(i-1,n),i=1,m)]; cap=capacity(index)
    if(mode/=3) signs(n+1:)=-1
    if(mode==2) group(n+1:)=2
    if(mode==3) then
      ! A feasible interior start preserves symmetry and avoids unnecessarily
      ! exciting very flat dual directions in smooth, symmetric kernels.
      alpha=nu*cap
    else if(mode==2) then
      target=nu*sum(capacity)
      target=target/2
      remaining=target
      do i=1,n
        alpha(i)=min(cap(i),max(0._dp,remaining)); remaining=remaining-alpha(i)
      end do
      alpha(n+1:)=alpha(:n)
    end if
    beta=0
    do i=1,m
      beta(index(i))=beta(index(i))+signs(i)*alpha(i)
    end do
    g=matmul(gram,beta)-y; self%n_iter_=0
    do it=1,self%max_iter
      call select_pair(g,alpha,cap,signs,index,group,epsilon,groups,a,b,gap)
      if(gap<=self%tol) then
        beta=0
        do i=1,m
          beta(index(i))=beta(index(i))+signs(i)*alpha(i)
        end do
        g=matmul(gram,beta)-y
        call select_pair(g,alpha,cap,signs,index,group,epsilon,groups,a,b,gap)
        if(gap<=self%tol) exit
      end if
      call working_pair(gram,g,alpha,cap,signs,index,group,epsilon,groups,a,b,step)
      if(a==0.or.b==0) exit
      i=index(a); j=index(b)
      if(step<=tiny(1._dp)) exit
      alpha(a)=max(0._dp,min(cap(a),alpha(a)+signs(a)*step))
      alpha(b)=max(0._dp,min(cap(b),alpha(b)-signs(b)*step))
      beta(i)=beta(i)+step; beta(j)=beta(j)-step
      g=g+step*(gram(:,i)-gram(:,j)); self%n_iter_=it
      if(mod(it,1000)==0) g=matmul(gram,beta)-y
    end do
    ! Reconstruct signed coefficients before final diagnostics to bound cache drift.
    beta=0
    do i=1,m
      beta(index(i))=beta(index(i))+signs(i)*alpha(i)
    end do
    g=matmul(gram,beta)-y
    call select_pair(g,alpha,cap,signs,index,group,epsilon,groups,a,b,self%kkt_violation_)
    self%converged_=self%kkt_violation_<=self%tol; self%fit_status_=merge(0,1,self%converged_)
    levels=0
    do h=1,groups
      low=huge(1._dp); high=-huge(1._dp); free_sum=0; count_free=0; fallback_sum=0; fallback_count=0
      do i=1,m
        if(group(i)/=h.or.cap(i)==0) cycle
        value=g(index(i))+signs(i)*epsilon
        if(alpha(i)>0.and.alpha(i)<cap(i)) then
          fallback_sum=fallback_sum+value; fallback_count=fallback_count+1
        end if
        ! Roundoff near a bound must not turn a saturated solution's threshold
        ! interval into a spurious free-vector equality.
        bound_tol=32*spacing(1._dp)*cap(i)
        if(alpha(i)>bound_tol.and.alpha(i)<cap(i)-bound_tol) then
          free_sum=free_sum+value; count_free=count_free+1
        end if
        if((signs(i)>0.and.alpha(i)<cap(i)-bound_tol).or.(signs(i)<0.and.alpha(i)>bound_tol)) low=min(low,value)
        if((signs(i)>0.and.alpha(i)>bound_tol).or.(signs(i)<0.and.alpha(i)<cap(i)-bound_tol)) high=max(high,value)
      end do
      if(count_free>0) then
        levels(h)=free_sum/count_free
      else if((low==huge(1._dp).or.high==-huge(1._dp)).and.fallback_count>0) then
        levels(h)=fallback_sum/fallback_count
      else
        levels(h)=low/2+high/2
      end if
    end do
    self%dual_full_=beta; self%support_=pack([(i,i=1,n)],abs(beta)>0)
    self%n_support_=size(self%support_); self%dual_coef_=beta(self%support_)
    if(self%kernel_fit_=='precomputed') then
      self%support_vectors_=reshape([real(dp)::],[0,0])
    else
      self%support_vectors_=self%x_fit_(self%support_,:)
    end if
    if(self%kernel_fit_=='linear') self%coef_=matmul(beta,self%x_fit_)
  end subroutine
  subroutine svr_fit(self,x,y,sample_weight)
    class(svr), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: gram(:,:),cap(:)
    real(dp) :: levels(2)
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    call require(ieee_is_finite(self%epsilon),'invalid SVR epsilon')
    call require(self%epsilon>=0,'invalid SVR epsilon'); cap=bounds(size(y),self%c,sample_weight)
    call setup(self,x,gram); call optimize(self,gram,y,cap,1,self%epsilon,0._dp,levels)
    self%intercept_=-levels(1)
  end subroutine
  subroutine nusvr_fit(self,x,y,sample_weight)
    class(nu_svr), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: gram(:,:),cap(:)
    real(dp) :: levels(2)
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    call require(ieee_is_finite(self%nu),'invalid NuSVR nu')
    call require(self%nu>0.and.self%nu<=1,'invalid NuSVR nu'); cap=bounds(size(y),self%c,sample_weight)
    call setup(self,x,gram); call optimize(self,gram,y,cap,2,0._dp,self%nu,levels)
    self%intercept_=-(levels(1)+levels(2))/2; self%epsilon_=max(0._dp,(levels(2)-levels(1))/2)
  end subroutine
  subroutine one_fit(self,x,sample_weight)
    class(one_class_svm), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: gram(:,:),cap(:)
    real(dp) :: levels(2)
    call require(ieee_is_finite(self%nu),'invalid OneClassSVM nu')
    call require(self%nu>0.and.self%nu<1,'OneClassSVM requires 0 < nu < 1')
    call setup(self,x,gram); cap=bounds(size(x,1),1._dp,sample_weight)
    call optimize(self,gram,spread(0._dp,1,size(x,1)),cap,3,0._dp,self%nu,levels)
    self%offset_=levels(1); self%intercept_=-self%offset_
  end subroutine
  function kernel_raw(self,x) result(values)
    class(kernel_model), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:)
    call require(allocated(self%dual_full_),'kernel SVM is not fitted')
    call check_xy(transpose(x),size(self%x_fit_,2))
    values=matmul(gram_matrix(self,x,self%x_fit_),self%dual_full_)
  end function
  function kernel_decision(self,x) result(values)
    class(kernel_model), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:)
    values=kernel_raw(self,x)+self%intercept_
  end function
  function regression_predict(self,x) result(values)
    class(regression_model), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:)
    values=self%decision_function(x)
  end function
  real(dp) function regression_score(self,x,y) result(value)
    class(regression_model), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    value=r2_score(y,self%predict(x))
  end function
  function one_predict(self,x) result(labels)
    class(one_class_svm), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: labels(:)
    labels=merge(1,-1,self%decision_function(x)>0)
  end function
  function one_score_samples(self,x) result(values)
    class(one_class_svm), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:)
    values=kernel_raw(self,x)
  end function
  function one_fit_predict(self,x,sample_weight) result(labels)
    class(one_class_svm), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(in), optional :: sample_weight(:)
    integer, allocatable :: labels(:)
    call self%fit(x,sample_weight); labels=self%predict(x)
  end function
end module
