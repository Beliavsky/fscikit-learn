! NuSVC formulation and scaling: sklearn 1.6.1 and bundled LIBSVM (BSD notices retained).
module sklearn_nu_classification
  use sklearn_common, only: dp,require,check_x,check_xy,unique_labels
  use sklearn_metrics, only: linear_kernel,rbf_kernel,polynomial_kernel,sigmoid_kernel,accuracy_score
  use ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: nu_svc
  type :: nu_svc
    real(dp) :: nu=.5_dp,gamma=-1,gamma_=0,coef0=0,tol=1.e-3_dp
    integer :: degree=3,max_iter=200000,n_features_in_=0,fit_status_=1
    character(16) :: kernel='rbf',gamma_rule='scale',decision_function_shape='ovr'
    logical :: break_ties=.false.,converged_=.false.
    integer, allocatable :: classes_(:),support_(:),n_support_(:),n_iter_(:)
    real(dp), allocatable :: support_vectors_(:,:),dual_coef_(:,:),coef_(:,:),intercept_(:)
    real(dp), allocatable :: pair_alpha_(:,:),pair_scales_(:),kkt_violation_(:)
    real(dp), allocatable, private :: x_fit(:,:),dual_full(:,:)
    integer, allocatable, private :: first_class(:),second_class(:)
    character(16), private :: fitted_kernel='rbf'
    real(dp), private :: fitted_coef0=0
    integer, private :: fitted_degree=3
  contains
    procedure :: fit
    procedure :: decision_function
    procedure :: predict
    procedure :: score
  end type
contains
  function kernel_matrix(self,x,y) result(gram)
    class(nu_svc), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp), allocatable :: gram(:,:)
    select case(trim(self%fitted_kernel))
    case('linear'); gram=linear_kernel(x,y)
    case('rbf'); gram=rbf_kernel(x,y,self%gamma_)
    case('poly','polynomial'); gram=polynomial_kernel(x,y,self%fitted_degree,self%gamma_,self%fitted_coef0)
    case('sigmoid'); gram=sigmoid_kernel(x,y,self%gamma_,self%fitted_coef0)
    case('precomputed'); gram=x
    end select
  end function
  subroutine optimize(gram,labels,cap,nu,tol,max_iter,alpha,dual,bias,r,gap,iterations)
    real(dp), intent(in) :: gram(:,:),labels(:),cap(:),nu,tol
    integer, intent(in) :: max_iter
    real(dp), intent(out) :: alpha(:),dual(:),bias,r,gap
    integer, intent(out) :: iterations
    real(dp) :: q(size(labels),size(labels)),g(size(labels)),levels(2),mass,remaining,low,high
    real(dp) :: delta,curvature,diff,gain,best_gain,bound_tol,free_sum
    integer :: i,j,h,it,left,a,b,free_count,n
    n=size(labels); q=gram*spread(labels,2,n)*spread(labels,1,n)
    alpha=0; mass=nu*sum(cap)/2
    do h=1,2
      remaining=mass
      do i=1,n
        if((labels(i)>0).neqv.(h==1)) cycle
        alpha(i)=min(cap(i),max(0._dp,remaining)); remaining=remaining-alpha(i)
      end do
    end do
    g=matmul(q,alpha); iterations=0
    do it=1,max_iter
      call violation(gap)
      if(gap<=tol) then
        g=matmul(q,alpha); call violation(gap)
        if(gap<=tol) exit
      end if
      best_gain=-1; a=0; b=0; delta=0
      do h=1,2
        left=0; low=huge(1._dp)
        do i=1,n
          if((labels(i)>0).neqv.(h==1)) cycle
          if(alpha(i)>=cap(i)) cycle
          if(g(i)<low) then
            low=g(i); left=i
          end if
        end do
        if(left==0) cycle
        do j=1,n
          if(labels(j)/=labels(left).or.j==left.or.alpha(j)<=0) cycle
          diff=g(j)-low
          if(diff<=0) cycle
          curvature=max(0._dp,q(left,left)+q(j,j)-2*q(left,j))
          remaining=min(cap(left)-alpha(left),alpha(j))
          if(curvature>0) remaining=min(remaining,diff/curvature)
          gain=remaining*(diff-curvature*remaining/2)
          if(gain>best_gain) then
            best_gain=gain; a=left; b=j; delta=remaining
          end if
        end do
      end do
      if(a==0.or.b==0.or.delta<=tiny(1._dp)) exit
      alpha(a)=min(cap(a),max(0._dp,alpha(a)+delta))
      alpha(b)=min(cap(b),max(0._dp,alpha(b)-delta))
      g=g+delta*(q(:,a)-q(:,b)); iterations=it
      if(mod(it,1000)==0) g=matmul(q,alpha)
    end do
    g=matmul(q,alpha); call violation(gap)
    do h=1,2
      low=huge(1._dp); high=-huge(1._dp); free_sum=0; free_count=0
      do i=1,n
        if((labels(i)>0).neqv.(h==1)) cycle
        bound_tol=32*epsilon(1._dp)*cap(i)
        if(alpha(i)>bound_tol.and.alpha(i)<cap(i)-bound_tol) then
          free_sum=free_sum+g(i); free_count=free_count+1
        end if
        if(alpha(i)<cap(i)-bound_tol) low=min(low,g(i))
        if(alpha(i)>bound_tol) high=max(high,g(i))
      end do
      if(free_count>0) then
        levels(h)=free_sum/free_count
      else if(low==huge(1._dp).or.high==-huge(1._dp)) then
        call require(.false.,'NuSVC unbounded margin normalization')
      else
        levels(h)=low/2+high/2
      end if
    end do
    r=levels(1)/2+levels(2)/2
    call require(ieee_is_finite(r),'NuSVC nonfinite margin normalization')
    call require(abs(r)>tiny(1._dp),'NuSVC zero margin normalization')
    dual=alpha*labels/r; bias=-(levels(1)/2-levels(2)/2)/r
    call require(all(ieee_is_finite(dual)).and.ieee_is_finite(bias),'NuSVC nonfinite coefficients')
  contains
    subroutine violation(value)
      real(dp), intent(out) :: value
      integer :: group,k
      real(dp) :: lo,hi
      value=0
      do group=1,2
        lo=huge(1._dp); hi=-huge(1._dp)
        do k=1,n
          if((labels(k)>0).neqv.(group==1)) cycle
          if(alpha(k)<cap(k)) lo=min(lo,g(k))
          if(alpha(k)>0) hi=max(hi,g(k))
        end do
        if(lo/=huge(1._dp).and.hi/=-huge(1._dp)) value=max(value,hi-lo)
      end do
    end subroutine
  end subroutine
  subroutine fit(self,x,y,sample_weight)
    class(nu_svc), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    real(dp), allocatable :: gram(:,:),weights(:),mass(:),labels(:),alpha(:),dual(:)
    real(dp) :: mean,variance,total,bias,r,gap
    integer, allocatable :: ids(:),left(:),right(:),supports(:)
    logical :: active(size(y))
    integer :: n,p,k,pairs,i,j,a,b,m,t,iterations,row
    call check_xy(x,size(y)); n=size(y); p=size(x,2)
    call require(ieee_is_finite(self%nu),'NuSVC nu must be finite and in (0,1]')
    call require(self%nu>0.and.self%nu<=1,'NuSVC nu must be finite and in (0,1]')
    call require(ieee_is_finite(self%tol),'NuSVC tolerance must be finite and positive')
    call require(self%tol>0.and.self%max_iter>0,'NuSVC tolerance and max_iter must be positive')
    call require(any(self%kernel==[character(16)::'linear','rbf','poly','polynomial','sigmoid','precomputed']), &
      'unknown NuSVC kernel')
    call require(self%gamma_rule=='scale'.or.self%gamma_rule=='auto','unknown NuSVC gamma rule')
    call require(ieee_is_finite(self%gamma).and.ieee_is_finite(self%coef0),'nonfinite NuSVC kernel parameters')
    call require((self%gamma==-1.or.self%gamma>=0).and.self%degree>=0,'invalid NuSVC gamma or degree')
    call shape_controls(self)
    self%classes_=unique_labels(y); k=size(self%classes_)
    call require(k>=2,'NuSVC requires at least two classes')
    weights=spread(1._dp,1,n)
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'NuSVC sample weight shape mismatch')
      call require(all(ieee_is_finite(sample_weight)),'NuSVC weights must be finite and nonnegative')
      call require(all(sample_weight>=0),'NuSVC weights must be finite and nonnegative')
      weights=sample_weight
    end if
    total=sum(weights)
    call require(ieee_is_finite(total),'NuSVC weights must have positive finite sum')
    call require(total>0,'NuSVC weights must have positive finite sum')
    allocate(mass(k))
    do a=1,k
      mass(a)=sum(weights,mask=y==self%classes_(a))
    end do
    call require(all(mass>0),'NuSVC every class needs positive sample weight')
    do a=1,k-1
      do b=a+1,k
        call require(self%nu*(mass(a)/2+mass(b)/2)<=min(mass(a),mass(b)), &
          'NuSVC specified nu is infeasible')
      end do
    end do
    self%gamma_=self%gamma
    if(self%gamma_==-1) then
      self%gamma_=1._dp/p
      if(self%gamma_rule=='scale') then
        mean=sum(x)/size(x); variance=sum((x-mean)**2)/size(x); self%gamma_=1
        if(variance>0) self%gamma_=1/(p*variance)
      end if
    end if
    self%fitted_kernel=self%kernel; self%fitted_coef0=self%coef0; self%fitted_degree=self%degree
    if(self%kernel=='precomputed') then
      self%gamma_=0
      call require(p==n,'NuSVC precomputed training matrix must be square')
      call require(all(abs(x-transpose(x))<=1.e-10_dp*max(1._dp,abs(x),abs(transpose(x)))), &
        'NuSVC precomputed training matrix must be symmetric')
    end if
    self%x_fit=x; self%n_features_in_=p; gram=kernel_matrix(self,x,x)
    call require(all(ieee_is_finite(gram)),'NuSVC nonfinite Gram matrix')
    pairs=k*(k-1)/2
    if(allocated(self%dual_full)) then
      deallocate(self%dual_full,self%pair_alpha_,self%pair_scales_,self%intercept_,self%kkt_violation_, &
        self%n_iter_,self%first_class,self%second_class)
    end if
    if(allocated(self%coef_)) deallocate(self%coef_)
    allocate(self%dual_full(n,pairs),self%pair_alpha_(n,pairs),self%pair_scales_(pairs), &
      self%intercept_(pairs),self%kkt_violation_(pairs),self%n_iter_(pairs), &
      self%first_class(pairs),self%second_class(pairs))
    self%dual_full=0; self%pair_alpha_=0; active=.false.; m=0; self%converged_=.true.
    do a=1,k-1
      do b=a+1,k
        m=m+1; self%first_class(m)=a; self%second_class(m)=b
        left=pack([(i,i=1,n)],y==self%classes_(a).and.weights>0)
        right=pack([(i,i=1,n)],y==self%classes_(b).and.weights>0); ids=[left,right]
        labels=merge(1._dp,-1._dp,y(ids)==self%classes_(a)); t=size(ids)
        allocate(alpha(t),dual(t))
        call optimize(gram(ids,ids),labels,weights(ids),self%nu,self%tol,self%max_iter,alpha,dual,bias,r,gap,iterations)
        self%pair_alpha_(ids,m)=alpha; self%dual_full(ids,m)=dual
        self%pair_scales_(m)=r; self%intercept_(m)=bias; self%kkt_violation_(m)=gap; self%n_iter_(m)=iterations
        active=active.or.abs(self%dual_full(:,m))>0
        self%converged_=self%converged_.and.gap<=self%tol; deallocate(alpha,dual)
      end do
    end do
    supports=[integer::]; self%n_support_=spread(0,1,k)
    do a=1,k
      ids=pack([(i,i=1,n)],active.and.y==self%classes_(a)); self%n_support_(a)=size(ids); supports=[supports,ids]
    end do
    self%support_=supports
    if(self%kernel=='precomputed') then
      self%support_vectors_=reshape([real(dp)::],[0,0])
    else
      self%support_vectors_=x(supports,:)
    end if
    self%dual_coef_=spread(spread(0._dp,1,k-1),2,size(supports))
    do m=1,pairs
      a=self%first_class(m); b=self%second_class(m)
      do j=1,size(supports)
        i=supports(j); row=0
        if(y(i)==self%classes_(a)) row=b-1
        if(y(i)==self%classes_(b)) row=a
        if(row>0) self%dual_coef_(row,j)=self%dual_full(i,m)
      end do
    end do
    if(k==2) then
      self%dual_coef_=-self%dual_coef_; self%intercept_=-self%intercept_; self%dual_full=-self%dual_full
    end if
    if(self%kernel=='linear') self%coef_=matmul(transpose(self%dual_full),x)
    self%fit_status_=merge(0,1,self%converged_)
  end subroutine
  subroutine shape_controls(self)
    class(nu_svc), intent(in) :: self
    call require(self%decision_function_shape=='ovo'.or.self%decision_function_shape=='ovr', &
      'NuSVC decision_function_shape must be ovo or ovr')
    call require(.not.(self%break_ties.and.self%decision_function_shape=='ovo'), &
      'NuSVC break_ties requires ovr decision scores')
  end subroutine
  function pair_scores(self,x) result(z)
    class(nu_svc), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:),gram(:,:)
    integer :: m
    call require(allocated(self%dual_full),'NuSVC is not fitted'); call check_x(x)
    call require(size(x,2)==self%n_features_in_,'NuSVC query feature count mismatch')
    call shape_controls(self); gram=kernel_matrix(self,x,self%x_fit)
    z=matmul(gram,self%dual_full)
    do m=1,size(z,2)
      z(:,m)=z(:,m)+self%intercept_(m)
    end do
  end function
  function decision_function(self,x) result(z)
    class(nu_svc), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:),raw(:,:),confidence(:,:)
    integer :: i,m,a,b,k
    raw=pair_scores(self,x); k=size(self%classes_)
    if(k==2.or.self%decision_function_shape=='ovo') then
      z=raw
      return
    end if
    allocate(z(size(x,1),k),confidence(size(x,1),k)); z=0; confidence=0
    do m=1,size(raw,2)
      a=self%first_class(m); b=self%second_class(m)
      confidence(:,a)=confidence(:,a)+raw(:,m); confidence(:,b)=confidence(:,b)-raw(:,m)
      do i=1,size(x,1)
        if(raw(i,m)>=0) then
          z(i,a)=z(i,a)+1
        else
          z(i,b)=z(i,b)+1
        end if
      end do
    end do
    z=z+confidence/(3*(abs(confidence)+1))
  end function
  function predict(self,x) result(y)
    class(nu_svc), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: y(size(x,1)),i,m,a,b
    integer, allocatable :: votes(:,:)
    real(dp), allocatable :: z(:,:)
    if(self%break_ties) then
      z=self%decision_function(x)
      if(size(self%classes_)>2) then
        do i=1,size(y)
          y(i)=self%classes_(maxloc(z(i,:),dim=1))
        end do
        return
      end if
    end if
    z=pair_scores(self,x)
    if(size(self%classes_)==2) then
      y=merge(self%classes_(2),self%classes_(1),z(:,1)>=0)
      return
    end if
    allocate(votes(size(x,1),size(self%classes_))); votes=0
    do m=1,size(z,2)
      a=self%first_class(m); b=self%second_class(m)
      do i=1,size(y)
        if(z(i,m)>0) then
          votes(i,a)=votes(i,a)+1
        else
          votes(i,b)=votes(i,b)+1
        end if
      end do
    end do
    do i=1,size(y)
      y(i)=self%classes_(maxloc(votes(i,:),dim=1))
    end do
  end function
  real(dp) function score(self,x,y) result(value)
    class(nu_svc), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    value=accuracy_score(y,self%predict(x))
  end function
end module
