module sklearn_svm
  use sklearn_common
  use sklearn_kernel_svm, only: svr,nu_svr,one_class_svm
  use sklearn_metrics, only: linear_kernel,rbf_kernel,polynomial_kernel,sigmoid_kernel,accuracy_score
  implicit none
  private
  public :: svc,svr,nu_svr,one_class_svm
  type :: binary_svm
    integer :: first=0,second=0
    real(dp), allocatable :: dual(:)
    real(dp) :: intercept=0
    logical :: converged=.false.
    integer :: n_iter=0
  end type
  type :: svc
    real(dp) :: c=1,gamma=-1,coef0=0,tol=1.e-4_dp
    integer :: degree=3,max_iter=10000,n_iter_=0
    character(16) :: kernel='rbf'
    logical :: converged_=.false.
    integer, allocatable :: classes_(:),support_(:)
    real(dp), allocatable :: x_fit_(:,:),support_vectors_(:,:)
    type(binary_svm), allocatable :: models_(:)
    real(dp) :: gamma_=0
  contains
    procedure :: fit=>svc_fit
    procedure :: decision_function=>svc_decision
    procedure :: predict=>svc_predict
    procedure :: score=>svc_score
  end type
contains
  function kernel_matrix(self,x,y) result(k)
    class(svc), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp), allocatable :: k(:,:)
    select case(trim(self%kernel))
    case('linear'); k=linear_kernel(x,y)
    case('rbf'); k=rbf_kernel(x,y,self%gamma_)
    case('poly','polynomial'); k=polynomial_kernel(x,y,self%degree,self%gamma_,self%coef0)
    case('sigmoid'); k=sigmoid_kernel(x,y,self%gamma_,self%coef0)
    case default; call require(.false.,'unknown SVC kernel')
    end select
  end function
  subroutine binary_fit(model,gram,target,c,tol,max_iter)
    type(binary_svm), intent(inout) :: model
    real(dp), intent(in) :: gram(:,:),target(:),c,tol
    integer, intent(in) :: max_iter
    real(dp) :: alpha(size(target)),errors(size(target)),ei,ej,oldi,oldj,newi,newj
    real(dp) :: low,high,eta,b,b1,b2,best,delta,violation
    integer :: n,it,i,j,h,changed,passes
    ! Deterministic two-coordinate SMO: cached residuals and exhaustive partner
    ! search when the maximal-error heuristic does not make progress.
    n=size(target); alpha=0; errors=-target; b=0; passes=0; model%converged=.false.
    do it=1,max_iter
      changed=0
      do i=1,n
        ei=errors(i)
        if(.not.((target(i)*ei < -tol.and.alpha(i)<c-1.e-12_dp).or. &
          (target(i)*ei>tol.and.alpha(i)>1.e-12_dp))) cycle
        best=-1; j=0
        do h=1,n
          if(h==i) cycle
          if(abs(ei-errors(h))>best) then
            best=abs(ei-errors(h)); j=h
          end if
        end do
        if(j==0) cycle
        if(.not.take_step(i,j)) then
          do h=1,n
            if(h==i.or.h==j) cycle
            if(take_step(i,h)) exit
          end do
        end if
      end do
      violation=0
      do i=1,n
        if(alpha(i)<c-1.e-10_dp) violation=max(violation,-target(i)*errors(i))
        if(alpha(i)>1.e-10_dp) violation=max(violation,target(i)*errors(i))
      end do
      if(violation<=tol) then
        model%converged=.true.; exit
      end if
      if(changed==0) then
        passes=passes+1
      else
        passes=0
      end if
      if(passes>=10) exit
    end do
    model%dual=alpha*target; model%intercept=b; model%n_iter=min(it,max_iter)
  contains
    logical function take_step(a,z) result(ok)
      integer, intent(in) :: a,z
      real(dp) :: left_gain,right_gain,derivative,step_l,step_h,previous_b
      ok=.false.; oldi=alpha(a); oldj=alpha(z); ei=errors(a); ej=errors(z)
      if(target(a)/=target(z)) then
        low=max(0._dp,oldj-oldi); high=min(c,c+oldj-oldi)
      else
        low=max(0._dp,oldi+oldj-c); high=min(c,oldi+oldj)
      end if
      if(high-low<1.e-14_dp) return
      eta=gram(a,a)+gram(z,z)-2*gram(a,z)
      if(eta>1.e-14_dp) then
        newj=max(low,min(high,oldj+target(z)*(ei-ej)/eta))
      else
        derivative=target(z)*(ei-ej); step_l=low-oldj; step_h=high-oldj
        left_gain=derivative*step_l-0.5_dp*eta*step_l*step_l
        right_gain=derivative*step_h-0.5_dp*eta*step_h*step_h
        newj=merge(low,high,left_gain>right_gain)
      end if
      if(abs(newj-oldj)<1.e-12_dp*(newj+oldj+1)) return
      newi=oldi+target(a)*target(z)*(oldj-newj)
      b1=b-ei-target(a)*(newi-oldi)*gram(a,a)-target(z)*(newj-oldj)*gram(a,z)
      b2=b-ej-target(a)*(newi-oldi)*gram(a,z)-target(z)*(newj-oldj)*gram(z,z)
      previous_b=b
      if(newi>1.e-12_dp.and.newi<c-1.e-12_dp) then
        b=b1
      else if(newj>1.e-12_dp.and.newj<c-1.e-12_dp) then
        b=b2
      else
        b=(b1+b2)/2
      end if
      errors=errors+target(a)*(newi-oldi)*gram(:,a)+target(z)*(newj-oldj)*gram(:,z)+b-previous_b
      alpha(a)=newi; alpha(z)=newj; changed=changed+1; ok=.true.
    end function
  end subroutine
  subroutine svc_fit(self,x,y)
    class(svc), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), allocatable :: gram(:,:),target(:)
    integer, allocatable :: ids(:)
    logical :: support(size(y))
    integer :: a,b,k,m,n,i
    real(dp) :: variance,mean
    call check_xy(x,size(y)); self%classes_=unique_labels(y); k=size(self%classes_); n=size(y)
    call require(k>=2.and.self%c>0.and.self%tol>0.and.self%max_iter>0,'invalid SVC controls')
    self%x_fit_=x; self%gamma_=self%gamma
    if(self%gamma_<0) then
      mean=sum(x)/size(x); variance=sum((x-mean)**2)/size(x)
      self%gamma_=1
      if(variance>0) self%gamma_=1/(size(x,2)*variance)
    end if
    call require(self%gamma_>=0,'invalid SVC gamma'); gram=kernel_matrix(self,x,x)
    if(allocated(self%models_)) deallocate(self%models_)
    allocate(self%models_(k*(k-1)/2)); m=0; support=.false.; self%converged_=.true.; self%n_iter_=0
    do a=1,k-1
      do b=a+1,k
        m=m+1; ids=pack([(i,i=1,n)],y==self%classes_(a).or.y==self%classes_(b))
        target=merge(-1._dp,1._dp,y(ids)==self%classes_(a))
        self%models_(m)%first=a; self%models_(m)%second=b
        call binary_fit(self%models_(m),gram(ids,ids),target,self%c,self%tol,self%max_iter)
        target=self%models_(m)%dual
        self%models_(m)%dual=spread(0._dp,1,n); self%models_(m)%dual(ids)=target
        support=support.or.abs(self%models_(m)%dual)>1.e-10_dp
        self%converged_=self%converged_.and.self%models_(m)%converged
        self%n_iter_=max(self%n_iter_,self%models_(m)%n_iter)
      end do
    end do
    self%support_=pack([(i,i=1,n)],support); self%support_vectors_=x(self%support_,:)
  end subroutine
  function svc_decision(self,x) result(z)
    class(svc), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:),gram(:,:)
    integer :: m
    call require(allocated(self%models_),'SVC is not fitted'); gram=kernel_matrix(self,x,self%x_fit_)
    allocate(z(size(x,1),size(self%models_)))
    do m=1,size(self%models_)
      z(:,m)=matmul(gram,self%models_(m)%dual)+self%models_(m)%intercept
    end do
  end function
  function svc_predict(self,x) result(y)
    class(svc), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: y(size(x,1)),i,m,winner
    integer, allocatable :: votes(:,:)
    real(dp), allocatable :: z(:,:)
    z=self%decision_function(x); allocate(votes(size(x,1),size(self%classes_))); votes=0
    do i=1,size(x,1)
      do m=1,size(self%models_)
        winner=self%models_(m)%first
        if(z(i,m)>0) winner=self%models_(m)%second
        votes(i,winner)=votes(i,winner)+1
      end do
      y(i)=self%classes_(maxloc(votes(i,:),dim=1))
    end do
  end function
  real(dp) function svc_score(self,x,y) result(v)
    class(svc), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    v=accuracy_score(y,self%predict(x))
  end function
end module
