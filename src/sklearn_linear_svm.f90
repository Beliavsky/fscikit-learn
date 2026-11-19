! LinearSVC/LinearSVR objectives: sklearn 1.6.1 svm/_classes.py and
! svm/src/liblinear/linear.cpp. BSD notices: LICENSES/scikit-learn-COPYING and liblinear-COPYRIGHT.
! Native primal active-set Newton/Armijo solver; no liblinear binary is linked.
module sklearn_linear_svm
  use sklearn_common, only: dp,require,check_x,check_xy,unique_labels,label_index,cholesky_factor,solve_lower,solve_upper
  use sklearn_metrics, only: accuracy_score,r2_score
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: linear_svc,linear_svr
  type :: linear_svc
    real(dp) :: c=1,tol=1.e-4_dp,intercept_scaling=1
    integer :: max_iter=1000,n_iter_=0,n_features_in_=0
    logical :: fit_intercept=.true.,class_weight_balanced=.false.,converged_=.false.
    integer, allocatable :: classes_(:),n_iter_per_class_(:)
    logical, allocatable :: converged_per_class_(:)
    real(dp), allocatable :: coef_(:,:),intercept_(:),objective_(:),gradient_norm_(:)
  contains
    procedure :: fit=>linear_svc_fit
    procedure :: decision_function=>linear_svc_decision
    procedure :: predict=>linear_svc_predict
    procedure :: score=>linear_svc_score
  end type
  type :: linear_svr
    real(dp) :: c=1,tol=1.e-4_dp,intercept_scaling=1,epsilon=0,intercept_=0,objective_=0,gradient_norm_=0
    integer :: max_iter=1000,n_iter_=0,n_features_in_=0
    logical :: fit_intercept=.true.,converged_=.false.
    real(dp), allocatable :: coef_(:)
  contains
    procedure :: fit=>linear_svr_fit
    procedure :: predict=>linear_svr_predict
    procedure :: score=>linear_svr_score
  end type
contains
  subroutine linear_svm_setup(x,c,tol,scaling,max_iter,intercept,a,w,sample_weight)
    real(dp), intent(in) :: x(:,:),c,tol,scaling
    integer, intent(in) :: max_iter
    logical, intent(in) :: intercept
    real(dp), allocatable, intent(out) :: a(:,:),w(:)
    real(dp), intent(in), optional :: sample_weight(:)
    integer :: n,p
    call check_x(x); n=size(x,1); p=size(x,2)
    call require(ieee_is_finite(c),'nonfinite linear SVM C')
    call require(c>0,'linear SVM C must be positive')
    call require(ieee_is_finite(tol),'nonfinite linear SVM tolerance')
    call require(tol>0,'linear SVM tolerance must be positive')
    call require(ieee_is_finite(scaling),'nonfinite linear SVM intercept scaling')
    call require(scaling>0,'linear SVM intercept scaling must be positive')
    call require(max_iter>0,'linear SVM max_iter must be positive')
    allocate(a(n,p+merge(1,0,intercept))); a(:,:p)=x
    if(intercept) a(:,p+1)=scaling
    w=spread(1._dp,1,n)
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'linear SVM sample weight count mismatch')
      call require(all(ieee_is_finite(sample_weight)),'nonfinite linear SVM sample weights')
      call require(all(sample_weight>=0),'negative linear SVM sample weights')
      w=sample_weight
    end if
    call require(sum(w)>0,'linear SVM needs positive total weight')
  end subroutine

  subroutine squared_state(a,y,cost,coef,classification,epsilon,objective,gradient,hessian)
    real(dp), intent(in) :: a(:,:),y(:),cost(:),coef(:),epsilon
    logical, intent(in) :: classification
    real(dp), intent(out) :: objective,gradient(:),hessian(:,:)
    real(dp), allocatable :: residual(:),loss_gradient(:),weights(:)
    integer :: j
    residual=matmul(a,coef)-y
    if(classification) then
      residual=max(0._dp,1-y*matmul(a,coef)); loss_gradient=-y*residual
    else
      residual=max(0._dp,abs(residual)-epsilon); loss_gradient=sign(residual,matmul(a,coef)-y)
    end if
    weights=merge(2*cost,0._dp,residual>0)
    objective=.5_dp*sum(coef**2)+sum(cost*residual**2)
    gradient=coef+2*matmul(transpose(a),cost*loss_gradient)
    hessian=matmul(transpose(a),a*spread(weights,2,size(a,2)))
    do j=1,size(coef)
      hessian(j,j)=hessian(j,j)+1
    end do
  end subroutine

  subroutine squared_solve(a,y,cost,classification,tube,tol,max_iter,coef,iterations,converged,objective,gradnorm)
    real(dp), intent(in) :: a(:,:),y(:),cost(:),tube,tol
    logical, intent(in) :: classification
    integer, intent(in) :: max_iter
    real(dp), allocatable, intent(out) :: coef(:)
    integer, intent(out) :: iterations
    logical, intent(out) :: converged
    real(dp), intent(out) :: objective,gradnorm
    real(dp), allocatable :: gradient(:),hessian(:,:),l(:,:),direction(:),candidate(:),trial_g(:),trial_h(:,:)
    real(dp) :: initial_norm,step,trial,descent,slack
    integer :: d,it,search
    d=size(a,2); coef=spread(0._dp,1,d)
    allocate(gradient(d),hessian(d,d),trial_g(d),trial_h(d,d))
    call squared_state(a,y,cost,coef,classification,tube,objective,gradient,hessian)
    initial_norm=max(1._dp,sqrt(sum(gradient**2))); iterations=0
    do it=1,max_iter
      if(sqrt(sum(gradient**2))<=tol*initial_norm) exit
      l=cholesky_factor(hessian); direction=-solve_upper(transpose(l),solve_lower(l,gradient))
      descent=dot_product(gradient,direction); step=1
      slack=10*epsilon(1._dp)*max(1._dp,abs(objective))
      do search=1,60
        candidate=coef+step*direction
        call squared_state(a,y,cost,candidate,classification,tube,trial,trial_g,trial_h)
        if(trial<=objective+1.e-4_dp*step*descent+slack) exit
        step=step*.5_dp
      end do
      call require(search<=60,'linear SVM line search failed')
      coef=candidate; objective=trial; gradient=trial_g; hessian=trial_h; iterations=it
    end do
    gradnorm=sqrt(sum(gradient**2)); converged=gradnorm<=tol*initial_norm
  end subroutine

  subroutine linear_svc_fit(self,x,y,sample_weight,class_weight)
    class(linear_svc), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(in), optional :: sample_weight(:),class_weight(:)
    real(dp), allocatable :: a(:,:),w(:),cw(:),target(:),cost(:),coef(:)
    integer, allocatable :: encoded(:)
    integer :: n,p,c,k,i,channels
    call check_xy(x,size(y)); n=size(y); p=size(x,2)
    call linear_svm_setup(x,self%c,self%tol,self%intercept_scaling,self%max_iter,self%fit_intercept,a,w,sample_weight)
    self%classes_=unique_labels(y); c=size(self%classes_)
    call require(c>=2,'LinearSVC requires at least two classes')
    encoded=[(label_index(self%classes_,y(i)),i=1,n)]; cw=spread(1._dp,1,c)
    if(present(class_weight)) then
      call require(.not.self%class_weight_balanced,'choose balanced or explicit LinearSVC class weights')
      call require(size(class_weight)==c,'LinearSVC class weight count mismatch')
      call require(all(ieee_is_finite(class_weight)),'nonfinite LinearSVC class weights')
      call require(all(class_weight>0),'LinearSVC class weights must be positive')
      cw=class_weight
    else if(self%class_weight_balanced) then
      do k=1,c
        cw(k)=real(n,dp)/(c*real(count(encoded==k),dp))
      end do
    end if
    channels=c
    if(c==2) channels=1
    if(allocated(self%coef_)) deallocate(self%coef_,self%intercept_,self%objective_,self%gradient_norm_, &
      self%n_iter_per_class_,self%converged_per_class_)
    allocate(self%coef_(channels,p),self%intercept_(channels),self%objective_(channels),self%gradient_norm_(channels))
    allocate(self%n_iter_per_class_(channels),self%converged_per_class_(channels))
    do k=1,channels
      if(c==2) then
        target=merge(1._dp,-1._dp,encoded==2); cost=self%c*w*cw(encoded)
      else
        target=merge(1._dp,-1._dp,encoded==k)
        ! liblinear OVR applies class weights to the positive class; negative cost stays C.
        cost=self%c*w*merge(cw(k),1._dp,encoded==k)
      end if
      call squared_solve(a,target,cost,.true.,0._dp,self%tol,self%max_iter,coef,self%n_iter_per_class_(k), &
        self%converged_per_class_(k),self%objective_(k),self%gradient_norm_(k))
      self%coef_(k,:)=coef(:p); self%intercept_(k)=0
      if(self%fit_intercept) self%intercept_(k)=coef(p+1)*self%intercept_scaling
    end do
    self%n_features_in_=p; self%n_iter_=maxval(self%n_iter_per_class_); self%converged_=all(self%converged_per_class_)
  end subroutine

  function linear_svc_decision(self,x) result(scores)
    class(linear_svc), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: scores(:,:)
    call require(allocated(self%coef_),'LinearSVC is not fitted'); call check_x(x)
    call require(size(x,2)==self%n_features_in_,'LinearSVC feature count mismatch')
    scores=matmul(x,transpose(self%coef_))+spread(self%intercept_,1,size(x,1))
  end function

  function linear_svc_predict(self,x) result(y)
    class(linear_svc), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: y(:)
    real(dp), allocatable :: scores(:,:)
    scores=self%decision_function(x)
    if(size(self%classes_)==2) then
      y=self%classes_(merge(2,1,scores(:,1)>0))
    else
      y=self%classes_(maxloc(scores,dim=2))
    end if
  end function

  real(dp) function linear_svc_score(self,x,y) result(value)
    class(linear_svc), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    value=accuracy_score(y,self%predict(x))
  end function

  subroutine linear_svr_fit(self,x,y,sample_weight)
    class(linear_svr), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: a(:,:),w(:),coef(:)
    integer :: p
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1])); p=size(x,2)
    call require(ieee_is_finite(self%epsilon),'nonfinite LinearSVR epsilon')
    call require(self%epsilon>=0,'LinearSVR epsilon must be nonnegative')
    call linear_svm_setup(x,self%c,self%tol,self%intercept_scaling,self%max_iter,self%fit_intercept,a,w,sample_weight)
    call squared_solve(a,y,self%c*w,.false.,self%epsilon,self%tol,self%max_iter,coef,self%n_iter_, &
      self%converged_,self%objective_,self%gradient_norm_)
    self%coef_=coef(:p); self%intercept_=0
    if(self%fit_intercept) self%intercept_=coef(p+1)*self%intercept_scaling
    self%n_features_in_=p
  end subroutine

  function linear_svr_predict(self,x) result(y)
    class(linear_svr), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:)
    call require(allocated(self%coef_),'LinearSVR is not fitted'); call check_x(x)
    call require(size(x,2)==self%n_features_in_,'LinearSVR feature count mismatch')
    y=matmul(x,self%coef_)+self%intercept_
  end function

  real(dp) function linear_svr_score(self,x,y) result(value)
    class(linear_svr), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    value=r2_score(y,self%predict(x))
  end function
end module
