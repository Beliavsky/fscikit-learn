! Laplace GPC: sklearn 1.6.1 gaussian_process/_gpc.py (GPML Algorithms 3.1, 3.2 and 5.1).
! BSD-3-Clause attribution retained in LICENSES/scikit-learn-COPYING.
module sklearn_gp_classification
  use sklearn_common, only: dp,require,check_x,check_xy,unique_labels,sigmoid,cholesky_factor,solve_lower,solve_upper
  use sklearn_metrics, only: accuracy_score
  use sklearn_gp_kernels, only: gp_kernel,rbf_covariance_kernel=>rbf_covariance,constant_covariance,operator(*)
  use sklearn_legacy_rng, only: legacy_random_state
  use lbfgsb_module, only: setulb
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: iso_c_binding, only: c_double
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_value,ieee_negative_inf
  implicit none
  private
  public :: gaussian_process_classifier
  real(dp), parameter :: lambdas(5)=[.41_dp,.4_dp,.37_dp,.44_dp,.39_dp]
  real(dp), parameter :: coefs(5)=[-1854.8214151_dp,3516.89893646_dp,221.29346712_dp,128.12323805_dp,-2010.49422654_dp]
  interface
    function c_log1p(x) bind(C,name='log1p') result(value)
      import c_double
      real(c_double), value :: x
      real(c_double) :: value
    end function
  end interface
  type :: laplace_state
    real(dp), allocatable :: f(:),pi(:),root(:),l(:,:),a(:)
    real(dp) :: value=0
    integer :: iterations=0
    logical :: converged=.false.
  end type
  type :: gaussian_process_classifier
    real(dp) :: amplitude=1,length_scale=1,log_marginal_likelihood_=0
    type(gp_kernel) :: kernel
    type(gp_kernel), allocatable :: kernels_(:)
    character(24) :: optimizer='none'
    integer :: n_restarts_optimizer=0,random_state=0,max_iter=15000,max_fun=15000
    real(dp) :: tol=1.e-5_dp,ftol=2.220446049250313e-9_dp
    logical :: warm_start=.false.
    integer :: max_iter_predict=100,n_features_in_=0,n_classes_=0,n_optimization_runs_=0
    integer, allocatable :: classes_(:),n_iter_(:),n_optimizer_iter_(:),n_function_evaluations_(:)
    logical, allocatable :: converged_(:),optimizer_converged_(:)
    character(60), allocatable :: optimization_messages_(:)
    character(60), allocatable :: optimization_run_messages_(:,:)
    logical, allocatable :: optimization_run_converged_(:,:)
    real(dp), allocatable :: x_train_(:,:),posterior_mode_(:,:),pi_(:,:),w_sqrt_(:,:),l_(:,:,:),alpha_(:,:)
    real(dp), allocatable :: log_marginal_likelihoods_(:),mode_cache_(:,:),optimization_values_(:,:),optimization_points_(:,:,:)
    real(dp), allocatable :: fit_initial_cache_(:,:)
    real(dp), allocatable :: optimization_eval_cache_(:,:,:)
    real(dp), allocatable :: optimization_initial_points_(:,:,:)
    real(dp), allocatable, private :: targets_(:,:)
    logical, private :: fitted_warm_start_=.false.
    integer, private :: fitted_mode_limit_=100
  contains
    procedure :: fit=>gpc_fit
    procedure :: predict_proba=>gpc_proba
    procedure :: predict=>gpc_predict
    procedure :: score=>gpc_score
    procedure, private :: likelihood_cached=>gpc_likelihood_cached,likelihood_theta=>gpc_likelihood_theta
    generic :: log_marginal_likelihood=>likelihood_cached,likelihood_theta
  end type
contains
  recursive real(dp) function numpy_sum(values) result(total)
    ! NumPy 2.2.6 loops_utils.h.src real pairwise reduction (retained BSD notice).
    real(dp), intent(in) :: values(:)
    real(dp) :: partial(8)
    integer :: n,i,split,whole
    n=size(values)
    if(n<8) then
      total=-0._dp
      do i=1,n
        total=total+values(i)
      end do
    else if(n<=128) then
      partial=values(:8);whole=n-mod(n,8)
      do i=9,whole,8
        partial=partial+values(i:i+7)
      end do
      total=((partial(1)+partial(2))+(partial(3)+partial(4))) &
        +((partial(5)+partial(6))+(partial(7)+partial(8)))
      do i=whole+1,n
        total=total+values(i)
      end do
    else
      split=n/2;split=split-mod(split,8)
      total=numpy_sum(values(:split))+numpy_sum(values(split+1:))
    end if
  end function
  function configured_kernel(self) result(kernel)
    class(gaussian_process_classifier), intent(in) :: self
    type(gp_kernel) :: kernel
    if(self%kernel%is_set()) then
      kernel=self%kernel
    else
      call require(ieee_is_finite(self%amplitude),'nonfinite GPC amplitude')
      call require(self%amplitude>0,'GPC amplitude must be positive')
      call require(ieee_is_finite(self%length_scale),'nonfinite GPC length scale')
      call require(self%length_scale>0,'GPC length scale must be positive')
      kernel=constant_covariance(self%amplitude,.true.)*rbf_covariance_kernel(self%length_scale,.true.)
    end if
  end function
  subroutine validate_controls(self)
    class(gaussian_process_classifier), intent(in) :: self
    call require(self%max_iter_predict>0,'GPC max_iter_predict must be positive')
    call require(any(self%optimizer==[character(24)::'none','fmin_l_bfgs_b']),'GPC unsupported optimizer')
    call require(self%random_state>=0,'GPC random_state must be nonnegative')
    call require(self%n_restarts_optimizer>=0,'GPC optimizer restart count must be nonnegative')
    call require(self%max_iter>0.and.self%max_fun>0,'GPC optimizer iteration and evaluation limits must be positive')
    call require(ieee_is_finite(self%tol).and.ieee_is_finite(self%ftol),'GPC optimizer tolerances must be finite')
    call require(self%tol>=0.and.self%ftol>=0,'GPC optimizer tolerances must be nonnegative')
    call require(self%ftol<=huge(1._dp)*epsilon(1._dp),'GPC optimizer ftol exceeds working range')
  end subroutine
  function posterior_mode(gram,target,max_iter,warm,cache) result(state)
    real(dp), intent(in) :: gram(:,:),target(:)
    integer, intent(in) :: max_iter
    logical, intent(in) :: warm
    real(dp), intent(inout) :: cache(:)
    type(laplace_state) :: state
    real(dp), allocatable :: f(:),w(:),bmat(:,:),b(:),z(:),loss(:),logs(:)
    real(dp) :: previous,current
    integer :: n,j,iteration
    n=size(target);allocate(f(n),bmat(n,n),loss(n),logs(n));f=0
    if(warm) f=cache
    previous=ieee_value(0._dp,ieee_negative_inf)
    do iteration=1,max_iter
      state%pi=sigmoid(f);w=state%pi*(1-state%pi);state%root=sqrt(w)
      bmat=gram*spread(state%root,2,n)*spread(state%root,1,n)
      do j=1,n
        bmat(j,j)=bmat(j,j)+1
      end do
      state%l=cholesky_factor(bmat);b=w*f+(target-state%pi)
      z=matmul(gram*spread(state%root,2,n),b)
      state%a=b-state%root*solve_upper(transpose(state%l),solve_lower(state%l,z));f=matmul(gram,state%a)
      z=-(2*target-1)*f
      do j=1,n
        if(z(j)>log(huge(1._dp))) then
          loss(j)=z(j)
        else
          loss(j)=c_log1p(exp(z(j)))
        end if
        logs(j)=log(state%l(j,j))
      end do
      current=-.5_dp*dot_product(state%a,f)-numpy_sum(loss)-numpy_sum(logs)
      if(iteration>1) then
        if(current-previous<1.e-10_dp) then
          state%converged=.true.;exit
        end if
      end if
      previous=current
    end do
    state%f=f;cache=f;state%value=previous;state%iterations=min(iteration,max_iter)
  end function
  real(dp) function laplace_value(kernel,x,target,max_iter,warm,cache,gradient) result(value)
    type(gp_kernel), intent(in) :: kernel
    real(dp), intent(in) :: x(:,:),target(:)
    integer, intent(in) :: max_iter
    logical, intent(in) :: warm
    real(dp), intent(inout) :: cache(:)
    real(dp), allocatable, optional, intent(out) :: gradient(:)
    type(laplace_state) :: state
    real(dp), allocatable :: gram(:,:),derivatives(:,:,:),r(:,:),c(:,:),s2(:),s3(:),b(:),rhs(:)
    real(dp) :: s1
    integer :: n,d,i,j
    gram=kernel%matrix(x);state=posterior_mode(gram,target,max_iter,warm,cache);value=state%value
    if(.not.present(gradient)) return
    d=kernel%n_dims();allocate(gradient(d));if(d==0) return
    n=size(target);allocate(r(n,n),c(n,n),rhs(n),s2(n));derivatives=kernel%gradient(x)
    do i=1,n
      rhs=0;rhs(i)=state%root(i)
      r(:,i)=state%root*solve_upper(transpose(state%l),solve_lower(state%l,rhs))
      c(:,i)=solve_lower(state%l,state%root*gram(:,i))
    end do
    do i=1,n
      s2(i)=-.5_dp*(gram(i,i)-sum(c(:,i)**2))*state%pi(i)*(1-state%pi(i))*(1-2*state%pi(i))
    end do
    do j=1,d
      s1=.5_dp*dot_product(matmul(state%a,derivatives(:,:,j)),state%a) &
        -.5_dp*dot_product(reshape(r,[n*n]),reshape(transpose(derivatives(:,:,j)),[n*n]))
      b=matmul(derivatives(:,:,j),target-state%pi);s3=b-matmul(gram,matmul(r,b))
      gradient(j)=s1+dot_product(s2,s3)
    end do
  end function
  subroutine optimize_one(self,channel,initial,point,bounds,value,iterations,evaluations,message,evaluation_cache)
    class(gaussian_process_classifier), intent(inout) :: self
    integer, intent(in) :: channel
    type(gp_kernel), intent(in) :: initial
    real(dp), intent(inout) :: point(:)
    real(dp), intent(in) :: bounds(:,:)
    real(dp), intent(out) :: value
    integer, intent(out) :: iterations,evaluations
    character(60), intent(out) :: message
    real(dp), intent(out) :: evaluation_cache(:)
    integer, parameter :: m=10
    type(gp_kernel) :: candidate
    real(dp), allocatable :: wa(:),g(:),derivative(:)
    real(dp), allocatable :: last_cache(:),accepted_cache(:),last_point(:),accepted_point(:)
    integer, allocatable :: iwa(:),nbd(:)
    real(dp) :: f,dsave(29)
    real(dp) :: last_value,accepted_value
    integer :: n,i,isave(44)
    logical :: lsave(4)
    logical :: has_evaluation,has_accepted
    character(60) :: task,csave
    n=size(point)
    call require(int(n,int64)*(2*m+5)+11*m*m+8*m<=huge(n),'GPC optimizer workspace exceeds integer capacity')
    allocate(wa((2*m+5)*n+11*m*m+8*m),iwa(3*n),nbd(n),g(n));nbd=0
    do i=1,n
      if(ieee_is_finite(bounds(i,1))) nbd(i)=1
      if(ieee_is_finite(bounds(i,2))) nbd(i)=3
      if(all(ieee_is_finite(bounds(i,:)))) nbd(i)=2
      point(i)=max(bounds(i,1),min(bounds(i,2),point(i)))
    end do
    wa=0;iwa=0;isave=0;dsave=0;lsave=.false.;csave='';task='START';f=0;g=0;iterations=0;evaluations=0
    has_evaluation=.false.;has_accepted=.false.
    do
      call setulb(n,m,point,bounds(:,1),bounds(:,2),nbd,f,g,self%ftol/epsilon(1._dp),self%tol, &
        wa,iwa,task,-1,csave,lsave,isave,dsave)
      if(task(:2)=='FG') then
        last_cache=self%mode_cache_(:,channel);last_point=point
        candidate=initial%with_theta(point)
        f=-laplace_value(candidate,self%x_train_,self%targets_(:,channel),self%max_iter_predict, &
          self%warm_start,self%mode_cache_(:,channel),derivative);g=-derivative;evaluations=evaluations+1
        last_value=f;has_evaluation=.true.
      else if(task(:5)=='NEW_X') then
        accepted_cache=last_cache;accepted_point=point;accepted_value=f;has_accepted=.true.
        iterations=iterations+1
        if(iterations>=self%max_iter) then
          task='STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT'
        else if(evaluations>self%max_fun) then
          task='STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT'
        end if
      else
        exit
      end if
    end do
    if(has_evaluation) then
      if(all(point==last_point).and.f==last_value) then
        evaluation_cache=last_cache
      else
        call require(has_accepted,'GPC optimizer has no accepted objective state')
        call require(all(point==accepted_point).and.f==accepted_value,'GPC optimizer objective state mismatch')
        evaluation_cache=accepted_cache
      end if
    else
      call require(.false.,'GPC optimizer performed no objective evaluation')
    end if
    value=f;message=task
  end subroutine
  subroutine optimize_channel(self,channel,initial)
    class(gaussian_process_classifier), intent(inout) :: self
    integer, intent(in) :: channel
    type(gp_kernel), intent(in) :: initial
    type(legacy_random_state) :: rng
    real(dp), allocatable :: bounds(:,:),point(:)
    integer :: run,i,best,iterations,evaluations
    character(60) :: message
    if(self%n_optimization_runs_==0) then
      self%kernels_(channel)=initial
      self%log_marginal_likelihoods_(channel)=laplace_value(initial,self%x_train_,self%targets_(:,channel), &
        self%max_iter_predict,self%warm_start,self%mode_cache_(:,channel));return
    end if
    bounds=initial%bounds();call rng%seed(self%random_state);best=1
    if(self%n_restarts_optimizer>0) call require(all(ieee_is_finite(bounds)),'GPC optimizer restarts require finite bounds')
    do run=1,self%n_optimization_runs_
      point=initial%theta()
      if(run>1) then
        do i=1,size(point)
          ! Pinned sklearn 1.6.1 GPC exponentiates a log-uniform draw before
          ! passing it as log theta (unlike GPR); the solver clips it to bounds.
          point(i)=exp(bounds(i,1)+(bounds(i,2)-bounds(i,1))*rng%uniform())
        end do
      end if
      self%optimization_initial_points_(run,:,channel)=point
      call optimize_one(self,channel,initial,point,bounds,self%optimization_values_(run,channel), &
        iterations,evaluations,message,self%optimization_eval_cache_(:,run,channel))
      self%optimization_points_(run,:,channel)=point
      self%optimization_run_messages_(run,channel)=message
      self%optimization_run_converged_(run,channel)=message(:4)=='CONV'
      self%n_optimizer_iter_(channel)=self%n_optimizer_iter_(channel)+iterations
      self%n_function_evaluations_(channel)=self%n_function_evaluations_(channel)+evaluations
      if(self%optimization_values_(run,channel)<self%optimization_values_(best,channel)) best=run
      if(best==run) then
        self%optimization_messages_(channel)=message;self%optimizer_converged_(channel)=message(:4)=='CONV'
      end if
    end do
    self%kernels_(channel)=initial%with_theta(self%optimization_points_(best,:,channel))
    self%log_marginal_likelihoods_(channel)=-self%optimization_values_(best,channel)
  end subroutine
  subroutine gpc_fit(self,x,y)
    class(gaussian_process_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    type(gp_kernel) :: initial
    type(laplace_state) :: state
    integer :: n,c,channels,k,d
    call check_xy(x,size(y));initial=configured_kernel(self);call validate_controls(self)
    self%classes_=unique_labels(y);c=size(self%classes_);call require(c>=2,'GPC requires at least two classes')
    n=size(y);channels=c;if(c==2) channels=1
    self%x_train_=x;self%n_features_in_=size(x,2);self%n_classes_=c;d=initial%n_dims();self%n_optimization_runs_=0
    self%fitted_warm_start_=self%warm_start;self%fitted_mode_limit_=self%max_iter_predict
    if(self%optimizer/='none'.and.d>0) self%n_optimization_runs_=self%n_restarts_optimizer+1
    if(allocated(self%pi_)) then
      deallocate(self%pi_,self%posterior_mode_,self%w_sqrt_,self%l_,self%alpha_,self%n_iter_,self%converged_, &
        self%log_marginal_likelihoods_,self%kernels_,self%mode_cache_,self%targets_,self%optimization_values_, &
        self%optimization_points_,self%n_optimizer_iter_,self%n_function_evaluations_,self%optimizer_converged_, &
        self%optimization_messages_,self%optimization_run_messages_,self%optimization_run_converged_,self%fit_initial_cache_, &
        self%optimization_eval_cache_,self%optimization_initial_points_)
    end if
    allocate(self%pi_(n,channels),self%posterior_mode_(n,channels),self%w_sqrt_(n,channels),self%mode_cache_(n,channels))
    allocate(self%targets_(n,channels),self%kernels_(channels),self%l_(n,n,channels),self%alpha_(n,channels))
    allocate(self%fit_initial_cache_(n,channels))
    allocate(self%n_iter_(channels),self%converged_(channels),self%log_marginal_likelihoods_(channels))
    allocate(self%n_optimizer_iter_(channels),self%n_function_evaluations_(channels),self%optimizer_converged_(channels))
    allocate(self%optimization_messages_(channels),self%optimization_values_(self%n_optimization_runs_,channels), &
      self%optimization_points_(self%n_optimization_runs_,d,channels))
    allocate(self%optimization_run_messages_(self%n_optimization_runs_,channels), &
      self%optimization_run_converged_(self%n_optimization_runs_,channels))
    allocate(self%optimization_eval_cache_(n,self%n_optimization_runs_,channels))
    allocate(self%optimization_initial_points_(self%n_optimization_runs_,d,channels))
    self%mode_cache_=0;self%n_optimizer_iter_=0;self%n_function_evaluations_=0
    self%optimizer_converged_=.true.;self%optimization_messages_='not requested'
    do k=1,channels
      if(c==2) then
        self%targets_(:,k)=merge(1._dp,0._dp,y==self%classes_(2))
      else
        self%targets_(:,k)=merge(1._dp,0._dp,y==self%classes_(k))
      end if
      call optimize_channel(self,k,initial)
      self%fit_initial_cache_(:,k)=self%mode_cache_(:,k)
      state=posterior_mode(self%kernels_(k)%matrix(x),self%targets_(:,k),self%max_iter_predict, &
        self%warm_start,self%mode_cache_(:,k))
      self%posterior_mode_(:,k)=state%f;self%pi_(:,k)=state%pi;self%w_sqrt_(:,k)=state%root
      self%l_(:,:,k)=state%l;self%alpha_(:,k)=self%targets_(:,k)-state%pi
      self%n_iter_(k)=state%iterations;self%converged_(k)=state%converged
    end do
    self%log_marginal_likelihood_=sum(self%log_marginal_likelihoods_)/channels
  end subroutine
  subroutine check_query(self,x)
    class(gaussian_process_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    call require(allocated(self%pi_),'GPC is not fitted');call check_x(x)
    call require(size(x,2)==self%n_features_in_,'GPC feature count mismatch')
  end subroutine
  function gpc_cross(self,x,channel) result(k)
    class(gaussian_process_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: channel
    real(dp), allocatable :: k(:,:)
    call check_query(self,x);k=self%kernels_(channel)%matrix(self%x_train_,x)
  end function
  function gpc_proba(self,x) result(probability)
    class(gaussian_process_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: probability(:,:),cross(:,:),mean(:),v(:),positive(:,:),diagonal(:)
    real(dp) :: variance,value
    integer :: i,k
    call check_query(self,x);allocate(positive(size(x,1),size(self%pi_,2)))
    do k=1,size(self%pi_,2)
      cross=self%kernels_(k)%matrix(self%x_train_,x);diagonal=self%kernels_(k)%diag(x)
      mean=matmul(transpose(cross),self%alpha_(:,k))
      do i=1,size(x,1)
        v=solve_lower(self%l_(:,:,k),self%w_sqrt_(:,k)*cross(:,i))
        variance=max(0._dp,diagonal(i)-dot_product(v,v))
        ! Algebraic simplification avoids division by predictive variance near zero.
        value=.5_dp*sum(coefs*erf(lambdas*mean(i)/sqrt(1+2*variance*lambdas**2)))+.5_dp*sum(coefs)
        positive(i,k)=max(0._dp,min(1._dp,value))
      end do
    end do
    if(self%n_classes_==2) then
      allocate(probability(size(x,1),2)); probability(:,2)=positive(:,1); probability(:,1)=1-positive(:,1)
    else
      probability=positive
      do i=1,size(x,1)
        value=sum(positive(i,:))
        if(value>0) then
          probability(i,:)=positive(i,:)/value
        else
          probability(i,:)=1._dp/self%n_classes_
        end if
      end do
    end if
  end function

  function gpc_predict(self,x) result(y)
    class(gaussian_process_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: y(:)
    real(dp), allocatable :: cross(:,:),mean(:),probability(:,:)
    if(self%n_classes_==2) then
      cross=gpc_cross(self,x,1); mean=matmul(transpose(cross),self%alpha_(:,1))
      y=self%classes_(merge(2,1,mean>0))
    else
      probability=self%predict_proba(x)
      ! sklearn OneVsRestClassifier assigns exact probability ties to the last class.
      y=self%classes_(maxloc(probability,dim=2,back=.true.))
    end if
  end function

  real(dp) function gpc_score(self,x,y) result(score)
    class(gaussian_process_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    score=accuracy_score(y,self%predict(x))
  end function
  real(dp) function gpc_likelihood_cached(self,eval_gradient) result(value)
    class(gaussian_process_classifier), intent(in) :: self
    logical, optional, intent(in) :: eval_gradient
    call require(allocated(self%pi_),'GPC is not fitted')
    if(present(eval_gradient)) call require(.not.eval_gradient,'GPC likelihood gradient requires theta')
    value=self%log_marginal_likelihood_
  end function
  real(dp) function gpc_likelihood_theta(self,theta,gradient,clone_kernel) result(value)
    class(gaussian_process_classifier), intent(inout) :: self
    real(dp), intent(in) :: theta(:)
    real(dp), allocatable, optional, intent(out) :: gradient(:)
    logical, optional, intent(in) :: clone_kernel
    type(gp_kernel) :: candidate
    real(dp), allocatable :: part(:),values(:)
    logical :: clone,shared
    integer :: channels,d,k
    call require(allocated(self%pi_),'GPC is not fitted');channels=size(self%kernels_);d=self%kernels_(1)%n_dims()
    if(channels>1) call require(.not.present(gradient),'GPC multiclass likelihood gradients are not supported')
    call require(size(theta)==d.or.size(theta)==d*channels,'GPC theta must match one kernel or all class kernels')
    shared=size(theta)==d;clone=.true.;if(present(clone_kernel)) clone=clone_kernel
    allocate(values(channels))
    do k=1,channels
      if(shared) then
        part=theta
      else
        part=theta((k-1)*d+1:k*d)
      end if
      candidate=self%kernels_(k)%with_theta(part)
      if(.not.clone) self%kernels_(k)=candidate
      values(k)=laplace_value(candidate,self%x_train_,self%targets_(:,k),self%fitted_mode_limit_, &
        self%fitted_warm_start_,self%mode_cache_(:,k),gradient)
    end do
    value=sum(values)/channels
  end function
end module
