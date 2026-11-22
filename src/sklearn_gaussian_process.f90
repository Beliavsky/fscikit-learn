! Dense GPR contracts from sklearn 1.6.1 gaussian_process/_gpr.py (BSD-3-Clause).
! Native storage and optimizer controls are original MIT contributions.
module sklearn_gaussian_process
  use sklearn_gp_classification, only: gaussian_process_classifier
  use sklearn_common, only: dp,require,check_x,check_xy,solve_lower,solve_upper
  use sklearn_gp_kernels, only: gp_kernel,rbf_covariance,constant_covariance,white_covariance,operator(*),operator(+)
  use sklearn_legacy_rng, only: legacy_random_state
  use lbfgsb_module, only: setulb
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_value,ieee_negative_inf,ieee_quiet_nan
  implicit none
  private
  public :: gaussian_process_classifier,gaussian_process_regressor
  type :: gaussian_process_regressor
    real(dp) :: alpha=1.e-10_dp,length_scale=1,amplitude=1,noise_level=0
    real(dp), allocatable :: alpha_values(:)
    logical :: normalize_y=.false.
    type(gp_kernel) :: kernel,kernel_
    character(24) :: optimizer='none'
    integer :: n_restarts_optimizer=0,random_state=0,max_iter=15000,max_fun=15000,n_targets=0
    real(dp) :: tol=1.e-5_dp,ftol=2.220446049250313e-9_dp
    integer :: n_targets_=1,n_features_in_=0,n_optimizer_iter_=0,n_function_evaluations_=0
    integer :: n_optimization_runs_=0
    logical :: converged_=.true.
    character(60) :: optimization_message_='not requested'
    real(dp) :: y_train_mean_=0,y_train_std_=1,log_marginal_likelihood_=0
    real(dp), allocatable :: x_train_(:,:),y_train_(:,:),alpha_(:),alpha_multi_(:,:),l_(:,:)
    real(dp), allocatable :: y_train_means_(:),y_train_stds_(:),optimization_values_(:),optimization_points_(:,:)
  contains
    procedure, private :: fit_vector=>gp_fit_vector,fit_matrix=>gp_fit_matrix
    generic :: fit=>fit_vector,fit_matrix
    procedure :: predict=>gp_predict
    procedure :: predict_multi=>gp_predict_multi
    procedure :: predict_std=>gp_predict_std
    procedure :: predict_std_multi=>gp_predict_std_multi
    procedure :: predict_cov=>gp_predict_cov
    procedure :: predict_cov_multi=>gp_predict_cov_multi
    procedure :: log_marginal_likelihood=>gp_likelihood
    procedure, private :: score_vector=>gp_score_vector,score_matrix=>gp_score_matrix
    generic :: score=>score_vector,score_matrix
  end type
contains
  recursive real(dp) function numpy_sum(values) result(total)
    ! NumPy 2.2.6 loops_utils.h.src pairwise reduction; retained NumPy BSD notice.
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
    class(gaussian_process_regressor), intent(in) :: self
    type(gp_kernel) :: kernel
    if(self%kernel%is_set()) then
      kernel=self%kernel
    else
      call require(all(ieee_is_finite([self%length_scale,self%amplitude,self%noise_level])), &
        'invalid Gaussian process kernel parameters')
      call require(self%noise_level>=0.and.self%length_scale>0.and.self%amplitude>0, &
        'invalid Gaussian process kernel parameters')
      kernel=constant_covariance(self%amplitude,.true.)*rbf_covariance(self%length_scale,.true.)+ &
        white_covariance(self%noise_level,.true.)
    end if
  end function
  function noise_variance(self,n) result(variance)
    class(gaussian_process_regressor), intent(in) :: self
    integer, intent(in) :: n
    real(dp), allocatable :: variance(:)
    allocate(variance(n))
    if(allocated(self%alpha_values)) then
      call require(size(self%alpha_values)==1.or.size(self%alpha_values)==n,'GP alpha_values sample count mismatch')
      call require(all(ieee_is_finite(self%alpha_values)),'GP alpha_values must be finite')
      call require(all(self%alpha_values>=0),'GP alpha_values must be nonnegative')
      variance=self%alpha_values(1)
      if(size(self%alpha_values)==n) variance=self%alpha_values
    else
      call require(ieee_is_finite(self%alpha),'invalid Gaussian process alpha')
      call require(self%alpha>=0,'invalid Gaussian process alpha');variance=self%alpha
    end if
  end function
  function covariance(self,x,y) result(k)
    class(gaussian_process_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: y(:,:)
    real(dp), allocatable :: k(:,:)
    type(gp_kernel) :: kernel
    if(allocated(self%alpha_multi_)) then
      kernel=self%kernel_
    else
      kernel=configured_kernel(self)
    end if
    k=kernel%matrix(x,y)
  end function
  real(dp) function likelihood_value(kernel,x,target,variance,gradient,l_out,coef_out) result(value)
    type(gp_kernel), intent(in) :: kernel
    real(dp), intent(in) :: x(:,:),target(:,:),variance(:)
    real(dp), allocatable, optional, intent(out) :: gradient(:),l_out(:,:),coef_out(:,:)
    real(dp), allocatable :: gram(:,:),l(:,:),coef(:,:),inverse(:,:),inner(:,:),derivatives(:,:,:),rhs(:)
    real(dp) :: pivot,logdet
    integer :: n,q,i,j,k,d
    n=size(x,1);q=size(target,2);value=ieee_value(0._dp,ieee_negative_inf)
    if(present(gradient)) then
      d=kernel%n_dims();allocate(gradient(d));gradient=0
    end if
    gram=kernel%matrix(x);allocate(l(n,n));l=0
    do i=1,n
      gram(i,i)=gram(i,i)+variance(i)
    end do
    do i=1,n
      do j=1,i
        pivot=gram(i,j)-dot_product(l(i,:j-1),l(j,:j-1))
        if(.not.ieee_is_finite(pivot)) return
        if(i==j) then
          if(pivot<=0) return
          l(i,j)=sqrt(pivot)
        else
          l(i,j)=pivot/l(j,j)
        end if
      end do
    end do
    allocate(coef(n,q));logdet=0
    do j=1,q
      coef(:,j)=solve_upper(transpose(l),solve_lower(l,target(:,j)))
    end do
    do i=1,n
      logdet=logdet+log(l(i,i))
    end do
    value=-.5_dp*sum(target*coef)-q*logdet-.5_dp*n*q*log(2*acos(-1._dp))
    if(present(l_out)) l_out=l
    if(present(coef_out)) coef_out=coef
    if(present(gradient)) then
      if(d==0) return
      allocate(inverse(n,n),rhs(n))
      do i=1,n
        rhs=0;rhs(i)=1;inverse(:,i)=solve_upper(transpose(l),solve_lower(l,rhs))
      end do
      inner=matmul(coef,transpose(coef))-q*inverse;derivatives=kernel%gradient(x)
      do k=1,d
        gradient(k)=.5_dp*sum(inner*derivatives(:,:,k))
      end do
    end if
  end function
  subroutine validate_controls(self)
    class(gaussian_process_regressor), intent(in) :: self
    call require(any(self%optimizer==[character(24)::'none','fmin_l_bfgs_b']),'GP unsupported optimizer')
    call require(self%n_targets>=0,'GP n_targets must be nonnegative (zero means automatic)')
    call require(self%random_state>=0,'GP random_state must be nonnegative')
    call require(self%n_restarts_optimizer>=0,'GP optimizer restart count must be nonnegative')
    call require(self%max_iter>0.and.self%max_fun>0,'GP optimizer iteration and evaluation limits must be positive')
    call require(ieee_is_finite(self%tol).and.ieee_is_finite(self%ftol),'GP optimizer tolerances must be finite')
    call require(self%tol>=0.and.self%ftol>=0,'GP optimizer tolerances must be nonnegative')
    call require(self%ftol<=huge(1._dp)*epsilon(1._dp),'GP optimizer ftol exceeds working range')
  end subroutine
  subroutine optimize_one(self,kernel,variance,point,bounds,value,iterations,evaluations,message)
    class(gaussian_process_regressor), intent(in) :: self
    type(gp_kernel), intent(in) :: kernel
    real(dp), intent(in) :: variance(:),bounds(:,:)
    real(dp), intent(inout) :: point(:)
    real(dp), intent(out) :: value
    integer, intent(out) :: iterations,evaluations
    character(60), intent(out) :: message
    integer, parameter :: m=10
    type(gp_kernel) :: candidate
    real(dp), allocatable :: wa(:),g(:),derivative(:)
    integer, allocatable :: iwa(:),nbd(:)
    real(dp) :: dsave(29),f
    integer :: isave(44),n,i
    logical :: lsave(4)
    character(60) :: task,csave
    n=size(point)
    call require(int(n,int64)*(2*m+5)+11*m*m+8*m<=huge(n),'GP optimizer workspace exceeds integer capacity')
    allocate(wa((2*m+5)*n+11*m*m+8*m),iwa(3*n),nbd(n),g(n))
    nbd=0
    do i=1,n
      if(ieee_is_finite(bounds(i,1))) nbd(i)=1
      if(ieee_is_finite(bounds(i,2))) nbd(i)=3
      if(all(ieee_is_finite(bounds(i,:)))) nbd(i)=2
      point(i)=max(bounds(i,1),min(bounds(i,2),point(i)))
    end do
    wa=0;iwa=0;isave=0;dsave=0;lsave=.false.;csave='';task='START';f=0;g=0;iterations=0;evaluations=0
    do
      call setulb(n,m,point,bounds(:,1),bounds(:,2),nbd,f,g,self%ftol/epsilon(1._dp),self%tol, &
        wa,iwa,task,-1,csave,lsave,isave,dsave)
      if(task(:2)=='FG') then
        candidate=kernel%with_theta(point)
        f=-likelihood_value(candidate,self%x_train_,self%y_train_,variance,derivative);g=-derivative
        evaluations=evaluations+1
      else if(task(:5)=='NEW_X') then
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
    value=f;message=task
  end subroutine
  subroutine optimize_kernel(self,variance)
    class(gaussian_process_regressor), intent(inout) :: self
    real(dp), intent(in) :: variance(:)
    type(gp_kernel) :: initial
    type(legacy_random_state) :: rng
    real(dp), allocatable :: bounds(:,:),point(:)
    character(60) :: message
    integer :: run,d,i,best,iterations,evaluations
    self%n_optimizer_iter_=0;self%n_function_evaluations_=0;self%n_optimization_runs_=0
    self%converged_=.true.;self%optimization_message_='not requested'
    if(allocated(self%optimization_values_)) deallocate(self%optimization_values_,self%optimization_points_)
    if(self%optimizer=='none'.or.self%kernel_%n_dims()==0) then
      allocate(self%optimization_values_(0),self%optimization_points_(0,self%kernel_%n_dims()));return
    end if
    initial=self%kernel_;d=initial%n_dims();bounds=initial%bounds();call rng%seed(self%random_state)
    if(self%n_restarts_optimizer>0) call require(all(ieee_is_finite(bounds)), &
      'GP optimizer restarts require finite bounds')
    allocate(self%optimization_values_(self%n_restarts_optimizer+1), &
      self%optimization_points_(self%n_restarts_optimizer+1,d));best=1
    do run=1,self%n_restarts_optimizer+1
      point=initial%theta()
      if(run>1) then
        do i=1,d
          point(i)=bounds(i,1)+(bounds(i,2)-bounds(i,1))*rng%uniform()
        end do
      end if
      call optimize_one(self,initial,variance,point,bounds,self%optimization_values_(run),iterations,evaluations,message)
      self%optimization_points_(run,:)=point;self%n_optimizer_iter_=self%n_optimizer_iter_+iterations
      self%n_function_evaluations_=self%n_function_evaluations_+evaluations
      if(self%optimization_values_(run)<self%optimization_values_(best)) best=run
      if(best==run) then
        self%optimization_message_=message;self%converged_=message(:4)=='CONV'
      end if
    end do
    self%n_optimization_runs_=self%n_restarts_optimizer+1
    self%kernel_=initial%with_theta(self%optimization_points_(best,:))
  end subroutine
  subroutine fit_core(self,x,y,matrix_targets)
    class(gaussian_process_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    logical, intent(in) :: matrix_targets
    real(dp), allocatable :: variance(:)
    real(dp) :: value
    integer :: n,q,j
    call check_xy(x,size(y,1));call check_x(y);call validate_controls(self)
    n=size(y,1);q=size(y,2)
    if(self%n_targets>0) call require(q==self%n_targets,'GP target count differs from n_targets')
    variance=noise_variance(self,n);self%kernel_=configured_kernel(self)
    self%x_train_=x;self%y_train_=y;self%n_targets_=q;self%n_features_in_=size(x,2)
    if(allocated(self%y_train_means_)) deallocate(self%y_train_means_,self%y_train_stds_)
    allocate(self%y_train_means_(q),self%y_train_stds_(q));self%y_train_means_=0;self%y_train_stds_=1
    if(self%normalize_y) then
      do j=1,q
        self%y_train_means_(j)=sum(y(:,j))/n
        if(q==1) self%y_train_means_(j)=numpy_sum(y(:,j))/n
        self%y_train_stds_(j)=sqrt(sum((y(:,j)-self%y_train_means_(j))**2)/n)
        if(q==1) self%y_train_stds_(j)=sqrt(numpy_sum((y(:,j)-self%y_train_means_(j))**2)/n)
        if(matrix_targets) then
          if(self%y_train_stds_(j)<10*epsilon(1._dp)) self%y_train_stds_(j)=1
        else
          if(self%y_train_stds_(j)==0) self%y_train_stds_(j)=1
        end if
        self%y_train_(:,j)=(y(:,j)-self%y_train_means_(j))/self%y_train_stds_(j)
      end do
    end if
    self%y_train_mean_=self%y_train_means_(1);self%y_train_std_=self%y_train_stds_(1)
    call optimize_kernel(self,variance)
    value=likelihood_value(self%kernel_,x,self%y_train_,variance,l_out=self%l_,coef_out=self%alpha_multi_)
    call require(ieee_is_finite(value),'GP kernel matrix is not positive definite')
    self%log_marginal_likelihood_=value
    if(self%n_optimization_runs_>0) self%log_marginal_likelihood_=-minval(self%optimization_values_)
    if(allocated(self%alpha_)) deallocate(self%alpha_)
    if(q==1) self%alpha_=self%alpha_multi_(:,1)
  end subroutine
  subroutine gp_fit_vector(self,x,y)
    class(gaussian_process_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    call fit_core(self,x,reshape(y,[size(y),1]),.false.)
  end subroutine
  subroutine gp_fit_matrix(self,x,y)
    class(gaussian_process_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    call fit_core(self,x,y,.true.)
  end subroutine
  integer function output_count(self) result(q)
    class(gaussian_process_regressor), intent(in) :: self
    call require(self%n_targets>=0,'GP n_targets must be nonnegative (zero means automatic)')
    q=max(1,self%n_targets)
    if(allocated(self%alpha_multi_)) q=self%n_targets_
  end function
  function gp_predict_multi(self,x) result(y)
    class(gaussian_process_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:,:)
    integer :: q,j
    call check_x(x);q=output_count(self)
    if(.not.allocated(self%alpha_multi_)) then
      allocate(y(size(x,1),q));y=0;return
    end if
    if(q==1) then
      allocate(y(size(x,1),1));y(:,1)=matmul(covariance(self,x,self%x_train_),self%alpha_)
      y(:,1)=y(:,1)*self%y_train_std_+self%y_train_mean_;return
    end if
    y=matmul(covariance(self,x,self%x_train_),self%alpha_multi_)
    do j=1,q
      y(:,j)=y(:,j)*self%y_train_stds_(j)+self%y_train_means_(j)
    end do
  end function
  function gp_predict(self,x) result(y)
    class(gaussian_process_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: y(size(x,1))
    real(dp), allocatable :: all_y(:,:)
    call require(output_count(self)==1,'GP multiple targets require predict_multi')
    all_y=self%predict_multi(x);y=all_y(:,1)
  end function
  function base_covariance(self,x) result(cov)
    class(gaussian_process_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: cov(:,:),cross(:,:),v(:,:)
    integer :: j
    cov=covariance(self,x)
    if(.not.allocated(self%alpha_multi_)) return
    cross=covariance(self,self%x_train_,x);allocate(v(size(cross,1),size(cross,2)))
    do j=1,size(x,1)
      v(:,j)=solve_lower(self%l_,cross(:,j))
    end do
    cov=cov-matmul(transpose(v),v)
  end function
  function gp_predict_cov_multi(self,x) result(cov)
    class(gaussian_process_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: cov(:,:,:),base(:,:)
    integer :: q,j
    q=output_count(self);base=base_covariance(self,x);allocate(cov(size(x,1),size(x,1),q))
    do j=1,q
      cov(:,:,j)=base
      if(allocated(self%alpha_multi_)) then
        if(q==1) then
          cov(:,:,j)=base*self%y_train_std_**2
        else
          cov(:,:,j)=base*self%y_train_stds_(j)**2
        end if
      end if
    end do
  end function
  function gp_predict_cov(self,x) result(cov)
    class(gaussian_process_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: cov(size(x,1),size(x,1))
    real(dp), allocatable :: all_cov(:,:,:)
    call require(output_count(self)==1,'GP multiple targets require predict_cov_multi')
    all_cov=self%predict_cov_multi(x);cov=all_cov(:,:,1)
  end function
  function gp_predict_std_multi(self,x) result(std)
    class(gaussian_process_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: std(:,:),cross(:,:),v(:),diagonal(:)
    type(gp_kernel) :: kernel
    integer :: j,k,q
    q=output_count(self);allocate(std(size(x,1),q))
    if(allocated(self%alpha_multi_)) then
      diagonal=self%kernel_%diag(x);cross=covariance(self,self%x_train_,x)
      do j=1,size(x,1)
        v=solve_lower(self%l_,cross(:,j));diagonal(j)=max(0._dp,diagonal(j)-dot_product(v,v))
      end do
    else
      kernel=configured_kernel(self);diagonal=kernel%diag(x)
    end if
    do k=1,q
      std(:,k)=sqrt(diagonal)
      if(allocated(self%alpha_multi_)) then
        if(q==1) then
          std(:,k)=std(:,k)*self%y_train_std_
        else
          std(:,k)=std(:,k)*self%y_train_stds_(k)
        end if
      end if
    end do
  end function
  function gp_predict_std(self,x) result(std)
    class(gaussian_process_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: std(size(x,1))
    real(dp), allocatable :: all_std(:,:)
    call require(output_count(self)==1,'GP multiple targets require predict_std_multi')
    all_std=self%predict_std_multi(x);std=all_std(:,1)
  end function
  real(dp) function gp_likelihood(self,theta,gradient,clone_kernel) result(value)
    class(gaussian_process_regressor), intent(inout) :: self
    real(dp), optional, intent(in) :: theta(:)
    real(dp), allocatable, optional, intent(out) :: gradient(:)
    logical, optional, intent(in) :: clone_kernel
    type(gp_kernel) :: candidate
    real(dp), allocatable :: variance(:)
    logical :: clone
    call require(allocated(self%alpha_multi_),'Gaussian process is not fitted')
    if(.not.present(theta)) then
      call require(.not.present(gradient),'GP likelihood gradient requires theta')
      value=self%log_marginal_likelihood_;return
    end if
    candidate=self%kernel_%with_theta(theta);clone=.true.
    if(present(clone_kernel)) clone=clone_kernel
    if(.not.clone) self%kernel_=candidate
    variance=noise_variance(self,size(self%x_train_,1))
    value=likelihood_value(candidate,self%x_train_,self%y_train_,variance,gradient)
  end function
  real(dp) function score_values(y,p,sample_weight) result(value)
    real(dp), intent(in) :: y(:,:),p(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    real(dp), allocatable :: weights(:)
    real(dp) :: numerator,denominator,total,mean
    integer :: j,n
    call require(all(shape(y)==shape(p)),'GP score target shape mismatch');call check_x(y)
    n=size(y,1);allocate(weights(n));weights=1
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'GP score weight count mismatch')
      call require(all(ieee_is_finite(sample_weight)),'GP score weights must be finite')
      call require(all(sample_weight>=0),'GP score weights must be nonnegative');weights=sample_weight
    end if
    total=sum(weights);call require(total>0,'GP score weights must have positive sum')
    if(n<2) then
      value=ieee_value(0._dp,ieee_quiet_nan);return
    end if
    value=0
    do j=1,size(y,2)
      mean=dot_product(weights,y(:,j))/total;numerator=sum(weights*(y(:,j)-p(:,j))**2)
      denominator=sum(weights*(y(:,j)-mean)**2)
      if(denominator==0) then
        value=value+merge(1._dp,0._dp,numerator==0)
      else
        value=value+1-numerator/denominator
      end if
    end do
    value=value/size(y,2)
  end function
  real(dp) function gp_score_vector(self,x,y,sample_weight) result(value)
    class(gaussian_process_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    real(dp), allocatable :: prediction(:)
    prediction=self%predict(x)
    value=score_values(reshape(y,[size(y),1]),reshape(prediction,[size(prediction),1]),sample_weight)
  end function
  real(dp) function gp_score_matrix(self,x,y,sample_weight) result(value)
    class(gaussian_process_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    value=score_values(y,self%predict_multi(x),sample_weight)
  end function
end module
