! Laplace GPC: sklearn 1.6.1 gaussian_process/_gpc.py (GPML Algorithms 3.1/3.2).
! BSD-3-Clause attribution retained in LICENSES/scikit-learn-COPYING.
module sklearn_gp_classification
  use sklearn_common, only: dp,require,check_x,check_xy,unique_labels,sigmoid,cholesky_factor,solve_lower,solve_upper
  use sklearn_metrics, only: accuracy_score
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: gaussian_process_classifier
  real(dp), parameter :: lambdas(5)=[.41_dp,.4_dp,.37_dp,.44_dp,.39_dp]
  real(dp), parameter :: coefs(5)=[-1854.8214151_dp,3516.89893646_dp,221.29346712_dp,128.12323805_dp,-2010.49422654_dp]
  type :: gaussian_process_classifier
    real(dp) :: amplitude=1,length_scale=1,log_marginal_likelihood_=0
    integer :: max_iter_predict=100,n_features_in_=0,n_classes_=0
    integer, allocatable :: classes_(:),n_iter_(:)
    logical, allocatable :: converged_(:)
    real(dp), allocatable :: x_train_(:,:),posterior_mode_(:,:),pi_(:,:),w_sqrt_(:,:),l_(:,:,:),alpha_(:,:)
    real(dp), allocatable :: log_marginal_likelihoods_(:)
    real(dp), private :: fitted_amplitude=1,fitted_length=1
  contains
    procedure :: fit=>gpc_fit
    procedure :: predict_proba=>gpc_proba
    procedure :: predict=>gpc_predict
    procedure :: score=>gpc_score
    procedure :: log_marginal_likelihood=>gpc_likelihood
  end type
contains
  function rbf_covariance(x,y,amplitude,length) result(k)
    real(dp), intent(in) :: x(:,:),y(:,:),amplitude,length
    real(dp), allocatable :: k(:,:)
    integer :: i,j
    allocate(k(size(x,1),size(y,1)))
    do j=1,size(y,1)
      do i=1,size(x,1)
        k(i,j)=amplitude*exp(-.5_dp*sum(((x(i,:)-y(j,:))/length)**2))
      end do
    end do
  end function

  subroutine gpc_fit(self,x,y)
    class(gaussian_process_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), allocatable :: gram(:,:),target(:),f(:),prob(:),w(:),root(:),bmat(:,:),l(:,:),b(:),a(:),z(:)
    real(dp) :: previous,current
    integer :: n,c,channels,j,k,iteration
    call check_xy(x,size(y))
    call require(ieee_is_finite(self%amplitude),'nonfinite GPC amplitude')
    call require(self%amplitude>0,'GPC amplitude must be positive')
    call require(ieee_is_finite(self%length_scale),'nonfinite GPC length scale')
    call require(self%length_scale>0,'GPC length scale must be positive')
    call require(self%max_iter_predict>0,'GPC max_iter_predict must be positive')
    self%classes_=unique_labels(y); c=size(self%classes_)
    call require(c>=2,'GPC requires at least two classes')
    n=size(y); channels=c
    if(c==2) channels=1
    self%x_train_=x; self%n_features_in_=size(x,2); self%n_classes_=c
    self%fitted_amplitude=self%amplitude; self%fitted_length=self%length_scale
    gram=rbf_covariance(x,x,self%amplitude,self%length_scale)
    if(allocated(self%pi_)) then
      deallocate(self%pi_,self%posterior_mode_,self%w_sqrt_,self%l_,self%alpha_, &
        self%n_iter_,self%converged_,self%log_marginal_likelihoods_)
    end if
    allocate(self%pi_(n,channels),self%posterior_mode_(n,channels),self%w_sqrt_(n,channels))
    allocate(self%l_(n,n,channels),self%alpha_(n,channels),self%n_iter_(channels),self%converged_(channels))
    allocate(self%log_marginal_likelihoods_(channels),target(n),f(n),bmat(n,n))
    self%converged_=.false.
    do k=1,channels
      if(c==2) then
        target=merge(1._dp,0._dp,y==self%classes_(2))
      else
        target=merge(1._dp,0._dp,y==self%classes_(k))
      end if
      f=0; previous=0
      do iteration=1,self%max_iter_predict
        prob=sigmoid(f); w=prob*(1-prob); root=sqrt(w)
        bmat=gram*spread(root,2,n)*spread(root,1,n)
        do j=1,n
          bmat(j,j)=bmat(j,j)+1
        end do
        l=cholesky_factor(bmat); b=w*f+target-prob
        z=root*matmul(gram,b)
        a=b-root*solve_upper(transpose(l),solve_lower(l,z)); f=matmul(gram,a)
        z=-(2*target-1)*f
        current=-.5_dp*dot_product(a,f)-sum(max(0._dp,z)+log(1+exp(-abs(z))))
        do j=1,n
          current=current-log(l(j,j))
        end do
        if(iteration>1) then
          if(current-previous<1.e-10_dp) then
            self%converged_(k)=.true.; exit
          end if
        end if
        previous=current
      end do
      ! sklearn retains the previous likelihood and pre-update pi/W/L on convergence.
      self%log_marginal_likelihoods_(k)=previous
      self%posterior_mode_(:,k)=f; self%pi_(:,k)=prob; self%w_sqrt_(:,k)=root
      self%l_(:,:,k)=l; self%alpha_(:,k)=target-prob
      self%n_iter_(k)=min(iteration,self%max_iter_predict)
    end do
    self%log_marginal_likelihood_=sum(self%log_marginal_likelihoods_)/channels
  end subroutine

  function gpc_cross(self,x) result(k)
    class(gaussian_process_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: k(:,:)
    call require(allocated(self%pi_),'GPC is not fitted')
    call check_x(x)
    call require(size(x,2)==self%n_features_in_,'GPC feature count mismatch')
    k=rbf_covariance(self%x_train_,x,self%fitted_amplitude,self%fitted_length)
  end function

  function gpc_proba(self,x) result(probability)
    class(gaussian_process_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: probability(:,:),cross(:,:),mean(:),v(:),positive(:,:)
    real(dp) :: variance,value
    integer :: i,k
    cross=gpc_cross(self,x); allocate(positive(size(x,1),size(self%pi_,2)))
    do k=1,size(self%pi_,2)
      mean=matmul(transpose(cross),self%alpha_(:,k))
      do i=1,size(x,1)
        v=solve_lower(self%l_(:,:,k),self%w_sqrt_(:,k)*cross(:,i))
        variance=max(0._dp,self%fitted_amplitude-dot_product(v,v))
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
      cross=gpc_cross(self,x); mean=matmul(transpose(cross),self%alpha_(:,1))
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

  real(dp) function gpc_likelihood(self) result(value)
    class(gaussian_process_classifier), intent(in) :: self
    call require(allocated(self%pi_),'GPC is not fitted')
    value=self%log_marginal_likelihood_
  end function
end module
