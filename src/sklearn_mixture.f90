module sklearn_mixture
  use sklearn_bayesian_mixture, only: bayesian_gaussian_mixture
  use sklearn_common
  use sklearn_cluster, only: kmeans
  implicit none
  private
  public :: gaussian_mixture,bayesian_gaussian_mixture
  type :: gaussian_mixture
    integer :: n_components=1,max_iter=100,n_init=1,random_state=0,n_iter_=0
    real(dp) :: tol=1.e-3_dp,reg_covar=1.e-6_dp,lower_bound_=0
    character(16) :: covariance_type='full'
    logical :: converged_=.false.
    real(dp), allocatable :: weights_(:),means_(:,:),covariances_(:,:,:),precisions_(:,:,:),logdet_(:)
  contains
    procedure :: fit=>mixture_fit
    procedure :: score_samples=>mixture_samples
    procedure :: score=>mixture_score
    procedure :: predict_proba=>mixture_proba
    procedure :: predict=>mixture_predict
    procedure :: bic=>mixture_bic
    procedure :: aic=>mixture_aic
  end type
contains
  subroutine update_parameters(self,x,responsibilities)
    class(gaussian_mixture), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),responsibilities(:,:)
    real(dp) :: nk(size(responsibilities,2)),delta(size(x,2)),trace
    real(dp) :: values(size(x,2)),vectors(size(x,2),size(x,2)),pooled(size(x,2),size(x,2))
    integer :: n,p,k,c,i,j
    n=size(x,1); p=size(x,2); k=size(responsibilities,2)
    nk=sum(responsibilities,dim=1)+10*epsilon(1._dp)
    self%weights_=nk/sum(nk); self%means_=matmul(transpose(responsibilities),x)/spread(nk,2,p)
    if(.not.allocated(self%covariances_)) allocate(self%covariances_(p,p,k),self%precisions_(p,p,k),self%logdet_(k))
    self%covariances_=0
    do c=1,k
      do i=1,n
        delta=x(i,:)-self%means_(c,:)
        do j=1,p
          self%covariances_(:,j,c)=self%covariances_(:,j,c)+responsibilities(i,c)*delta*delta(j)
        end do
      end do
      self%covariances_(:,:,c)=self%covariances_(:,:,c)/nk(c)
    end do
    if(self%covariance_type=='tied') then
      pooled=0
      do c=1,k
        pooled=pooled+self%covariances_(:,:,c)*nk(c)/sum(nk)
      end do
      do c=1,k
        self%covariances_(:,:,c)=pooled
      end do
    end if
    do c=1,k
      if(self%covariance_type=='diag'.or.self%covariance_type=='spherical') then
        trace=0
        do j=1,p
          trace=trace+self%covariances_(j,j,c)/p
        end do
        do j=1,p
          delta(j)=self%covariances_(j,j,c)
        end do
        if(self%covariance_type=='spherical') delta=trace
        self%covariances_(:,:,c)=0
        do j=1,p
          self%covariances_(j,j,c)=delta(j)
        end do
      end if
      do j=1,p
        self%covariances_(j,j,c)=self%covariances_(j,j,c)+self%reg_covar
      end do
      call symmetric_eigen(self%covariances_(:,:,c),values,vectors)
      call require(all(values>0),'mixture covariance is singular; increase reg_covar')
      self%logdet_(c)=sum(log(values))
      self%precisions_(:,:,c)=matmul(vectors/spread(values,1,p),transpose(vectors))
    end do
  end subroutine
  function joint_log_probability(self,x) result(z)
    class(gaussian_mixture), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    real(dp) :: delta(size(x,2))
    integer :: i,c
    call require(allocated(self%weights_),'gaussian mixture is not fitted')
    call check_xy(transpose(x),size(self%means_,2)); allocate(z(size(x,1),size(self%weights_)))
    do i=1,size(x,1)
      do c=1,size(self%weights_)
        delta=x(i,:)-self%means_(c,:)
        z(i,c)=log(self%weights_(c))-0.5_dp*(size(x,2)*log(2*acos(-1._dp))+self%logdet_(c)+ &
          dot_product(delta,matmul(self%precisions_(:,:,c),delta)))
      end do
    end do
  end function
  subroutine mixture_fit(self,x)
    class(gaussian_mixture), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    type(gaussian_mixture) :: best_model
    type(kmeans) :: km
    real(dp), allocatable :: resp(:,:),z(:,:)
    real(dp) :: previous,lower,best
    integer :: run,it,i,k,n
    call check_x(x); n=size(x,1); k=self%n_components
    call require(k>0.and.k<=n.and.n>1,'invalid mixture component count')
    call require(self%n_init>0.and.self%max_iter>0.and.self%tol>0.and.self%reg_covar>0,'invalid mixture controls')
    call require(any(self%covariance_type==['full     ','tied     ','diag     ','spherical']),'unknown covariance type')
    if(allocated(self%covariances_)) deallocate(self%covariances_,self%precisions_,self%logdet_)
    allocate(resp(n,k)); best=-huge(1._dp)
    do run=1,self%n_init
      km%n_clusters=k; km%n_init=1; km%random_state=self%random_state+run-1; call km%fit(x)
      resp=0
      do i=1,n
        resp(i,km%labels_(i)+1)=1
      end do
      call update_parameters(self,x,resp); previous=-huge(1._dp); self%converged_=.false.
      do it=1,self%max_iter
        z=joint_log_probability(self,x); lower=0
        do i=1,n
          lower=lower+logsumexp(z(i,:)); resp(i,:)=exp(z(i,:)-logsumexp(z(i,:)))
        end do
        lower=lower/n; call update_parameters(self,x,resp)
        if(abs(lower-previous)<self%tol) then
          self%converged_=.true.; exit
        end if
        previous=lower
      end do
      self%n_iter_=min(it,self%max_iter); self%lower_bound_=lower
      if(lower>best) then
        best=lower
        select type(self)
        type is(gaussian_mixture)
          best_model=self
        end select
      end if
    end do
    select type(self)
    type is(gaussian_mixture)
      self=best_model
    end select
  end subroutine
  function mixture_samples(self,x) result(scores)
    class(gaussian_mixture), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: scores(size(x,1))
    real(dp), allocatable :: z(:,:)
    integer :: i
    z=joint_log_probability(self,x)
    do i=1,size(x,1)
      scores(i)=logsumexp(z(i,:))
    end do
  end function
  real(dp) function mixture_score(self,x) result(v)
    class(gaussian_mixture), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    v=sum(self%score_samples(x))/size(x,1)
  end function
  function mixture_proba(self,x) result(prob)
    class(gaussian_mixture), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: prob(:,:)
    integer :: i
    prob=joint_log_probability(self,x)
    do i=1,size(x,1)
      prob(i,:)=exp(prob(i,:)-logsumexp(prob(i,:)))
    end do
  end function
  function mixture_predict(self,x) result(labels)
    class(gaussian_mixture), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: labels(size(x,1))
    real(dp), allocatable :: z(:,:)
    z=joint_log_probability(self,x); labels=maxloc(z,dim=2)-1
  end function
  integer function parameter_count(self) result(n)
    class(gaussian_mixture), intent(in) :: self
    integer :: k,p
    call require(allocated(self%means_),'mixture is not fitted'); k=size(self%means_,1); p=size(self%means_,2)
    select case(trim(self%covariance_type))
    case('full'); n=k*p*(p+1)/2
    case('tied'); n=p*(p+1)/2
    case('diag'); n=k*p
    case('spherical'); n=k
    end select
    n=n+k*p+k-1
  end function
  real(dp) function mixture_bic(self,x) result(v)
    class(gaussian_mixture), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    v=-2*sum(self%score_samples(x))+parameter_count(self)*log(real(size(x,1),dp))
  end function
  real(dp) function mixture_aic(self,x) result(v)
    class(gaussian_mixture), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    v=-2*sum(self%score_samples(x))+2*parameter_count(self)
  end function
end module
