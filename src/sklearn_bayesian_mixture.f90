! sklearn 1.6.1 variational Bayesian Gaussian mixtures; BSD provenance in THIRD_PARTY_NOTICES.md.
module sklearn_bayesian_mixture
  use sklearn_common, only: dp,require,check_x,logsumexp,cholesky_factor,solve_lower
  use sklearn_centroid_core, only: fit_centroids,centroid_seeds
  use sklearn_legacy_rng, only: legacy_random_state
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_value,ieee_negative_inf
  implicit none
  private
  public :: bayesian_gaussian_mixture
  type :: bayesian_gaussian_mixture
    integer :: n_components=1,max_iter=100,n_init=1,random_state=0,n_iter_=0,n_features_in_=0
    real(dp) :: tol=1.e-3_dp,reg_covar=1.e-6_dp,lower_bound_=0
    real(dp) :: weight_concentration_prior=0,mean_precision_prior=1,degrees_of_freedom_prior=0
    logical :: warm_start=.false.,converged_=.false.
    character(16) :: covariance_type='full'
    character(24) :: weight_concentration_prior_type='dirichlet_process'
    character(20) :: init_params='kmeans'
    real(dp), allocatable :: mean_prior(:),covariance_prior(:,:)
    real(dp) :: weight_concentration_prior_=0,mean_precision_prior_=0,degrees_of_freedom_prior_=0
    real(dp), allocatable :: mean_prior_(:),covariance_prior_(:,:),weights_(:),means_(:,:),mean_precision_(:)
    real(dp), allocatable :: weight_concentration_(:,:),degrees_of_freedom_(:),covariances_(:,:,:)
    real(dp), allocatable :: precisions_(:,:,:),precisions_cholesky_(:,:,:),log_det_precision_(:),lower_bounds_(:)
    character(16), private :: fitted_covariance=''
    character(24), private :: fitted_weight_type=''
  contains
    procedure :: fit
    procedure :: fit_predict
    procedure :: predict
    procedure :: predict_proba
    procedure :: score_samples
    procedure :: score
    procedure :: sample
  end type
contains
  real(dp) function psi(value) result(result)
    real(dp), intent(in) :: value
    real(dp) :: x,r,r2
    call require(value>=1/huge(1._dp),'Bayesian mixture digamma argument is too small')
    x=value; result=0
    do while(x<16)
      result=result-1/x; x=x+1
    end do
    r=1/x; r2=r*r
    result=result+log(x)-.5_dp*r-r2*(1._dp/12-r2*(1._dp/120-r2*(1._dp/252- &
      r2*(1._dp/240-r2*(1._dp/132-r2*691._dp/32760)))))
  end function
  subroutine parameters(self,x)
    class(bayesian_gaussian_mixture), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: centered(size(x,1),size(x,2)),mean(size(x,2)),factor(size(x,2),size(x,2)),v
    integer :: n,p,j,l
    call check_x(x); n=size(x,1); p=size(x,2)
    call require(n>=2.and.self%n_components>=1.and.self%n_components<=n,'invalid Bayesian mixture component/sample count')
    call require(self%max_iter>=0.and.self%n_init>=1,'invalid Bayesian mixture iteration controls')
    call require(self%random_state>=0,'Bayesian mixture seed must be nonnegative')
    call require(ieee_is_finite(self%tol).and.ieee_is_finite(self%reg_covar),'Bayesian mixture controls must be finite')
    call require(self%tol>=0.and.self%reg_covar>=0,'Bayesian mixture tolerance/regularization must be nonnegative')
    call require(any(self%covariance_type==[character(16)::'full','tied','diag','spherical']), &
      'unknown Bayesian mixture covariance type')
    call require(self%weight_concentration_prior_type=='dirichlet_process'.or. &
      self%weight_concentration_prior_type=='dirichlet_distribution','unknown Bayesian mixture weight prior type')
    call require(any(self%init_params==[character(20)::'kmeans','random','random_from_data','k-means++']), &
      'unknown Bayesian mixture initialization')
    call require(ieee_is_finite(self%weight_concentration_prior).and.ieee_is_finite(self%mean_precision_prior).and. &
      ieee_is_finite(self%degrees_of_freedom_prior),'Bayesian mixture scalar priors must be finite')
    call require(self%weight_concentration_prior>=0.and.self%mean_precision_prior>0, &
      'invalid Bayesian mixture weight/mean precision prior')
    self%weight_concentration_prior_=self%weight_concentration_prior
    if(self%weight_concentration_prior==0) self%weight_concentration_prior_=1._dp/self%n_components
    self%mean_precision_prior_=self%mean_precision_prior
    self%degrees_of_freedom_prior_=self%degrees_of_freedom_prior
    if(self%degrees_of_freedom_prior==0) self%degrees_of_freedom_prior_=real(p,dp)
    call require(self%degrees_of_freedom_prior_>p-1,'Bayesian mixture degrees-of-freedom prior must exceed p-1')
    mean=sum(x,dim=1)/n; self%mean_prior_=mean
    if(allocated(self%mean_prior)) then
      call require(size(self%mean_prior)==p,'Bayesian mixture mean prior shape mismatch')
      call require(all(ieee_is_finite(self%mean_prior)),'Bayesian mixture mean prior must be finite')
      self%mean_prior_=self%mean_prior
    end if
    self%covariance_prior_=spread(spread(0._dp,1,p),2,p)
    if(allocated(self%covariance_prior)) then
      call require(all(shape(self%covariance_prior)==[p,p]),'Bayesian mixture covariance prior shape mismatch')
      call require(all(ieee_is_finite(self%covariance_prior)),'Bayesian mixture covariance prior must be finite')
      self%covariance_prior_=self%covariance_prior
      call require(maxval(abs(self%covariance_prior-transpose(self%covariance_prior)))<= &
        1.e-10_dp*(1+maxval(abs(self%covariance_prior))),'Bayesian mixture covariance prior must be symmetric')
      if(self%covariance_type=='diag'.or.self%covariance_type=='spherical') then
        do j=1,p
          call require(self%covariance_prior(j,j)>0,'Bayesian mixture covariance prior diagonal must be positive')
          do l=1,p
            if(l/=j) call require(self%covariance_prior(j,l)==0,'Bayesian mixture variance prior must be diagonal')
          end do
        end do
        if(self%covariance_type=='spherical') call require(all(abs([(self%covariance_prior(j,j),j=1,p)]- &
          self%covariance_prior(1,1))<=1.e-12_dp*self%covariance_prior(1,1)), &
          'Bayesian mixture spherical prior must have equal diagonal entries')
      else
        factor=cholesky_factor(self%covariance_prior)
      end if
    else
      centered=x-spread(mean,1,n)
      if(self%covariance_type=='full'.or.self%covariance_type=='tied') then
        self%covariance_prior_=matmul(transpose(centered),centered)/(n-1)
      else
        do j=1,p
          self%covariance_prior_(j,j)=sum(centered(:,j)**2)/(n-1)
        end do
        if(self%covariance_type=='spherical') then
          v=sum([(self%covariance_prior_(j,j),j=1,p)])/p
          do j=1,p
            self%covariance_prior_(j,j)=v
          end do
        end if
      end if
    end if
    call require(all(ieee_is_finite(self%covariance_prior_)),'Bayesian mixture covariance prior overflow')
    self%n_features_in_=p
  end subroutine
  subroutine update(self,x,resp)
    class(bayesian_gaussian_mixture), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),resp(:,:)
    real(dp) :: nk(size(resp,2)),xsum(size(resp,2),size(x,2)),xk(size(resp,2),size(x,2))
    real(dp) :: sk(size(x,2),size(x,2),size(resp,2)),delta(size(x,2)),average,remaining
    integer :: n,p,k,c,i,j,l
    n=size(x,1); p=size(x,2); k=size(resp,2)
    nk=sum(resp,dim=1)+10*epsilon(1._dp); xsum=matmul(transpose(resp),x); xk=xsum/spread(nk,2,p)
    if(self%weight_concentration_prior_type=='dirichlet_process') then
      self%weight_concentration_=spread(spread(0._dp,1,k),2,2); remaining=0
      do c=k,1,-1
        self%weight_concentration_(c,1)=1+nk(c)
        self%weight_concentration_(c,2)=self%weight_concentration_prior_+remaining; remaining=remaining+nk(c)
      end do
    else
      self%weight_concentration_=reshape(self%weight_concentration_prior_+nk,[k,1])
    end if
    self%mean_precision_=self%mean_precision_prior_+nk
    self%means_=(spread(self%mean_prior_*self%mean_precision_prior_,1,k)+spread(nk,2,p)*xk)/ &
      spread(self%mean_precision_,2,p)
    sk=0
    select case(self%covariance_type)
    case('full')
      do c=1,k
        do i=1,n
          delta=x(i,:)-xk(c,:)
          do j=1,p
            sk(:,j,c)=sk(:,j,c)+resp(i,c)*delta*delta(j)
          end do
        end do
        sk(:,:,c)=sk(:,:,c)/nk(c)
      end do
    case('tied')
      sk(:,:,1)=(matmul(transpose(x),x)-matmul(transpose(xk)*spread(nk,1,p),xk))/sum(nk)
    case default
      do c=1,k
        do j=1,p
          sk(j,j,c)=sum(resp(:,c)*x(:,j)**2)/nk(c)-2*xk(c,j)*xsum(c,j)/nk(c)+xk(c,j)**2
        end do
      end do
    end select
    do c=1,k
      if(self%covariance_type=='tied'.and.c>1) cycle
      do j=1,p
        sk(j,j,c)=sk(j,j,c)+self%reg_covar
      end do
      if(self%covariance_type=='spherical') then
        average=sum([(sk(j,j,c),j=1,p)])/p; sk(:,:,c)=0
        do j=1,p
          sk(j,j,c)=average
        end do
      end if
    end do
    self%degrees_of_freedom_=self%degrees_of_freedom_prior_+nk
    self%covariances_=spread(spread(spread(0._dp,1,p),2,p),3,k)
    if(self%covariance_type=='tied') then
      self%degrees_of_freedom_=self%degrees_of_freedom_prior_+sum(nk)/k
      self%covariances_(:,:,1)=self%covariance_prior_+sk(:,:,1)*sum(nk)/k
      do c=1,k
        delta=xk(c,:)-self%mean_prior_
        do j=1,p
          self%covariances_(:,j,1)=self%covariances_(:,j,1)+self%mean_precision_prior_/k* &
            nk(c)/self%mean_precision_(c)*delta*delta(j)
        end do
      end do
      self%covariances_(:,:,1)=self%covariances_(:,:,1)/self%degrees_of_freedom_(1)
      do c=2,k
        self%covariances_(:,:,c)=self%covariances_(:,:,1)
      end do
    else
      do c=1,k
        delta=xk(c,:)-self%mean_prior_
        select case(self%covariance_type)
        case('full')
          self%covariances_(:,:,c)=self%covariance_prior_+nk(c)*sk(:,:,c)
          do j=1,p
            self%covariances_(:,j,c)=self%covariances_(:,j,c)+ &
              nk(c)*self%mean_precision_prior_/self%mean_precision_(c)*delta*delta(j)
          end do
        case('diag')
          do j=1,p
            self%covariances_(j,j,c)=self%covariance_prior_(j,j)+nk(c)*(sk(j,j,c)+ &
              self%mean_precision_prior_/self%mean_precision_(c)*delta(j)**2)
          end do
        case('spherical')
          average=self%covariance_prior_(1,1)+nk(c)*(sk(1,1,c)+ &
            self%mean_precision_prior_/self%mean_precision_(c)*sum(delta**2)/p)
          do j=1,p
            self%covariances_(j,j,c)=average
          end do
        end select
        self%covariances_(:,:,c)=self%covariances_(:,:,c)/self%degrees_of_freedom_(c)
      end do
    end if
    call finish_parameters(self)
  end subroutine
  subroutine finish_parameters(self)
    class(bayesian_gaussian_mixture), intent(inout) :: self
    real(dp) :: l(self%n_features_in_,self%n_features_in_),inverse(self%n_features_in_,self%n_features_in_)
    real(dp) :: unit(self%n_features_in_),stick,total
    integer :: p,k,c,j
    p=self%n_features_in_; k=self%n_components
    call require(all(ieee_is_finite(self%covariances_)),'Bayesian mixture covariance overflow')
    self%precisions_cholesky_=spread(spread(spread(0._dp,1,p),2,p),3,k)
    self%precisions_=self%precisions_cholesky_; self%log_det_precision_=spread(0._dp,1,k)
    do c=1,k
      if(self%covariance_type=='tied'.and.c>1) then
        self%precisions_cholesky_(:,:,c)=self%precisions_cholesky_(:,:,1)
        self%precisions_(:,:,c)=self%precisions_(:,:,1); self%log_det_precision_(c)=self%log_det_precision_(1)
        cycle
      end if
      if(self%covariance_type=='diag'.or.self%covariance_type=='spherical') then
        do j=1,p
          call require(self%covariances_(j,j,c)>0,'Bayesian mixture covariance must be positive definite')
          self%precisions_cholesky_(j,j,c)=1/sqrt(self%covariances_(j,j,c))
          self%precisions_(j,j,c)=self%precisions_cholesky_(j,j,c)**2
        end do
        self%log_det_precision_(c)=sum(log([(self%precisions_cholesky_(j,j,c),j=1,p)]))
      else
        l=cholesky_factor(self%covariances_(:,:,c))
        do j=1,p
          unit=0; unit(j)=1; inverse(:,j)=solve_lower(l,unit)
        end do
        self%precisions_cholesky_(:,:,c)=transpose(inverse)
        self%precisions_(:,:,c)=matmul(transpose(inverse),inverse)
        self%log_det_precision_(c)=-sum(log([(l(j,j),j=1,p)]))
      end if
    end do
    self%weights_=spread(0._dp,1,k)
    if(self%weight_concentration_prior_type=='dirichlet_process') then
      stick=1
      do c=1,k
        total=sum(self%weight_concentration_(c,:))
        self%weights_(c)=self%weight_concentration_(c,1)/total*stick
        stick=stick*self%weight_concentration_(c,2)/total
      end do
    else
      self%weights_=self%weight_concentration_(:,1)
    end if
    self%weights_=self%weights_/sum(self%weights_)
  end subroutine
  subroutine initialize(self,x,rng)
    class(bayesian_gaussian_mixture), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    type(legacy_random_state), intent(inout) :: rng
    real(dp) :: resp(size(x,1),self%n_components),value
    real(dp), allocatable :: centers(:,:)
    integer, allocatable :: labels(:),pool(:)
    integer :: i,c,n,k,j,t,it,chosen
    n=size(x,1); k=self%n_components; resp=0
    select case(self%init_params)
    case('kmeans')
      call fit_centroids(x,k,1,self%random_state,'k-means++',centers,labels,value,it,rng)
      do i=1,n
        resp(i,labels(i)+1)=1
      end do
    case('random')
      do i=1,n
        do c=1,k
          resp(i,c)=rng%uniform()
        end do
        resp(i,:)=resp(i,:)/sum(resp(i,:))
      end do
    case('random_from_data')
      pool=[(i,i=1,n)]
      do i=n,2,-1
        j=rng%integer_index(i); t=pool(i); pool(i)=pool(j); pool(j)=t
      end do
      do c=1,k
        resp(pool(c),c)=1
      end do
    case('k-means++')
      centers=centroid_seeds(x,k,'k-means++',rng)
      do c=1,k
        chosen=1
        do i=1,n
          if(all(x(i,:)==centers(c,:))) then
            chosen=i; exit
          end if
        end do
        resp(chosen,c)=1
      end do
    end select
    call update(self,x,resp)
  end subroutine
  subroutine query_check(self,x)
    class(bayesian_gaussian_mixture), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    call require(allocated(self%weights_),'Bayesian mixture is not fitted'); call check_x(x)
    call require(self%covariance_type==self%fitted_covariance.and. &
      self%weight_concentration_prior_type==self%fitted_weight_type.and.self%n_components==size(self%means_,1), &
      'Bayesian mixture controls changed after fitting')
    call require(size(x,2)==size(self%means_,2),'Bayesian mixture query feature count mismatch')
  end subroutine
  function joint_log_probability(self,x) result(values)
    class(bayesian_gaussian_mixture), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:,:)
    real(dp) :: log_weights(self%n_components),offset,total,correction,quadratic
    real(dp) :: transformed(size(x,2)),precision(size(x,2))
    integer :: p,k,c,i,j
    call query_check(self,x); p=size(x,2); k=self%n_components; offset=0
    if(self%weight_concentration_prior_type=='dirichlet_process') then
      do c=1,k
        total=psi(sum(self%weight_concentration_(c,:)))
        log_weights(c)=psi(self%weight_concentration_(c,1))-total+offset
        offset=offset+psi(self%weight_concentration_(c,2))-total
      end do
    else
      total=psi(sum(self%weight_concentration_(:,1)))
      do c=1,k
        log_weights(c)=psi(self%weight_concentration_(c,1))-total
      end do
    end if
    allocate(values(size(x,1),k))
    do c=1,k
      correction=0
      do j=0,p-1
        correction=correction+psi(.5_dp*(self%degrees_of_freedom_(c)-j))
      end do
      correction=.5_dp*(p*log(2._dp)+correction-p/self%mean_precision_(c)-p*log(self%degrees_of_freedom_(c)))
      precision=[(self%precisions_cholesky_(j,j,c)**2,j=1,p)]
      do i=1,size(x,1)
        if(self%covariance_type=='full'.or.self%covariance_type=='tied') then
          transformed=matmul(x(i,:),self%precisions_cholesky_(:,:,c))- &
            matmul(self%means_(c,:),self%precisions_cholesky_(:,:,c))
          quadratic=sum(transformed**2)
        else
          quadratic=sum(self%means_(c,:)**2*precision)-2*dot_product(x(i,:),self%means_(c,:)*precision)+ &
            dot_product(x(i,:)**2,precision)
        end if
        values(i,c)=log_weights(c)-.5_dp*(p*log(2*acos(-1._dp))+quadratic)+self%log_det_precision_(c)+correction
      end do
    end do
  end function
  real(dp) function bound(self,log_resp) result(value)
    class(bayesian_gaussian_mixture), intent(in) :: self
    real(dp), intent(in) :: log_resp(:,:)
    real(dp) :: wishart,weight_norm,ld,a,b,entropy
    integer :: p,c,i,j
    p=self%n_features_in_; entropy=0; wishart=0
    do c=1,self%n_components
      do i=1,size(log_resp,1)
        if(ieee_is_finite(log_resp(i,c))) entropy=entropy-exp(log_resp(i,c))*log_resp(i,c)
      end do
      ld=self%log_det_precision_(c)-.5_dp*p*log(self%degrees_of_freedom_(c))
      wishart=wishart-self%degrees_of_freedom_(c)*ld-self%degrees_of_freedom_(c)*p*.5_dp*log(2._dp)
      do j=0,p-1
        wishart=wishart-log_gamma(.5_dp*(self%degrees_of_freedom_(c)-j))
      end do
    end do
    weight_norm=0
    if(self%weight_concentration_prior_type=='dirichlet_process') then
      do c=1,self%n_components
        a=self%weight_concentration_(c,1); b=self%weight_concentration_(c,2)
        weight_norm=weight_norm-(log_gamma(a)+log_gamma(b)-log_gamma(a+b))
      end do
    else
      weight_norm=log_gamma(sum(self%weight_concentration_(:,1)))-sum(log_gamma(self%weight_concentration_(:,1)))
    end if
    value=entropy-wishart-weight_norm-.5_dp*p*sum(log(self%mean_precision_))
  end function
  subroutine fit(self,x)
    class(bayesian_gaussian_mixture), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    type(bayesian_gaussian_mixture) :: best_model
    type(legacy_random_state) :: rng
    real(dp), allocatable :: log_resp(:,:),history(:),larger(:)
    real(dp) :: previous,lower,best,starting,normalizer
    integer :: runs,run,i,count,capacity
    integer(int64) :: it
    logical :: do_init,stopped
    do_init=.true.; starting=self%lower_bound_
    if(self%warm_start.and.allocated(self%weights_)) then
      call require(size(x,2)==size(self%means_,2).and.self%n_components==size(self%means_,1).and. &
        self%covariance_type==self%fitted_covariance.and.self%weight_concentration_prior_type==self%fitted_weight_type, &
        'Bayesian mixture warm-start configuration mismatch')
      do_init=.false.
    end if
    call parameters(self,x)
    self%fitted_covariance=self%covariance_type; self%fitted_weight_type=self%weight_concentration_prior_type
    runs=self%n_init
    if(.not.do_init) runs=1
    call rng%seed(self%random_state); best=ieee_value(0._dp,ieee_negative_inf)
    do run=1,runs
      if(do_init) call initialize(self,x,rng)
      previous=starting
      if(do_init) previous=ieee_value(0._dp,ieee_negative_inf)
      if(allocated(history)) deallocate(history)
      allocate(history(min(32,self%max_iter))); count=0; stopped=.false.; lower=previous
      if(self%max_iter==0) lower=ieee_value(0._dp,ieee_negative_inf)
      do it=1,int(self%max_iter,int64)
        log_resp=joint_log_probability(self,x)
        do i=1,size(x,1)
          normalizer=logsumexp(log_resp(i,:)); log_resp(i,:)=log_resp(i,:)-normalizer
        end do
        call update(self,x,exp(log_resp)); lower=bound(self,log_resp)
        call require(ieee_is_finite(lower),'Bayesian mixture lower bound is not finite')
        if(count==size(history)) then
          capacity=int(min(2_int64*max(1,size(history)),int(self%max_iter,int64)))
          allocate(larger(capacity)); larger(:count)=history; call move_alloc(larger,history)
        end if
        count=count+1; history(count)=lower
        if(ieee_is_finite(previous)) then
          if(abs(lower-previous)<self%tol) then
            stopped=.true.; exit
          end if
        end if
        previous=lower
      end do
      if(run==1.or.lower>best.or.self%max_iter==0) then
        best=lower
        select type(self)
        type is(bayesian_gaussian_mixture)
          best_model=self
        end select
        best_model%converged_=stopped; best_model%n_iter_=int(min(it,int(self%max_iter,int64)))
        best_model%lower_bound_=lower; best_model%lower_bounds_=history(:count)
      end if
    end do
    select type(self)
    type is(bayesian_gaussian_mixture)
      self=best_model
    end select
  end subroutine
  function fit_predict(self,x) result(labels)
    class(bayesian_gaussian_mixture), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: labels(size(x,1))
    call self%fit(x); labels=self%predict(x)
  end function
  function predict(self,x) result(labels)
    class(bayesian_gaussian_mixture), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: labels(size(x,1))
    real(dp), allocatable :: values(:,:)
    values=joint_log_probability(self,x); labels=maxloc(values,dim=2)-1
  end function
  function predict_proba(self,x) result(values)
    class(bayesian_gaussian_mixture), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:,:)
    integer :: i
    values=joint_log_probability(self,x)
    do i=1,size(x,1)
      values(i,:)=exp(values(i,:)-logsumexp(values(i,:)))
    end do
  end function
  function score_samples(self,x) result(values)
    class(bayesian_gaussian_mixture), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: values(size(x,1))
    real(dp), allocatable :: joint(:,:)
    integer :: i
    joint=joint_log_probability(self,x)
    do i=1,size(x,1)
      values(i)=logsumexp(joint(i,:))
    end do
  end function
  real(dp) function score(self,x) result(value)
    class(bayesian_gaussian_mixture), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    value=sum(self%score_samples(x))/size(x,1)
  end function
  subroutine sample(self,n_samples,x,labels)
    class(bayesian_gaussian_mixture), intent(in) :: self
    integer, intent(in) :: n_samples
    real(dp), allocatable, intent(out) :: x(:,:)
    integer, allocatable, intent(out) :: labels(:)
    type(legacy_random_state) :: rng
    real(dp) :: l(self%n_features_in_,self%n_features_in_),z(self%n_features_in_),u,total
    integer :: counts(self%n_components),i,j,c,row
    call require(allocated(self%weights_),'Bayesian mixture is not fitted')
    call query_check(self,self%means_)
    call require(n_samples>0,'Bayesian mixture sample count must be positive')
    call require(self%random_state>=0,'Bayesian mixture seed must be nonnegative')
    counts=0; call rng%seed(self%random_state)
    do i=1,n_samples
      u=rng%uniform(); total=0
      do c=1,self%n_components
        total=total+self%weights_(c)
        if(u<total.or.c==self%n_components) then
          counts(c)=counts(c)+1; exit
        end if
      end do
    end do
    allocate(x(n_samples,self%n_features_in_),labels(n_samples)); row=0
    do c=1,self%n_components
      l=cholesky_factor(self%covariances_(:,:,c))
      do i=1,counts(c)
        do j=1,self%n_features_in_
          z(j)=rng%normal()
        end do
        row=row+1; x(row,:)=self%means_(c,:)+matmul(l,z); labels(row)=c-1
      end do
    end do
  end subroutine
end module
