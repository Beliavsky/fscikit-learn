! GaussianMixture EM and initialization: sklearn 1.6.1 mixture/_base.py and
! _gaussian_mixture.py (BSD). Typed storage and IEEE integration are MIT.
module sklearn_mixture
  use sklearn_bayesian_mixture, only: bayesian_gaussian_mixture
  use sklearn_common
  use sklearn_kmeans_core, only: k_means,kmeans_result
  use sklearn_kmeans_seeding, only: kmeans_plusplus_state
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_random_binomial, only: legacy_binomial
  use sklearn_kernel_dispatch, only: kernel_blas_dot
  use,intrinsic::ieee_arithmetic
  use,intrinsic::ieee_exceptions
  implicit none
  private
  public :: gaussian_mixture,bayesian_gaussian_mixture
  type :: gaussian_mixture
    integer :: n_components=1,max_iter=100,n_init=1,random_state=0,n_iter_=0,n_features_in_=0
    real(dp) :: tol=1.e-3_dp,reg_covar=1.e-6_dp,lower_bound_=0
    character(16) :: covariance_type='full'
    character(20) :: init_params='kmeans'
    logical :: converged_=.false.,warm_start=.false.
    real(dp), allocatable :: weights_init(:),means_init(:,:),precisions_init(:,:,:)
    real(dp), allocatable :: weights_(:),means_(:,:),covariances_(:,:,:),precisions_(:,:,:),logdet_(:)
    real(dp), allocatable :: precisions_cholesky_(:,:,:)
    character(16),private::fitted_covariance=''
  contains
    procedure :: fit=>mixture_fit
    procedure :: fit_predict=>mixture_fit_predict
    procedure :: score_samples=>mixture_samples
    procedure :: score=>mixture_score
    procedure :: predict_proba=>mixture_proba
    procedure :: predict=>mixture_predict
    procedure :: sample=>mixture_sample
    procedure :: bic=>mixture_bic
    procedure :: aic=>mixture_aic
  end type
contains
  recursive real(dp)function numpy_sum(a)result(value)
    real(dp),intent(in)::a(:)
    real(dp)::lane(8)
    integer::n,i,whole,half
    n=size(a);value=-0._dp
    if(n<8)then
      do i=1,n
        value=value+a(i)
      end do
    else if(n<=128)then
      lane=a(:8);whole=n-mod(n,8)
      do i=9,whole,8
        lane=lane+a(i:i+7)
      end do
      value=((lane(1)+lane(2))+(lane(3)+lane(4)))+((lane(5)+lane(6))+(lane(7)+lane(8)))
      do i=whole+1,n
        value=value+a(i)
      end do
    else
      half=n/2;half=half-mod(half,8);value=numpy_sum(a(:half))+numpy_sum(a(half+1:))
    end if
  end function
  function product(a,b)result(c)
    real(dp),intent(in)::a(:,:),b(:,:)
    real(dp)::c(size(a,1),size(b,2))
    integer::i,j
    call require(size(a,2)==size(b,1),'Gaussian mixture internal product shape mismatch')
    do j=1,size(b,2)
      do i=1,size(a,1)
        c(i,j)=kernel_blas_dot(a(i,:),b(:,j),size(a,1),size(b,2),i,j)
      end do
    end do
  end function
  subroutine controls(self,x)
    class(gaussian_mixture),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp)::values(size(x,2)),vectors(size(x,2),size(x,2))
    integer::p,k,c,j,l,q
    call check_x(x);p=size(x,2);k=self%n_components
    call require(size(x,1)>=2.and.k>=1.and.k<=size(x,1),'invalid mixture component count')
    call require(self%n_init>=1.and.self%max_iter>=0,'invalid mixture iteration controls')
    call require(ieee_is_finite(self%tol).and.ieee_is_finite(self%reg_covar),'mixture controls must be finite')
    call require(self%tol>=0.and.self%reg_covar>=0,'mixture tolerance/regularization must be nonnegative')
    call require(self%random_state>=0,'mixture seed must be nonnegative')
    call require(any(self%covariance_type==[character(16)::'full','tied','diag','spherical']),'unknown covariance type')
    call require(any(self%init_params==[character(20)::'kmeans','random','random_from_data','k-means++']), &
      'unknown mixture initialization')
    if(allocated(self%weights_init))then
      call require(size(self%weights_init)==k,'mixture initial weight shape mismatch')
      call require(all(ieee_is_finite(self%weights_init)),'mixture initial weights must be finite')
      call require(all(self%weights_init>=0).and.all(self%weights_init<=1),'mixture initial weights must lie in [0,1]')
      call require(abs(1-numpy_sum(self%weights_init))<=1.e-8_dp,'mixture initial weights must sum to one')
    end if
    if(allocated(self%means_init))then
      call require(all(shape(self%means_init)==[k,p]),'mixture initial mean shape mismatch')
      call check_x(self%means_init)
    end if
    if(allocated(self%precisions_init))then
      q=k;if(self%covariance_type=='tied')q=1
      call require(all(shape(self%precisions_init)==[p,p,q]),'mixture initial precision shape mismatch')
      call require(all(ieee_is_finite(self%precisions_init)),'mixture initial precisions must be finite')
      do c=1,q
        if(self%covariance_type=='full'.or.self%covariance_type=='tied')then
          call require(all(abs(self%precisions_init(:,:,c)-transpose(self%precisions_init(:,:,c))) &
            <=1.e-8_dp+1.e-5_dp*abs(transpose(self%precisions_init(:,:,c)))), &
            'mixture initial precision must be symmetric')
          call symmetric_eigen(self%precisions_init(:,:,c),values,vectors)
          call require(all(values>0),'mixture initial precision must be positive definite')
        else
          do j=1,p
            call require(self%precisions_init(j,j,c)>0,'mixture initial precisions must be positive')
            do l=1,p
              if(l/=j)call require(self%precisions_init(j,l,c)==0,'diagonal mixture precision requires diagonal storage')
            end do
          end do
          if(self%covariance_type=='spherical')call require(all([(self%precisions_init(j,j,c)== &
            self%precisions_init(1,1,c),j=1,p)]),'spherical mixture precision requires equal diagonal entries')
        end if
      end do
    end if
  end subroutine
  subroutine finish_parameters(self)
    class(gaussian_mixture),intent(inout)::self
    real(dp)::l(self%n_features_in_,self%n_features_in_),inverse(self%n_features_in_,self%n_features_in_)
    real(dp)::unit(self%n_features_in_)
    integer::p,k,c,j
    p=self%n_features_in_;k=self%n_components
    call require(all(ieee_is_finite(self%covariances_)),'mixture covariance exceeds finite range')
    self%precisions_cholesky_=spread(spread(spread(0._dp,1,p),2,p),3,k)
    do c=1,k
      if(self%covariance_type=='tied'.and.c>1)then
        self%precisions_cholesky_(:,:,c)=self%precisions_cholesky_(:,:,1);cycle
      end if
      if(self%covariance_type=='diag'.or.self%covariance_type=='spherical')then
        do j=1,p
          call require(self%covariances_(j,j,c)>0,'mixture covariance is singular; increase reg_covar')
          self%precisions_cholesky_(j,j,c)=1/sqrt(self%covariances_(j,j,c))
        end do
      else
        l=cholesky_factor(self%covariances_(:,:,c))
        do j=1,p
          unit=0;unit(j)=1;inverse(:,j)=solve_lower(l,unit)
        end do
        self%precisions_cholesky_(:,:,c)=transpose(inverse)
      end if
    end do
    call finish_precision(self)
  end subroutine
  subroutine finish_precision(self)
    class(gaussian_mixture),intent(inout)::self
    integer::p,k,c,j
    p=self%n_features_in_;k=self%n_components
    self%precisions_=self%precisions_cholesky_;self%logdet_=spread(0._dp,1,k)
    do c=1,k
      self%precisions_(:,:,c)=product(self%precisions_cholesky_(:,:,c),transpose(self%precisions_cholesky_(:,:,c)))
      self%logdet_(c)=-2*numpy_sum(log([(self%precisions_cholesky_(j,j,c),j=1,p)]))
    end do
  end subroutine
  subroutine update_parameters(self,x,resp,initial)
    class(gaussian_mixture),intent(inout)::self
    real(dp),intent(in)::x(:,:),resp(:,:)
    logical,optional,intent(in)::initial
    real(dp)::nk(size(resp,2)),delta(size(x,1),size(x,2)),average(size(x,2)),v
    real(dp),allocatable::cross(:,:),pooled(:,:)
    integer::n,p,k,c,i,j
    n=size(x,1);p=size(x,2);k=size(resp,2)
    ! Axis zero is strided in NumPy's C-order responsibility matrix.
    nk=0
    do i=1,n
      nk=nk+resp(i,:)
    end do
    nk=nk+10*epsilon(1._dp)
    if(k==1)nk(1)=numpy_sum(resp(:,1))+10*epsilon(1._dp)
    self%weights_=nk/numpy_sum(nk)
    if(present(initial))then
      if(initial)self%weights_=nk/n
    end if
    cross=product(transpose(resp),x);self%means_=cross/spread(nk,2,p)
    self%covariances_=spread(spread(spread(0._dp,1,p),2,p),3,k)
    select case(self%covariance_type)
    case('full')
      do c=1,k
        delta=x-spread(self%means_(c,:),1,n)
        self%covariances_(:,:,c)=product(transpose(delta)*spread(resp(:,c),1,p),delta)/nk(c)
      end do
    case('tied')
      pooled=(product(transpose(x),x)-product(transpose(self%means_)*spread(nk,1,p),self%means_))/numpy_sum(nk)
      self%covariances_=spread(pooled,3,k)
    case('diag','spherical')
      pooled=self%means_*cross/spread(nk,2,p)
      cross=product(transpose(resp),x*x)/spread(nk,2,p)-2*pooled+self%means_**2
      do c=1,k
        average=cross(c,:)+self%reg_covar
        if(self%covariance_type=='spherical')average=numpy_sum(average)/p
        do j=1,p
          self%covariances_(j,j,c)=average(j)
        end do
      end do
    end select
    if(self%covariance_type=='full'.or.self%covariance_type=='tied')then
      do c=1,k
        do j=1,p
          self%covariances_(j,j,c)=self%covariances_(j,j,c)+self%reg_covar
        end do
      end do
    end if
    if(present(initial))then
      if(initial.and.allocated(self%precisions_init))return
    end if
    call finish_parameters(self)
  end subroutine
  subroutine initialize(self,x,rng)
    class(gaussian_mixture),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    type(legacy_random_state),intent(inout)::rng
    type(kmeans_result)::km
    real(dp)::resp(size(x,1),self%n_components),l(size(x,2),size(x,2))
    real(dp),allocatable::centers(:,:),old_cov(:,:,:)
    integer,allocatable::ids(:),pool(:)
    integer::n,p,k,i,c,j,t,q
    n=size(x,1);p=size(x,2);k=self%n_components;resp=0
    if(allocated(self%precisions_init).and.allocated(self%covariances_))old_cov=self%covariances_
    if(.not.(allocated(self%weights_init).and.allocated(self%means_init).and.allocated(self%precisions_init)))then
      select case(self%init_params)
      case('kmeans')
        km=k_means(x,k,n_init=1,random_state_stream=rng)
        do i=1,n
          resp(i,km%labels(i)+1)=1
        end do
      case('random')
        do i=1,n
          do c=1,k
            resp(i,c)=rng%uniform()
          end do
          resp(i,:)=resp(i,:)/numpy_sum(resp(i,:))
        end do
      case('random_from_data')
        pool=[(i,i=1,n)]
        do i=n,2,-1
          j=rng%integer_index(i);t=pool(i);pool(i)=pool(j);pool(j)=t
        end do
        do c=1,k
          resp(pool(c),c)=1
        end do
      case('k-means++')
        call kmeans_plusplus_state(x,k,rng,centers,seed_indices=ids)
        do c=1,k
          resp(ids(c),c)=1
        end do
      end select
      ! Source estimates covariance even when explicit means override the means.
      call update_parameters(self,x,resp,initial=.true.)
    end if
    if(allocated(self%weights_init))self%weights_=self%weights_init
    if(allocated(self%means_init))self%means_=self%means_init
    if(allocated(self%precisions_init))then
      if(allocated(self%covariances_))deallocate(self%covariances_)
      if(allocated(old_cov))self%covariances_=old_cov
      self%precisions_cholesky_=spread(spread(spread(0._dp,1,p),2,p),3,k)
      do c=1,k
        q=c;if(self%covariance_type=='tied')q=1
        if(self%covariance_type=='full'.or.self%covariance_type=='tied')then
          l=cholesky_factor(self%precisions_init(p:1:-1,p:1:-1,q))
          self%precisions_cholesky_(:,:,c)=l(p:1:-1,p:1:-1)
        else
          do j=1,p
            self%precisions_cholesky_(j,j,c)=sqrt(self%precisions_init(j,j,q))
          end do
        end if
      end do
      call finish_precision(self)
    end if
  end subroutine
  subroutine query_check(self,x)
    class(gaussian_mixture),intent(in)::self
    real(dp),intent(in)::x(:,:)
    call require(allocated(self%weights_),'gaussian mixture is not fitted')
    call check_x(x)
    call require(size(x,2)==self%n_features_in_,'mixture query feature count mismatch')
    call require(self%covariance_type==self%fitted_covariance,'mixture covariance type changed after fitting')
  end subroutine
  function joint_log_probability(self,x)result(z)
    class(gaussian_mixture),intent(in)::self
    real(dp),intent(in)::x(:,:)
    real(dp),allocatable::z(:,:),work(:,:),mu(:,:),prec(:,:)
    real(dp)::quadratic(size(x,1)),logweight
    integer::i,c,j,p,n
    call query_check(self,x);n=size(x,1);p=size(x,2);allocate(z(n,size(self%weights_)),mu(1,p))
    do c=1,size(self%weights_)
      mu(1,:)=self%means_(c,:)
      if(self%covariance_type=='full'.or.self%covariance_type=='tied')then
        work=product(x,self%precisions_cholesky_(:,:,c))-spread( &
          reshape(product(mu,self%precisions_cholesky_(:,:,c)),[p]),1,n)
        do i=1,n
          quadratic(i)=numpy_sum(work(i,:)**2)
        end do
      else
        allocate(prec(1,p));prec(1,:)=[(self%precisions_cholesky_(j,j,c)**2,j=1,p)]
        if(self%covariance_type=='diag')then
          work=product(x,transpose(mu*prec));quadratic=numpy_sum(mu(1,:)**2*prec(1,:))-2*work(:,1)
          work=product(x*x,transpose(prec));quadratic=quadratic+work(:,1)
        else
          work=product(x,transpose(mu*prec));quadratic=numpy_sum(mu(1,:)**2)*prec(1,1)-2*work(:,1)
          do i=1,n
            quadratic(i)=quadratic(i)+numpy_sum(x(i,:)**2)*prec(1,1)
          end do
        end if
        deallocate(prec)
      end if
      logweight=ieee_value(0._dp,ieee_negative_inf)
      if(self%weights_(c)>0)logweight=log(self%weights_(c))
      z(:,c)=logweight-.5_dp*(p*log(2*acos(-1._dp))+quadratic+self%logdet_(c))
    end do
  end function
  subroutine mixture_fit(self,x)
    class(gaussian_mixture),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    type(gaussian_mixture)::best_model
    type(legacy_random_state)::rng
    real(dp),allocatable::resp(:,:),z(:,:)
    real(dp)::previous,lower,best,scores(size(x,1))
    logical::fresh,stopped,modes(size(ieee_all)),flags(size(ieee_all))
    integer::run,it,i,runs
    select type(self)
    type is(gaussian_mixture)
    class default
      call require(.false.,'GaussianMixture fit requires a direct estimator')
    end select
    call ieee_get_halting_mode(ieee_all,modes);call ieee_get_flag(ieee_all,flags)
    call ieee_set_halting_mode(ieee_all,.false.)
    call controls(self,x)
    fresh=.true.;if(self%warm_start)fresh=.not.allocated(self%weights_)
    if(.not.fresh)then
      call require(size(x,2)==self%n_features_in_,'warm mixture feature count mismatch')
      call require(self%n_components==size(self%weights_),'warm mixture component count mismatch')
      call require(self%covariance_type==self%fitted_covariance,'warm mixture covariance type mismatch')
    end if
    self%n_features_in_=size(x,2);self%fitted_covariance=self%covariance_type
    runs=1;if(fresh)runs=self%n_init
    call rng%seed(self%random_state);best=ieee_value(0._dp,ieee_negative_inf)
    allocate(resp(size(x,1),self%n_components));self%converged_=.false.
    do run=1,runs
      if(fresh)call initialize(self,x,rng)
      lower=self%lower_bound_;if(fresh)lower=ieee_value(0._dp,ieee_negative_inf)
      stopped=.false.;it=0
      if(self%max_iter==0)then
        call require(allocated(self%covariances_),'zero-iteration precision initialization has no covariance')
        select type(self)
        type is(gaussian_mixture)
          best_model=self
        end select
        best_model%n_iter_=0;best_model%converged_=.false.;best_model%lower_bound_=best
      else
        do it=1,self%max_iter
          previous=lower;z=joint_log_probability(self,x)
          do i=1,size(x,1)
            scores(i)=logsumexp(z(i,:));resp(i,:)=exp(z(i,:)-scores(i))
          end do
          lower=numpy_sum(scores)/size(x,1);call update_parameters(self,x,resp)
          if(abs(lower-previous)<self%tol)then
            stopped=.true.;exit
          end if
        end do
        if(lower>best.or.best==ieee_value(0._dp,ieee_negative_inf))then
          best=lower
          select type(self)
          type is(gaussian_mixture)
            best_model=self
          end select
          best_model%n_iter_=min(it,self%max_iter);best_model%lower_bound_=lower;best_model%converged_=stopped
        end if
      end if
    end do
    select type(self)
    type is(gaussian_mixture)
      self=best_model
    end select
    call finish_precision(self)
    call ieee_set_flag(ieee_all,flags);call ieee_set_halting_mode(ieee_all,modes)
  end subroutine
  function mixture_fit_predict(self,x)result(labels)
    class(gaussian_mixture),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    integer::labels(size(x,1))
    logical::modes(size(ieee_all)),flags(size(ieee_all))
    call ieee_get_halting_mode(ieee_all,modes);call ieee_get_flag(ieee_all,flags)
    call ieee_set_halting_mode(ieee_all,.false.)
    call self%fit(x);labels=self%predict(x)
    call ieee_set_flag(ieee_all,flags);call ieee_set_halting_mode(ieee_all,modes)
  end function
  subroutine mixture_sample(self,n_samples,x,labels)
    class(gaussian_mixture),intent(in)::self
    integer,optional,intent(in)::n_samples
    real(dp),allocatable,intent(out)::x(:,:)
    integer,allocatable,intent(out)::labels(:)
    type(legacy_random_state)::rng
    real(dp),allocatable::z(:,:),s(:),u(:,:),v(:,:),factor(:,:)
    real(dp)::remaining_p
    integer::n,p,k,c,i,j,row,remaining
    integer,allocatable::counts(:)
    logical::modes(size(ieee_all)),flags(size(ieee_all))
    call require(allocated(self%means_),'gaussian mixture is not fitted')
    call query_check(self,self%means_);n=1;if(present(n_samples))n=n_samples
    call require(n>=1,'mixture sample count must be positive')
    call require(self%random_state>=0,'mixture seed must be nonnegative')
    call ieee_get_halting_mode(ieee_all,modes);call ieee_get_flag(ieee_all,flags)
    call ieee_set_halting_mode(ieee_all,.false.)
    p=self%n_features_in_;k=size(self%weights_)
    call require(all(ieee_is_finite(self%weights_)),'mixture sampling weights must be finite')
    call require(all(self%weights_>=0).and.all(self%weights_<=1),'mixture sampling weights must lie in [0,1]')
    call require(numpy_sum(self%weights_(:k-1))<=1+1.e-12_dp,'mixture sampling probabilities exceed one')
    allocate(counts(k));counts=0
    call rng%seed(self%random_state);remaining=n;remaining_p=1
    do c=1,k-1
      counts(c)=legacy_binomial(remaining,self%weights_(c)/remaining_p,rng);remaining=remaining-counts(c)
      if(remaining<=0)exit
      remaining_p=remaining_p-self%weights_(c)
    end do
    if(remaining>0)counts(k)=remaining
    allocate(x(n,p),labels(n));row=0
    do c=1,k
      allocate(z(counts(c),p))
      do i=1,counts(c)
        do j=1,p
          z(i,j)=rng%normal()
        end do
      end do
      if(self%covariance_type=='full'.or.self%covariance_type=='tied')then
        allocate(s(p),u(p,p),v(p,p));call dense_svd(self%covariances_(:,:,c),s,u,v)
        factor=spread(sqrt(s),2,p)*transpose(v)
        x(row+1:row+counts(c),:)=product(z,factor)+spread(self%means_(c,:),1,counts(c))
        deallocate(s,u,v)
      else
        x(row+1:row+counts(c),:)=z*spread(sqrt([(self%covariances_(j,j,c),j=1,p)]),1,counts(c))+ &
          spread(self%means_(c,:),1,counts(c))
      end if
      labels(row+1:row+counts(c))=c-1;row=row+counts(c);deallocate(z)
    end do
    call ieee_set_flag(ieee_all,flags);call ieee_set_halting_mode(ieee_all,modes)
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
