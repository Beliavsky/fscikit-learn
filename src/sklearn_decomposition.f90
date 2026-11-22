module sklearn_decomposition
  use sklearn_minibatch_nmf, only: mini_batch_nmf
  use sklearn_lda, only: latent_dirichlet_allocation
  use sklearn_dictionary, only: sparse_pca,dictionary_learning,sparse_coder,sparse_encode, &
    mini_batch_dictionary_learning,mini_batch_sparse_pca
  use sklearn_incremental_pca, only: incremental_pca
  use sklearn_fast_ica, only: fast_ica
  use sklearn_common
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_randomized_svd_core, only: randomized_svd
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_value,ieee_negative_inf
  use sklearn_kernel_pca, only: kernel_pca
  use sklearn_factor_analysis, only: factor_analysis
  implicit none
  private
  public :: mini_batch_nmf
  public :: mini_batch_dictionary_learning,mini_batch_sparse_pca
  public :: latent_dirichlet_allocation
  public :: sparse_pca,dictionary_learning,sparse_coder,sparse_encode
  public :: incremental_pca
  public :: fast_ica,pca, truncated_svd, nmf, kernel_pca, factor_analysis
  type :: pca
    integer :: n_components=0,n_samples_=0,n_components_=0
    character(16) :: dimension_mode='count',svd_solver='full',fit_svd_solver_='full'
    character(8) :: power_iteration_normalizer='auto'
    integer :: iterated_power=-1,n_oversamples=10,random_state=0
    real(dp) :: variance_fraction=.95_dp
    real(dp), allocatable :: mle_log_likelihood_(:)
    logical :: whiten=.false.
    real(dp), allocatable :: components_(:,:),mean_(:),explained_variance_(:)
    real(dp), allocatable :: explained_variance_ratio_(:),singular_values_(:)
    real(dp) :: noise_variance_=0
  contains
    procedure :: fit=>pca_fit
    procedure :: transform=>pca_transform
    procedure :: inverse_transform=>pca_inverse
    procedure :: fit_transform=>pca_fit_transform
    procedure :: get_covariance=>pca_covariance
    procedure :: get_precision=>pca_precision
    procedure :: score_samples=>pca_score_samples
    procedure :: score=>pca_score
  end type
  type :: truncated_svd
    integer :: n_components=2
    real(dp), allocatable :: components_(:,:),singular_values_(:),explained_variance_(:)
    real(dp), allocatable :: explained_variance_ratio_(:)
  contains
    procedure :: fit=>svd_fit
    procedure :: transform=>svd_transform
    procedure :: inverse_transform=>svd_inverse
    procedure :: fit_transform=>svd_fit_transform
  end type
  type :: nmf
    integer :: n_components=2,max_iter=1000,n_iter_=0,random_state=0
    real(dp) :: tol=1.e-6_dp,reconstruction_err_=0
    logical :: converged_=.false.
    real(dp), allocatable :: components_(:,:)
  contains
    procedure :: fit=>nmf_fit
    procedure :: fit_transform=>nmf_fit_transform
    procedure :: transform=>nmf_transform
    procedure :: inverse_transform=>nmf_inverse
  end type
contains
  subroutine oriented_eigen(gram,values,vectors)
    real(dp), intent(in) :: gram(:,:)
    real(dp), intent(out) :: values(:),vectors(:,:)
    integer :: j,k
    call symmetric_eigen(gram,values,vectors); values=max(values,0._dp)
    do j=1,size(values)
      k=maxloc(abs(vectors(:,j)),dim=1)
      if(vectors(k,j)<0) vectors(:,j)=-vectors(:,j)
    end do
  end subroutine
  subroutine pca_fit(self,x)
    class(pca), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    call pca_fit_core(self,x)
  end subroutine
  subroutine pca_fit_core(self,x,fit_transformed)
    class(pca), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable, optional, intent(out) :: fit_transformed(:,:)
    real(dp), allocatable :: centered(:,:),u(:,:),values(:),vectors(:,:),singular(:),gram(:,:),rs(:),vt(:,:)
    type(legacy_random_state) :: rng
    real(dp) :: total,cumulative
    integer :: k,n,p,j,r,spectrum_size,index
    call check_x(x);n=size(x,1);p=size(x,2);self%n_samples_=n;r=min(n,p)
    call require(n>1,'PCA requires at least two samples')
    call require(any(self%dimension_mode==[character(16)::'count','variance','mle','zero']),'unsupported PCA dimension_mode')
    call require(any(self%svd_solver==[character(16)::'full','randomized','covariance_eigh','auto']),'unsupported PCA svd_solver')
    call require(self%iterated_power>=-1,'PCA iterated_power must be >= -1 (automatic)')
    call require(self%n_oversamples>0,'PCA n_oversamples must be positive')
    call require(self%random_state>=0,'PCA random_state must be nonnegative')
    call require(any(self%power_iteration_normalizer==[character(8)::'auto','none','LU','QR']), &
      'unsupported PCA power_iteration_normalizer')
    k=self%n_components;if(k==0) k=r
    if(self%dimension_mode=='count') then
      call require(k>0.and.k<=r,'invalid PCA n_components')
    else if(self%dimension_mode=='variance') then
      call require(ieee_is_finite(self%variance_fraction),'PCA variance_fraction must be finite')
      call require(self%variance_fraction>0.and.self%variance_fraction<1,'PCA variance_fraction must be in (0,1)')
    else if(self%dimension_mode=='mle') then
      call require(n>=p,'PCA MLE requires samples >= features')
    else
      k=0
    end if
    self%fit_svd_solver_=self%svd_solver
    if(self%svd_solver=='auto') then
      if(p<=1000.and.int(n,int64)>=10_int64*p) then
        self%fit_svd_solver_='covariance_eigh'
      else if(max(n,p)<=500.or.self%dimension_mode=='mle') then
        self%fit_svd_solver_='full'
      else if(self%dimension_mode=='count'.and.k>=1.and.k<.8_dp*r) then
        self%fit_svd_solver_='randomized'
      else
        self%fit_svd_solver_='full'
      end if
    end if
    if(self%fit_svd_solver_=='randomized') call require(self%dimension_mode=='count', &
      'PCA randomized solver requires a positive integer component count')
    if(allocated(self%mle_log_likelihood_)) deallocate(self%mle_log_likelihood_)
    self%mean_=sum(x,dim=1)/n;allocate(values(p),singular(p),vectors(p,p));values=0;singular=0;vectors=0
    spectrum_size=r
    select case(self%fit_svd_solver_)
    case('full')
      centered=x-spread(self%mean_,1,n);allocate(u(n,p))
      call dense_svd(centered,singular,u,vectors);values=singular**2/(n-1);total=sum(values)
    case('covariance_eigh')
      gram=(matmul(transpose(x),x)-n*spread(self%mean_,2,p)*spread(self%mean_,1,p))/(n-1)
      call oriented_eigen(gram,values,vectors);singular=sqrt(values*real(n-1,dp));total=sum(values);spectrum_size=p
    case('randomized')
      call require(k<=huge(k)-self%n_oversamples,'PCA randomized sample dimension exceeds integer capacity')
      centered=x-spread(self%mean_,1,n);call rng%seed(self%random_state)
      if(self%iterated_power==-1) then
        call randomized_svd(centered,k,rng,rs,u,vt,n_oversamples=self%n_oversamples, &
          power_iteration_normalizer=self%power_iteration_normalizer,flip_sign=.false.)
      else
        call randomized_svd(centered,k,rng,rs,u,vt,n_iter=self%iterated_power,n_oversamples=self%n_oversamples, &
          power_iteration_normalizer=self%power_iteration_normalizer,flip_sign=.false.)
      end if
      ! The thin native SVD omits zero-singular left vectors. Complete only
      ! this PCA caller's null bases, leaving previous core consumers unchanged.
      call pca_complete_columns(u);gram=transpose(vt);call pca_complete_columns(gram);vt=transpose(gram)
      do j=1,k
        index=maxloc(abs(vt(j,:)),dim=1)
        if(vt(j,index)>=0) cycle
        vt(j,:)=-vt(j,:);u(:,j)=-u(:,j)
      end do
      vectors(:,:k)=transpose(vt);singular(:k)=rs;values(:k)=rs**2/(n-1);total=sum(centered**2)/(n-1)
    end select
    select case(self%dimension_mode)
    case('variance')
      k=1
      if(total>0) then
        cumulative=0;k=r
        do j=1,r
          cumulative=cumulative+values(j)/total
          if(cumulative>self%variance_fraction) then
            k=j;exit
          end if
        end do
      end if
    case('mle')
      allocate(self%mle_log_likelihood_(p));self%mle_log_likelihood_=ieee_value(0._dp,ieee_negative_inf)
      do j=1,p-1
        self%mle_log_likelihood_(j+1)=pca_rank_likelihood(values,j,n)
      end do
      k=maxloc(self%mle_log_likelihood_,dim=1)-1
    case('zero');k=0
    end select
    self%n_components_=k;self%singular_values_=singular(:k)
    self%components_=transpose(vectors(:,:k));self%explained_variance_=values(:k)
    self%explained_variance_ratio_=values(:k)*0
    if(total>0) self%explained_variance_ratio_=values(:k)/total
    self%noise_variance_=0
    if(k<r) then
      if(self%fit_svd_solver_=='randomized') then
        self%noise_variance_=(total-sum(values(:k)))/(r-k)
      else
        self%noise_variance_=sum(values(k+1:spectrum_size))/(spectrum_size-k)
      end if
    end if
    if(present(fit_transformed)) then
      if(self%fit_svd_solver_=='randomized') then
        if(self%whiten) then
          fit_transformed=u*sqrt(real(n-1,dp))
        else
          fit_transformed=u*spread(singular(:k),1,n)
        end if
      else
        fit_transformed=self%transform(x)
      end if
    end if
  end subroutine
  subroutine pca_complete_columns(a)
    real(dp), intent(inout) :: a(:,:)
    real(dp), allocatable :: candidate(:),best(:)
    real(dp) :: norm,best_norm
    integer :: i,j,pass
    do j=1,size(a,2)
      if(sum(a(:,j)**2)>0) cycle
      best_norm=-1
      do i=1,size(a,1)
        allocate(candidate(size(a,1)));candidate=0;candidate(i)=1
        do pass=1,2
          candidate=candidate-matmul(a(:,:j-1),matmul(transpose(a(:,:j-1)),candidate))
        end do
        norm=sum(candidate*candidate)
        if(norm>best_norm) then
          best=candidate;best_norm=norm
        end if
        deallocate(candidate)
      end do
      call require(best_norm>epsilon(1._dp),'PCA null basis completion failed');a(:,j)=best/sqrt(best_norm)
    end do
  end subroutine
  real(dp) function pca_rank_likelihood(spectrum,rank,n) result(value)
    ! sklearn 1.6.1 _pca.py _assess_dimension (Minka NIPS 2000).
    real(dp), intent(in) :: spectrum(:)
    integer, intent(in) :: rank,n
    real(dp), parameter :: small=1.e-15_dp
    real(dp) :: pu,pl,pv,pp,pa,variance,m,product,adjusted(size(spectrum)),dimension
    integer :: p,i,j
    p=size(spectrum);value=ieee_value(0._dp,ieee_negative_inf)
    if(spectrum(rank)<small) return
    pu=-rank*log(2._dp)
    do i=1,rank
      dimension=real(p-i+1,dp)/2
      pu=pu+log_gamma(dimension)-log(acos(-1._dp))*dimension
    end do
    pl=-sum(log(spectrum(:rank)))*n/2
    variance=max(small,sum(spectrum(rank+1:))/real(p-rank,dp))
    pv=-log(variance)*n*(p-rank)/2
    m=real(p,dp)*rank-rank*real(rank+1,dp)/2
    pp=log(2*acos(-1._dp))*(m+rank)/2
    pa=0;adjusted=spectrum;adjusted(rank+1:)=variance
    do i=1,rank
      do j=i+1,p
        product=(spectrum(i)-spectrum(j))*(1/adjusted(j)-1/adjusted(i))
        call require(product>0,'PCA MLE spectrum is numerically invalid')
        pa=pa+log(product)+log(real(n,dp))
      end do
    end do
    value=pu+pl+pv+pp-pa/2-rank*log(real(n,dp))/2
  end function
  function pca_covariance(self) result(cov)
    class(pca), intent(in) :: self
    real(dp), allocatable :: cov(:,:),components(:,:),difference(:)
    integer :: j
    call require(allocated(self%components_),'PCA is not fitted')
    components=self%components_
    if(self%whiten) components=components*spread(sqrt(self%explained_variance_),2,size(components,2))
    difference=max(self%explained_variance_-self%noise_variance_,0._dp)
    cov=matmul(transpose(components),components*spread(difference,2,size(components,2)))
    do j=1,size(cov,1)
      cov(j,j)=cov(j,j)+self%noise_variance_
    end do
  end function
  function pca_probability_factor(self) result(l)
    class(pca), intent(in) :: self
    real(dp), allocatable :: l(:,:),cov(:,:)
    call require(allocated(self%components_),'PCA is not fitted')
    call require(self%noise_variance_>0.or. &
      (size(self%components_,1)==size(self%components_,2).and.all(self%explained_variance_>0)), &
      'PCA covariance must be positive definite for precision and scoring')
    cov=self%get_covariance();l=cholesky_factor(cov)
  end function
  function pca_precision(self) result(precision)
    class(pca), intent(in) :: self
    real(dp), allocatable :: precision(:,:),l(:,:),rhs(:)
    integer :: p,j
    l=pca_probability_factor(self);p=size(l,1);allocate(precision(p,p),rhs(p))
    do j=1,p
      rhs=0;rhs(j)=1;precision(:,j)=solve_upper(transpose(l),solve_lower(l,rhs))
    end do
  end function
  function pca_score_samples(self,x) result(scores)
    class(pca), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: scores(:),l(:,:),centered(:,:)
    real(dp) :: normalizer
    integer :: i,j
    call require(allocated(self%components_),'PCA is not fitted')
    call check_xy(transpose(x),size(self%mean_));l=pca_probability_factor(self)
    normalizer=size(x,2)*log(2*acos(-1._dp))
    do j=1,size(l,1)
      normalizer=normalizer+2*log(l(j,j))
    end do
    centered=x-spread(self%mean_,1,size(x,1));allocate(scores(size(x,1)))
    do i=1,size(x,1)
      scores(i)=-.5_dp*(sum(solve_lower(l,centered(i,:))**2)+normalizer)
    end do
  end function
  real(dp) function pca_score(self,x) result(score)
    class(pca), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    score=sum(self%score_samples(x))/size(x,1)
  end function
  function pca_transform(self,x) result(z)
    class(pca), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call require(allocated(self%components_),'PCA is not fitted')
    call check_xy(transpose(x),size(self%mean_))
    z=matmul(x-spread(self%mean_,1,size(x,1)),transpose(self%components_))
    if(self%whiten) z=z/spread(max(sqrt(self%explained_variance_),epsilon(1._dp)),1,size(x,1))
  end function
  function pca_inverse(self,z) result(x)
    class(pca), intent(in) :: self
    real(dp), intent(in) :: z(:,:)
    real(dp), allocatable :: x(:,:),work(:,:)
    call require(allocated(self%components_),'PCA is not fitted')
    if(size(self%components_,1)==0) then
      call require(size(z,1)>0,'X must be nonempty')
      call require(size(z,2)==0,'X and y sample counts differ')
    else
      call check_xy(transpose(z),size(self%components_,1))
    end if
    work=z
    if(self%whiten) work=work*spread(sqrt(self%explained_variance_),1,size(z,1))
    x=matmul(work,self%components_)+spread(self%mean_,1,size(z,1))
  end function
  function pca_fit_transform(self,x) result(z)
    class(pca), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call pca_fit_core(self,x,z)
  end function
  subroutine svd_fit(self,x)
    class(truncated_svd), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: u(size(x,1),size(x,2)),values(size(x,2)),vectors(size(x,2),size(x,2)),total
    real(dp), allocatable :: z(:,:)
    integer :: k,j
    call check_x(x); k=self%n_components
    call require(k>0.and.k<=min(size(x,1),size(x,2)),'invalid SVD n_components')
    call dense_svd(x,values,u,vectors)
    self%components_=transpose(vectors(:,:k)); self%singular_values_=values(:k)
    z=matmul(x,transpose(self%components_)); self%explained_variance_=values(:k)*0
    do j=1,k
      self%explained_variance_(j)=sum((z(:,j)-sum(z(:,j))/size(x,1))**2)/size(x,1)
    end do
    total=0
    do j=1,size(x,2)
      total=total+sum((x(:,j)-sum(x(:,j))/size(x,1))**2)/size(x,1)
    end do
    self%explained_variance_ratio_=self%explained_variance_*0
    if(total>0) self%explained_variance_ratio_=self%explained_variance_/total
  end subroutine
  function svd_transform(self,x) result(z)
    class(truncated_svd), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call require(allocated(self%components_),'truncated_svd is not fitted')
    call check_xy(transpose(x),size(self%components_,2)); z=matmul(x,transpose(self%components_))
  end function
  function svd_inverse(self,z) result(x)
    class(truncated_svd), intent(in) :: self
    real(dp), intent(in) :: z(:,:)
    real(dp), allocatable :: x(:,:)
    call require(allocated(self%components_),'truncated_svd is not fitted')
    call check_xy(transpose(z),size(self%components_,1)); x=matmul(z,self%components_)
  end function
  function svd_fit_transform(self,x) result(z)
    class(truncated_svd), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x); z=self%transform(x)
  end function
  function nmf_fit_transform(self,x) result(w)
    class(nmf), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: w(:,:),h(:,:),numerator(:,:),denominator(:,:)
    real(dp) :: previous,error,initial,scale
    integer :: it,n,p,k
    call check_x(x); call require(all(x>=0),'NMF requires nonnegative X')
    n=size(x,1); p=size(x,2); k=self%n_components
    call require(k>0.and.self%max_iter>0.and.self%tol>0,'invalid NMF controls')
    allocate(w(n,k),h(k,p)); call seed_rng(self%random_state)
    call random_number(w); call random_number(h); scale=sqrt(sum(x)/(n*p*k))
    w=(w+0.1_dp)*scale; h=(h+0.1_dp)*scale
    initial=sqrt(sum((x-matmul(w,h))**2)); previous=initial; self%converged_=.false.
    do it=1,self%max_iter
      numerator=matmul(x,transpose(h)); denominator=matmul(w,matmul(h,transpose(h)))
      w=w*numerator/max(denominator,1.e-15_dp)
      numerator=matmul(transpose(w),x); denominator=matmul(matmul(transpose(w),w),h)
      h=h*numerator/max(denominator,1.e-15_dp)
      error=sqrt(sum((x-matmul(w,h))**2))
      if(abs(previous-error)<=self%tol*max(initial,epsilon(1._dp))) then
        self%converged_=.true.; exit
      end if
      previous=error
    end do
    self%n_iter_=min(it,self%max_iter); self%components_=h; self%reconstruction_err_=error
  end function
  subroutine nmf_fit(self,x)
    class(nmf), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: w(:,:)
    w=self%fit_transform(x)
  end subroutine
  function nmf_transform(self,x) result(w)
    class(nmf), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: w(:,:),gram(:,:),numerator(:,:)
    real(dp) :: previous,error,initial
    integer :: it
    call require(allocated(self%components_),'NMF is not fitted')
    call check_xy(transpose(x),size(self%components_,2)); call require(all(x>=0),'NMF requires nonnegative X')
    allocate(w(size(x,1),size(self%components_,1))); w=1
    gram=matmul(self%components_,transpose(self%components_)); numerator=matmul(x,transpose(self%components_))
    initial=sqrt(sum((x-matmul(w,self%components_))**2)); previous=initial
    do it=1,self%max_iter
      w=w*numerator/max(matmul(w,gram),1.e-15_dp)
      error=sqrt(sum((x-matmul(w,self%components_))**2))
      if(abs(previous-error)<=self%tol*max(initial,epsilon(1._dp))) exit
      previous=error
    end do
  end function
  function nmf_inverse(self,w) result(x)
    class(nmf), intent(in) :: self
    real(dp), intent(in) :: w(:,:)
    real(dp), allocatable :: x(:,:)
    call require(allocated(self%components_),'NMF is not fitted')
    call check_xy(transpose(w),size(self%components_,1)); x=matmul(w,self%components_)
  end function
end module
