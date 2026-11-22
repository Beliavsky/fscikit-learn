module sklearn_decomposition
  use sklearn_minibatch_nmf, only: mini_batch_nmf,nmf_initialize_factors
  use sklearn_lda, only: latent_dirichlet_allocation
  use sklearn_dictionary, only: sparse_pca,dictionary_learning,sparse_coder,sparse_encode, &
    mini_batch_dictionary_learning,mini_batch_sparse_pca
  use sklearn_incremental_pca, only: incremental_pca
  use sklearn_fast_ica, only: fast_ica
  use sklearn_common
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_sparse, only: csr_matrix
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
    integer :: n_components=2,n_components_=0,n_features_in_=0,n_iter=5,n_oversamples=10,random_state=0
    character(16) :: algorithm='full'
    character(8) :: power_iteration_normalizer='auto'
    real(dp), allocatable :: components_(:,:),singular_values_(:),explained_variance_(:)
    real(dp), allocatable :: explained_variance_ratio_(:)
  contains
    procedure, private :: fit_dense=>svd_fit,fit_sparse=>svd_fit_sparse
    generic :: fit=>fit_dense,fit_sparse
    procedure, private :: transform_dense=>svd_transform,transform_sparse=>svd_transform_sparse
    generic :: transform=>transform_dense,transform_sparse
    procedure :: inverse_transform=>svd_inverse
    procedure, private :: fit_transform_dense=>svd_fit_transform,fit_transform_sparse=>svd_fit_transform_sparse
    generic :: fit_transform=>fit_transform_dense,fit_transform_sparse
  end type
  type :: nmf
    integer :: n_components=2,max_iter=1000,n_iter_=0,random_state=0,n_components_=0,n_features_in_=0
    real(dp) :: tol=1.e-6_dp,reconstruction_err_=0,alpha_w=0,alpha_h=-1,l1_ratio=0
    real(dp) :: cd_violation_=0,cd_initial_violation_=0
    logical :: converged_=.false.,shuffle=.false.
    character(8) :: solver='mu',init='auto'
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
  subroutine svd_controls(self,n,p,sparse)
    class(truncated_svd), intent(in) :: self
    integer, intent(in) :: n,p
    logical, intent(in) :: sparse
    call require(any(self%algorithm==[character(16)::'full','randomized']),'unsupported SVD algorithm')
    call require(self%n_iter>=0,'SVD n_iter must be nonnegative')
    call require(self%n_oversamples>0,'SVD n_oversamples must be positive')
    call require(self%random_state>=0,'SVD random_state must be nonnegative')
    call require(any(self%power_iteration_normalizer==[character(8)::'auto','none','LU','QR']), &
      'unsupported SVD power_iteration_normalizer')
    if(sparse) call require(self%algorithm=='randomized','SVD CSR fitting requires randomized algorithm')
    if(self%algorithm=='full') then
      call require(self%n_components>0.and.self%n_components<=min(n,p),'invalid SVD n_components')
    else
      call require(p>=2,'randomized SVD requires at least two features')
      call require(self%n_components>0.and.self%n_components<=p,'invalid SVD n_components')
      call require(self%n_components<=huge(1)-self%n_oversamples,'SVD sampling dimension exceeds integer capacity')
    end if
  end subroutine
  subroutine svd_orient(self,u,vt)
    class(truncated_svd), intent(inout) :: self
    real(dp), intent(inout) :: u(:,:),vt(:,:)
    real(dp), allocatable :: v(:,:)
    integer :: j,index
    ! Complete only this estimator's zero-singular bases, as in PCA.
    call pca_complete_columns(u);v=transpose(vt);call pca_complete_columns(v);vt=transpose(v)
    do j=1,size(vt,1)
      index=maxloc(abs(vt(j,:)),dim=1)
      if(vt(j,index)>=0) cycle
      vt(j,:)=-vt(j,:);u(:,j)=-u(:,j)
    end do
    self%components_=vt;self%n_components_=size(vt,1);self%n_features_in_=size(vt,2)
  end subroutine
  subroutine svd_statistics(self,z,total)
    class(truncated_svd), intent(inout) :: self
    real(dp), intent(in) :: z(:,:),total
    integer :: j
    self%explained_variance_=self%singular_values_*0
    do j=1,size(z,2)
      self%explained_variance_(j)=sum((z(:,j)-sum(z(:,j))/size(z,1))**2)/size(z,1)
    end do
    self%explained_variance_ratio_=self%explained_variance_*0
    if(total>0) self%explained_variance_ratio_=self%explained_variance_/total
  end subroutine
  subroutine svd_fit(self,x)
    class(truncated_svd), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: u(:,:),values(:),vectors(:,:),vt(:,:),z(:,:)
    real(dp) :: total
    type(legacy_random_state) :: rng
    integer :: k,j,n,p
    call check_x(x);n=size(x,1);p=size(x,2);k=self%n_components;call svd_controls(self,n,p,.false.)
    if(self%algorithm=='full') then
      allocate(u(n,p),values(p),vectors(p,p));call dense_svd(x,values,u,vectors)
      self%components_=transpose(vectors(:,:k));self%singular_values_=values(:k)
      self%n_components_=k;self%n_features_in_=p
    else
      call rng%seed(self%random_state)
      call randomized_svd(x,k,rng,values,u,vt,n_iter=self%n_iter,n_oversamples=self%n_oversamples, &
        power_iteration_normalizer=self%power_iteration_normalizer,flip_sign=.false.)
      call svd_orient(self,u,vt);self%singular_values_=values
    end if
    z=matmul(x,transpose(self%components_));total=0
    do j=1,p
      total=total+sum((x(:,j)-sum(x(:,j))/n)**2)/n
    end do
    call svd_statistics(self,z,total)
  end subroutine
  subroutine svd_fit_sparse(self,x)
    class(truncated_svd), intent(inout) :: self
    type(csr_matrix), intent(in) :: x
    real(dp), allocatable :: u(:,:),s(:),vt(:,:),z(:,:)
    type(legacy_random_state) :: rng
    call x%validate();call require(x%n_rows>0.and.x%n_cols>0,'SVD CSR shape must be nonempty')
    call require(all(ieee_is_finite(x%data)),'SVD CSR data must be finite')
    call svd_controls(self,x%n_rows,x%n_cols,.true.);call rng%seed(self%random_state)
    call randomized_svd(x,self%n_components,rng,s,u,vt,n_iter=self%n_iter,n_oversamples=self%n_oversamples, &
      power_iteration_normalizer=self%power_iteration_normalizer,flip_sign=.false.)
    call svd_orient(self,u,vt);self%singular_values_=s;z=x%matmul(transpose(self%components_))
    call svd_statistics(self,z,svd_sparse_variance(x))
  end subroutine
  real(dp) function svd_sparse_variance(x) result(total)
    type(csr_matrix), intent(in) :: x
    real(dp), allocatable :: means(:),squares(:),work(:)
    integer, allocatable :: marker(:),touched(:),counts(:)
    integer :: pass,row,e,col,count,j
    allocate(means(x%n_cols),squares(x%n_cols),work(x%n_cols),marker(x%n_cols),touched(x%n_cols),counts(x%n_cols))
    means=0;squares=0;work=0;counts=0
    do pass=1,2
      marker=0
      do row=1,x%n_rows
        count=0
        do e=x%indptr(row),x%indptr(row+1)-1
          col=x%indices(e)
          if(marker(col)/=row) then
            marker(col)=row;work(col)=0;count=count+1;touched(count)=col
          end if
          work(col)=work(col)+x%data(e)
        end do
        do j=1,count
          col=touched(j)
          if(pass==1) then
            means(col)=means(col)+work(col);counts(col)=counts(col)+1
          else
            squares(col)=squares(col)+(work(col)-means(col))**2
          end if
        end do
      end do
      if(pass==1) means=means/x%n_rows
    end do
    total=sum((squares+(x%n_rows-counts)*means**2)/x%n_rows)
  end function
  function svd_transform_sparse(self,x) result(z)
    class(truncated_svd), intent(in) :: self
    type(csr_matrix), intent(in) :: x
    real(dp), allocatable :: z(:,:)
    call require(allocated(self%components_),'truncated_svd is not fitted');call x%validate()
    call require(x%n_rows>0.and.x%n_cols>0,'SVD CSR shape must be nonempty')
    call require(all(ieee_is_finite(x%data)),'SVD CSR data must be finite')
    call require(x%n_cols==size(self%components_,2),'SVD CSR feature count mismatch')
    z=x%matmul(transpose(self%components_))
  end function
  function svd_fit_transform_sparse(self,x) result(z)
    class(truncated_svd), intent(inout) :: self
    type(csr_matrix), intent(in) :: x
    real(dp), allocatable :: z(:,:)
    call self%fit(x);z=self%transform(x)
  end function
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
  function nmf_fit_transform(self,x,w0,h0) result(w)
    class(nmf), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: w0(:,:),h0(:,:)
    real(dp), allocatable :: w(:,:),h(:,:),numerator(:,:),denominator(:,:)
    real(dp) :: previous,error,initial,scale
    integer :: it,n,p,k
    call nmf_controls(self)
    if(self%solver=='cd') then
      w=nmf_cd_fit(self,x,w0,h0);return
    end if
    call require(self%init=='auto'.and..not.present(w0).and..not.present(h0), &
      'NMF initialization options require cd solver')
    call check_x(x); call require(all(x>=0),'NMF requires nonnegative X')
    n=size(x,1); p=size(x,2); k=self%n_components
    call require(k>0.and.self%max_iter>0.and.self%tol>0,'invalid NMF controls')
    self%n_components_=k;self%n_features_in_=p;self%cd_violation_=0;self%cd_initial_violation_=0
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
  subroutine nmf_fit(self,x,w0,h0)
    class(nmf), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: w0(:,:),h0(:,:)
    real(dp), allocatable :: w(:,:)
    w=self%fit_transform(x,w0,h0)
  end subroutine
  function nmf_transform(self,x) result(w)
    class(nmf), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: w(:,:),gram(:,:),numerator(:,:)
    real(dp) :: previous,error,initial
    integer :: it
    call require(allocated(self%components_),'NMF is not fitted')
    call nmf_controls(self)
    if(self%solver=='cd') then
      w=nmf_cd_transform(self,x);return
    end if
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
  subroutine nmf_controls(self)
    class(nmf), intent(in) :: self
    call require(self%solver=='mu'.or.self%solver=='cd','unknown NMF solver')
    call require(self%n_components>=0.and.self%max_iter>0.and.self%random_state>=0,'invalid NMF controls')
    call require(all(ieee_is_finite([self%tol,self%alpha_w,self%alpha_h,self%l1_ratio])), &
      'NMF controls must be finite')
    call require(self%tol>=0.and.self%alpha_w>=0.and.(self%alpha_h==-1.or.self%alpha_h>=0), &
      'NMF tolerance/regularization must be nonnegative (alpha_h -1 means same)')
    call require(self%l1_ratio>=0.and.self%l1_ratio<=1,'NMF l1_ratio must lie in [0,1]')
    call require(any(self%init==[character(8)::'auto','random','custom','nndsvd','nndsvda','nndsvdar']), &
      'unknown NMF initializer')
    if(self%solver=='mu') call require(self%alpha_w==0.and.self%alpha_h==-1.and.self%l1_ratio==0.and..not.self%shuffle, &
      'NMF regularization/shuffle controls require cd solver')
  end subroutine
  subroutine nmf_cd_data(x)
    real(dp), intent(in) :: x(:,:)
    call check_x(x);call require(all(x>=0),'NMF requires nonnegative X')
    call require(maxval(x)<=huge(1._dp)**(1._dp/3)/ &
      (64*real(max(size(x,1),size(x,2)),dp)),'NMF data exceeds working range')
  end subroutine
  subroutine nmf_cd_regularization(self,n,p,l1w,l2w,l1h,l2h)
    class(nmf), intent(in) :: self
    integer, intent(in) :: n,p
    real(dp), intent(out) :: l1w,l2w,l1h,l2h
    real(dp) :: ah
    ah=self%alpha_h;if(ah==-1) ah=self%alpha_w
    call require(max(self%alpha_w,ah)<=huge(1._dp)**(1._dp/3)/(64*real(max(n,p),dp)), &
      'NMF regularization exceeds working range')
    l1w=p*self%alpha_w*self%l1_ratio;l2w=p*self%alpha_w*(1-self%l1_ratio)
    l1h=n*ah*self%l1_ratio;l2h=n*ah*(1-self%l1_ratio)
  end subroutine
  real(dp) function nmf_cd_sweep(x,w,other,l1,l2,shuffle,rng) result(violation)
    real(dp), intent(in) :: x(:,:),other(:,:),l1,l2
    real(dp), intent(inout) :: w(:,:)
    logical, intent(in) :: shuffle
    type(legacy_random_state), intent(inout) :: rng
    real(dp), allocatable :: gram(:,:),cross(:,:)
    real(dp) :: gradient,projected,hessian,limit,candidate
    integer :: order(size(other,2)),i,r,s,t,k,j,temp
    k=size(other,2);limit=huge(1._dp)**(1._dp/3)/(64*real(max(size(x,1),size(x,2),k),dp))
    call require(all(ieee_is_finite(w)).and.all(ieee_is_finite(other)),'NMF factors must be finite')
    call require(maxval(w)<=limit.and.maxval(other)<=limit,'NMF factors exceed working range')
    gram=matmul(transpose(other),other);cross=matmul(x,other)
    do t=1,k
      gram(t,t)=gram(t,t)+l2
    end do
    cross=cross-l1;order=[(t,t=1,k)]
    if(shuffle) then
      do i=k,2,-1
        j=rng%integer_index(i);temp=order(i);order(i)=order(j);order(j)=temp
      end do
    end if
    violation=0
    do s=1,k
      t=order(s);hessian=gram(t,t)
      do i=1,size(w,1)
        gradient=-cross(i,t)
        do r=1,k
          gradient=gradient+gram(t,r)*w(i,r)
        end do
        projected=gradient
        if(w(i,t)==0) projected=min(0._dp,gradient)
        violation=violation+abs(projected)
        if(hessian==0) cycle
        if(hessian<1) call require(abs(gradient)<=huge(1._dp)*hessian/4,'NMF coordinate step exceeds working range')
        candidate=max(w(i,t)-gradient/hessian,0._dp)
        call require(candidate<=limit,'NMF factors exceed working range')
        w(i,t)=candidate
      end do
    end do
  end function
  subroutine nmf_cd_solve(self,x,w,h,update_h,n_iter,violation,initial,converged)
    class(nmf), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(inout) :: w(:,:),h(:,:)
    logical, intent(in) :: update_h
    integer, intent(out) :: n_iter
    real(dp), intent(out) :: violation,initial
    logical, intent(out) :: converged
    real(dp), allocatable :: ht(:,:)
    real(dp) :: l1w,l2w,l1h,l2h
    type(legacy_random_state) :: rng
    integer :: it
    call nmf_cd_regularization(self,size(x,1),size(x,2),l1w,l2w,l1h,l2h)
    call rng%seed(self%random_state);ht=transpose(h);initial=0;converged=.false.
    do it=1,self%max_iter
      violation=nmf_cd_sweep(x,w,ht,l1w,l2w,self%shuffle,rng)
      if(update_h) violation=violation+nmf_cd_sweep(transpose(x),ht,w,l1h,l2h,self%shuffle,rng)
      if(it==1) initial=violation
      if(initial==0) then
        converged=.true.;exit
      end if
      if(violation/initial<=self%tol) then
        converged=.true.;exit
      end if
    end do
    n_iter=min(it,self%max_iter);if(update_h) h=transpose(ht)
  end subroutine
  function nmf_cd_fit(self,x,w0,h0) result(w)
    class(nmf), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: w0(:,:),h0(:,:)
    real(dp), allocatable :: w(:,:),h(:,:)
    integer :: k,n,p
    call nmf_cd_data(x);n=size(x,1);p=size(x,2);k=self%n_components
    if(k==0) then
      k=p
      if(self%init=='custom'.and.present(h0)) k=size(h0,1)
    end if
    call require(k>0,'NMF component count must be positive')
    if(any(self%init==[character(8)::'nndsvd','nndsvda','nndsvdar'])) &
      call require(k<=min(n,p),'NMF NNDSVD requires components <= min(samples,features)')
    if(self%init=='custom') then
      call require(present(w0).and.present(h0),'NMF custom initialization requires W and H')
      call require(all(shape(w0)==[n,k]).and.all(shape(h0)==[k,p]),'NMF initializer shape mismatch')
      call require(all(ieee_is_finite(w0)).and.all(ieee_is_finite(h0)),'NMF initializer must be finite')
      call require(all(w0>=0).and.all(h0>=0),'NMF initializer must be nonnegative')
      call require(maxval(w0)>0.and.maxval(h0)>0,'NMF initializer must not be all zero')
    end if
    call nmf_initialize_factors(x,k,self%init,self%random_state,w,h,w0,h0)
    self%n_components_=k;self%n_features_in_=p
    call nmf_cd_solve(self,x,w,h,.true.,self%n_iter_,self%cd_violation_,self%cd_initial_violation_,self%converged_)
    self%components_=h;self%reconstruction_err_=sqrt(sum((x-matmul(w,h))**2))
  end function
  function nmf_cd_transform(self,x) result(w)
    class(nmf), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: w(:,:),h(:,:)
    real(dp) :: violation,initial
    logical :: converged
    integer :: n_iter
    call nmf_cd_data(x)
    call require(size(x,2)==self%n_features_in_,'NMF query feature count mismatch')
    call require(maxval(self%components_)>0,'NMF transform components must not be all zero')
    allocate(w(size(x,1),self%n_components_));w=0;h=self%components_
    call nmf_cd_solve(self,x,w,h,.false.,n_iter,violation,initial,converged)
  end function
end module
