module sklearn_discriminant_analysis
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_value,ieee_quiet_nan,ieee_positive_inf,ieee_negative_inf
  use sklearn_covariance, only: ledoit_wolf
  use sklearn_common
  use sklearn_metrics, only: accuracy_score
  implicit none
  private
  public :: linear_discriminant_analysis,quadratic_discriminant_analysis
  type :: quadratic_discriminant_analysis
    real(dp) :: reg_param=0
    integer, allocatable :: classes_(:)
    real(dp), allocatable :: means_(:,:),priors_(:),covariance_(:,:,:),precision_(:,:,:),logdet_(:)
  contains
    procedure :: fit=>qda_fit
    procedure :: decision_function=>qda_decision
    procedure :: predict_proba=>discriminant_proba
    procedure :: predict=>discriminant_predict
    procedure :: score=>discriminant_score
  end type
  type, extends(quadratic_discriminant_analysis) :: linear_discriminant_analysis
    character(8) :: solver='lsqr',shrinkage_mode='fixed'
    logical :: store_covariance=.false.
    integer :: n_components=0,n_components_=0,within_rank_=0,between_rank_=0
    real(dp) :: tol=1.e-4_dp,total_shrinkage_=-1
    real(dp), allocatable :: class_shrinkage_(:)
    real(dp), allocatable :: scalings_(:,:),xbar_(:),explained_variance_ratio_(:),eigenvalues_(:)
    real(dp) :: shrinkage=0
    real(dp), allocatable :: coef_(:,:),intercept_(:)
  contains
    procedure :: fit=>lda_fit
    procedure :: decision_function=>lda_decision
    procedure :: transform=>lda_transform
    procedure :: fit_transform=>lda_fit_transform
  end type
contains
  subroutine class_statistics(self,x,y,allow_singletons,store_covariance)
    class(quadratic_discriminant_analysis), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    logical, optional, intent(in) :: allow_singletons,store_covariance
    logical :: singleton,store
    real(dp) :: delta(size(x,2))
    integer :: c,i,j,n,p,k,nc
    singleton=.false.;if(present(allow_singletons)) singleton=allow_singletons
    store=.true.;if(present(store_covariance)) store=store_covariance
    call check_xy(x,size(y)); self%classes_=unique_labels(y); k=size(self%classes_)
    n=size(y); p=size(x,2); call require(k>=2,'discriminant analysis requires at least two classes')
    if(allocated(self%means_)) deallocate(self%means_)
    if(allocated(self%priors_)) deallocate(self%priors_)
    if(allocated(self%covariance_)) deallocate(self%covariance_)
    if(allocated(self%precision_)) deallocate(self%precision_)
    if(allocated(self%logdet_)) deallocate(self%logdet_)
    allocate(self%means_(k,p),self%priors_(k));self%means_=0
    if(store) then
      allocate(self%covariance_(p,p,k),self%precision_(p,p,k),self%logdet_(k))
      self%covariance_=0;self%precision_=0;self%logdet_=0
    end if
    do c=1,k
      nc=count(y==self%classes_(c))
      call require(singleton.or.nc>=2,'each discriminant class needs at least two samples')
      self%priors_(c)=real(nc,dp)/n
      do i=1,n
        if(y(i)==self%classes_(c)) self%means_(c,:)=self%means_(c,:)+x(i,:)
      end do
      self%means_(c,:)=self%means_(c,:)/nc
      if(.not.store) cycle
      do i=1,n
        if(y(i)/=self%classes_(c)) cycle
        delta=x(i,:)-self%means_(c,:)
        do j=1,p
          self%covariance_(:,j,c)=self%covariance_(:,j,c)+delta*delta(j)
        end do
      end do
      if(nc>1) self%covariance_(:,:,c)=self%covariance_(:,:,c)/(nc-1)
    end do
  end subroutine
  subroutine invert_covariance(cov,precision,logdet)
    real(dp), intent(in) :: cov(:,:)
    real(dp), intent(out) :: precision(:,:),logdet
    real(dp) :: vals(size(cov,1)),vecs(size(cov,1),size(cov,1))
    call symmetric_eigen(cov,vals,vecs)
    call require(all(vals>epsilon(1._dp)*maxval(abs(vals))), &
      'singular covariance; set positive reg_param or shrinkage')
    logdet=sum(log(vals)); precision=matmul(vecs/spread(vals,1,size(vals)),transpose(vecs))
  end subroutine
  subroutine qda_fit(self,x,y)
    class(quadratic_discriminant_analysis), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    integer :: c,j
    call require(self%reg_param>=0.and.self%reg_param<=1,'invalid QDA regularization')
    call class_statistics(self,x,y)
    do c=1,size(self%classes_)
      self%covariance_(:,:,c)=(1-self%reg_param)*self%covariance_(:,:,c)
      do j=1,size(x,2)
        self%covariance_(j,j,c)=self%covariance_(j,j,c)+self%reg_param
      end do
      call invert_covariance(self%covariance_(:,:,c),self%precision_(:,:,c),self%logdet_(c))
    end do
  end subroutine
  subroutine lda_fit(self,x,y)
    class(linear_discriminant_analysis), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), allocatable :: pooled(:,:),group(:,:),estimate(:,:)
    real(dp) :: trace
    integer :: c,j,p,nc,i,row
    call require(any(self%solver==[character(8)::'lsqr','svd','eigen']),'unknown LDA solver')
    call require(ieee_is_finite(self%tol).and.ieee_is_finite(self%shrinkage),'LDA controls must be finite')
    call require(self%tol>=0,'LDA tol must be nonnegative')
    call require(self%shrinkage>=0.and.self%shrinkage<=1,'invalid LDA shrinkage')
    call require(any(self%shrinkage_mode==[character(8)::'fixed','auto']),'unknown LDA shrinkage_mode')
    call require(self%n_components>=0,'LDA n_components must be nonnegative')
    if(allocated(self%class_shrinkage_)) deallocate(self%class_shrinkage_)
    self%total_shrinkage_=-1
    if(allocated(self%eigenvalues_)) deallocate(self%eigenvalues_)
    if(allocated(self%scalings_)) deallocate(self%scalings_)
    if(allocated(self%xbar_)) deallocate(self%xbar_)
    if(allocated(self%explained_variance_ratio_)) deallocate(self%explained_variance_ratio_)
    self%n_components_=0;self%within_rank_=0;self%between_rank_=0
    if(self%solver=='svd') then
      call require(self%shrinkage_mode=='fixed','LDA SVD solver does not support automatic shrinkage')
      call require(self%shrinkage==0,'LDA SVD solver does not support shrinkage')
      call lda_svd_fit(self,x,y)
      return
    end if
    call class_statistics(self,x,y,self%solver=='eigen'.or.self%shrinkage_mode=='auto')
    p=size(x,2);allocate(pooled(p,p));pooled=0
    if(self%shrinkage_mode=='auto'.and.self%solver=='lsqr') &
      call require(size(x,1)>size(self%classes_),'LDA auto needs more samples than classes')
    if(self%solver=='eigen') call require(size(x,1)>size(self%classes_),'LDA eigen needs more samples than classes')
    call require(self%n_components<=min(p,size(self%classes_)-1),'LDA n_components exceeds class/feature limit')
    if(self%shrinkage_mode=='auto') then
      allocate(self%class_shrinkage_(size(self%classes_)))
      do c=1,size(self%classes_)
        nc=count(y==self%classes_(c));allocate(group(nc,p));row=0
        do i=1,size(y)
          if(y(i)/=self%classes_(c)) cycle
          row=row+1;group(row,:)=x(i,:)
        end do
        estimate=lda_auto_covariance(group,self%class_shrinkage_(c))
        pooled=pooled+self%priors_(c)*estimate;deallocate(group)
      end do
    else
      ! sklearn lsqr covariance is population covariance, weighted by class prior.
      do c=1,size(self%classes_)
        nc=count(y==self%classes_(c))
        pooled=pooled+self%covariance_(:,:,c)*(nc-1)/size(y)
      end do
      trace=0
      do j=1,p
        trace=trace+pooled(j,j)/p
      end do
      pooled=(1-self%shrinkage)*pooled
      do j=1,p
        pooled(j,j)=pooled(j,j)+self%shrinkage*trace
      end do
    end if
    if(self%solver=='eigen') then
      call lda_eigen_fit(self,x,pooled)
      return
    end if
    do c=1,size(self%classes_)
      self%covariance_(:,:,c)=pooled
      if(self%shrinkage_mode=='auto') then
        call lda_pseudo_covariance(pooled,self%precision_(:,:,c),self%logdet_(c))
      else
        call invert_covariance(pooled,self%precision_(:,:,c),self%logdet_(c))
      end if
    end do
    self%coef_=matmul(self%means_,self%precision_(:,:,1))
    self%intercept_=log(self%priors_)-0.5_dp*sum(self%coef_*self%means_,dim=2)
  end subroutine
  function lda_auto_covariance(x,amount) result(covariance)
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(out) :: amount
    real(dp), allocatable :: covariance(:,:),centered(:,:),mean(:),variance(:),scale(:),bound(:)
    type(ledoit_wolf) :: estimate
    integer :: n,p
    n=size(x,1);p=size(x,2);mean=sum(x,dim=1)/n;centered=x-spread(mean,1,n)
    ! Corrected two-pass variance and StandardScaler's mean-dependent constant bound.
    variance=max(0._dp,(sum(centered*centered,dim=1)-sum(centered,dim=1)**2/n)/n)
    scale=sqrt(variance);bound=n*epsilon(1._dp)*variance+(n*mean*epsilon(1._dp))**2
    where(variance<=bound) scale=1
    centered=centered/spread(scale,1,n)
    call estimate%fit(centered);amount=estimate%shrinkage_
    covariance=estimate%covariance_*spread(scale,2,p)*spread(scale,1,p)
  end function
  subroutine lda_pseudo_covariance(covariance,precision,logdet)
    real(dp), intent(in) :: covariance(:,:)
    real(dp), intent(out) :: precision(:,:),logdet
    real(dp) :: s(size(covariance,1)),u(size(covariance,1),size(covariance,1))
    real(dp) :: v(size(covariance,1),size(covariance,1)),inverse(size(covariance,1)),cutoff
    integer :: j
    call dense_svd(covariance,s,u,v);cutoff=epsilon(1._dp)*maxval(s);inverse=0
    do j=1,size(s)
      if(s(j)>cutoff.and.sum(u(:,j)*u(:,j))>.5_dp) inverse(j)=1/s(j)
    end do
    precision=matmul(v*spread(inverse,1,size(s)),transpose(u))
    if(all(inverse>0)) then
      logdet=sum(log(s))
    else
      logdet=ieee_value(0._dp,ieee_negative_inf)
    end if
  end subroutine
  subroutine lda_eigen_fit(self,x,within)
    class(linear_discriminant_analysis), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),within(:,:)
    real(dp), allocatable :: centered(:,:),total(:,:),between(:,:),l(:,:),upper(:,:),inverse_l(:,:)
    real(dp), allocatable :: whitened(:,:),vectors(:,:),identity(:,:),ratio(:),precision(:,:)
    real(dp) :: trace,energy,logdet
    integer :: n,p,i,j,c,components,index
    n=size(x,1);p=size(x,2);components=min(p,size(self%classes_)-1)
    if(self%n_components>0) components=self%n_components
    if(self%shrinkage_mode=='auto') then
      total=lda_auto_covariance(x,self%total_shrinkage_)
    else
      centered=x-spread(sum(x,dim=1)/n,1,n)
      total=matmul(transpose(centered),centered)/n
      trace=0
      do j=1,p
        trace=trace+total(j,j)/p
      end do
      total=(1-self%shrinkage)*total
      do j=1,p
        total(j,j)=total(j,j)+self%shrinkage*trace
      end do
    end if
    between=total-within;l=cholesky_factor(within);upper=transpose(l)
    allocate(identity(p,p),inverse_l(p,p));identity=0
    do j=1,p
      identity(j,j)=1;inverse_l(:,j)=solve_lower(l,identity(:,j))
    end do
    whitened=matmul(inverse_l,matmul(between,transpose(inverse_l)))
    ! Roundoff can make the two triangles differ; the source eigh reads the lower.
    do j=1,p
      do i=j+1,p
        whitened(j,i)=whitened(i,j)
      end do
    end do
    allocate(self%eigenvalues_(p),vectors(p,p),self%scalings_(p,p))
    call symmetric_eigen(whitened,self%eigenvalues_,vectors)
    do j=1,p
      self%scalings_(:,j)=solve_upper(upper,vectors(:,j))
    end do
    energy=sum(self%eigenvalues_);allocate(ratio(p))
    if(energy/=0) then
      ratio=self%eigenvalues_/energy
      ratio=ratio(argsort(-ratio))
    else
      ! NumPy sorts NaNs before +Inf and -Inf after reversing ascending order.
      index=0
      do i=1,p
        if(self%eigenvalues_(i)/=0) cycle
        index=index+1;ratio(index)=ieee_value(0._dp,ieee_quiet_nan)
      end do
      do i=1,p
        if(self%eigenvalues_(i)<=0) cycle
        index=index+1;ratio(index)=ieee_value(0._dp,ieee_positive_inf)
      end do
      do i=1,p
        if(self%eigenvalues_(i)>=0) cycle
        index=index+1;ratio(index)=ieee_value(0._dp,ieee_negative_inf)
      end do
    end if
    self%explained_variance_ratio_=ratio(:components);self%n_components_=components
    self%coef_=matmul(matmul(self%means_,self%scalings_),transpose(self%scalings_))
    self%intercept_=log(self%priors_)-.5_dp*sum(self%means_*self%coef_,dim=2)
    precision=matmul(self%scalings_,transpose(self%scalings_));logdet=0
    do j=1,p
      logdet=logdet+2*log(l(j,j))
    end do
    do c=1,size(self%classes_)
      self%covariance_(:,:,c)=within;self%precision_(:,:,c)=precision;self%logdet_(c)=logdet
    end do
  end subroutine
  subroutine lda_thin_svd(a,s,u,v)
    real(dp), intent(in) :: a(:,:)
    real(dp), allocatable, intent(out) :: s(:),u(:,:),v(:,:)
    integer :: m,n
    m=size(a,1);n=size(a,2)
    if(m>=n) then
      allocate(s(n),u(m,n),v(n,n));call dense_svd(a,s,u,v)
    else
      allocate(s(m),u(m,m),v(n,m));call dense_svd(transpose(a),s,v,u)
    end if
  end subroutine
  subroutine lda_svd_fit(self,x,y)
    class(linear_discriminant_analysis), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), allocatable :: centered(:,:),std(:),s(:),u(:,:),v(:,:),within(:,:),between(:,:),coef(:,:)
    real(dp) :: total
    integer :: n,p,k,c,i,j,row,r,max_components
    call class_statistics(self,x,y,.true.,self%store_covariance)
    n=size(x,1);p=size(x,2);k=size(self%classes_)
    call require(n>k,'LDA SVD needs more samples than classes')
    max_components=min(p,k-1)
    call require(self%n_components<=max_components,'LDA n_components exceeds class/feature limit')
    if(self%n_components>0) max_components=self%n_components
    allocate(centered(n,p));row=0
    do c=1,k
      do i=1,n
        if(y(i)/=self%classes_(c)) cycle
        row=row+1;centered(row,:)=x(i,:)-self%means_(c,:)
      end do
    end do
    self%xbar_=matmul(self%priors_,self%means_)
    std=sqrt(sum((centered-spread(sum(centered,dim=1)/n,1,n))**2,dim=1)/n)
    where(std==0) std=1
    centered=centered/spread(std,1,n)/sqrt(real(n-k,dp))
    call lda_thin_svd(centered,s,u,v);r=count(s>self%tol);self%within_rank_=r
    call require(r>0,'LDA SVD has no retained within-class directions')
    within=v(:,:r)/spread(std,2,r)/spread(s(:r),1,p)
    between=matmul(self%means_-spread(self%xbar_,1,k),within)
    do c=1,k
      between(c,:)=between(c,:)*sqrt(real(n,dp)*self%priors_(c)/(k-1))
    end do
    call lda_thin_svd(between,s,u,v);total=sum(s*s)
    if(total>0) then
      self%explained_variance_ratio_=s(:min(max_components,size(s)))**2/total
    else
      allocate(self%explained_variance_ratio_(min(max_components,size(s))))
      self%explained_variance_ratio_=ieee_value(0._dp,ieee_quiet_nan)
    end if
    r=count(s>self%tol*s(1));self%between_rank_=r
    self%scalings_=matmul(within,v(:,:r));self%n_components_=min(max_components,r)
    coef=matmul(self%means_-spread(self%xbar_,1,k),self%scalings_)
    self%intercept_=log(self%priors_)-.5_dp*sum(coef*coef,dim=2)
    self%coef_=matmul(coef,transpose(self%scalings_))
    self%intercept_=self%intercept_-matmul(self%coef_,self%xbar_)
    ! Preserve the inherited covariance storage only when explicitly requested.
    if(.not.self%store_covariance) return
    self%covariance_(:,:,1)=0
    do c=1,k
      do i=1,n
        if(y(i)/=self%classes_(c)) cycle
        do j=1,p
          self%covariance_(:,j,1)=self%covariance_(:,j,1)+ &
            (x(i,:)-self%means_(c,:))*(x(i,j)-self%means_(c,j))/n
        end do
      end do
    end do
    do c=2,k
      self%covariance_(:,:,c)=self%covariance_(:,:,1)
    end do
  end subroutine
  function lda_transform(self,x) result(z)
    class(linear_discriminant_analysis), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call require(any(self%solver==[character(8)::'svd','eigen']),'LDA transform requires SVD solver or eigen solver')
    if(self%solver=='svd') then
      call require(allocated(self%scalings_),'LDA SVD is not fitted')
    else
      call require(allocated(self%scalings_),'LDA eigen model is not fitted')
    end if
    call check_xy(transpose(x),size(self%scalings_,1))
    if(self%solver=='svd') then
      z=matmul(x-spread(self%xbar_,1,size(x,1)),self%scalings_(:,:self%n_components_))
    else
      z=matmul(x,self%scalings_(:,:self%n_components_))
    end if
  end function
  function lda_fit_transform(self,x,y) result(z)
    class(linear_discriminant_analysis), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x,y);z=self%transform(x)
  end function
  function qda_decision(self,x) result(z)
    class(quadratic_discriminant_analysis), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    real(dp) :: delta(size(x,2))
    integer :: i,c
    call require(allocated(self%means_),'discriminant model is not fitted')
    call check_xy(transpose(x),size(self%means_,2)); allocate(z(size(x,1),size(self%classes_)))
    do i=1,size(x,1)
      do c=1,size(self%classes_)
        delta=x(i,:)-self%means_(c,:)
        z(i,c)=log(self%priors_(c))-0.5_dp*(self%logdet_(c)+ &
          dot_product(delta,matmul(self%precision_(:,:,c),delta)))
      end do
    end do
  end function
  function lda_decision(self,x) result(z)
    class(linear_discriminant_analysis), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:),difference(:),weight(:)
    integer :: c
    call require(allocated(self%coef_),'LDA is not fitted')
    call check_xy(transpose(x),size(self%coef_,2))
    if(self%shrinkage_mode=='auto') then
      ! Evaluate the same affine classifier in a centered, first-class gauge.
      ! Forming enormous common intercepts loses small, informative differences.
      allocate(z(size(x,1),size(self%classes_)));z=0
      do c=2,size(self%classes_)
        difference=self%means_(c,:)-self%means_(1,:)
        weight=matmul(difference,self%precision_(:,:,1))
        z(:,c)=matmul(x-spread(self%means_(1,:),1,size(x,1))-.5_dp*spread(difference,1,size(x,1)),weight)+ &
          log(self%priors_(c))-log(self%priors_(1))
      end do
    else
      z=matmul(x,transpose(self%coef_))+spread(self%intercept_,1,size(x,1))
    end if
  end function
  function discriminant_proba(self,x) result(prob)
    class(quadratic_discriminant_analysis), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: prob(:,:)
    integer :: i
    prob=self%decision_function(x)
    do i=1,size(x,1)
      prob(i,:)=exp(prob(i,:)-logsumexp(prob(i,:)))
    end do
  end function
  function discriminant_predict(self,x) result(y)
    class(quadratic_discriminant_analysis), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: y(size(x,1)),i
    real(dp), allocatable :: z(:,:)
    z=self%decision_function(x)
    do i=1,size(y)
      y(i)=self%classes_(maxloc(z(i,:),dim=1))
    end do
  end function
  real(dp) function discriminant_score(self,x,y) result(v)
    class(quadratic_discriminant_analysis), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    v=accuracy_score(y,self%predict(x))
  end function
end module
