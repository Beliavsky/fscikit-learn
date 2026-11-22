module sklearn_discriminant_analysis
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_value,ieee_quiet_nan
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
    character(8) :: solver='lsqr'
    logical :: store_covariance=.false.
    integer :: n_components=0,n_components_=0,within_rank_=0,between_rank_=0
    real(dp) :: tol=1.e-4_dp
    real(dp), allocatable :: scalings_(:,:),xbar_(:),explained_variance_ratio_(:)
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
    real(dp), allocatable :: pooled(:,:)
    real(dp) :: trace
    integer :: c,j,p,nc
    call require(any(self%solver==[character(8)::'lsqr','svd']),'unknown LDA solver')
    call require(ieee_is_finite(self%tol).and.ieee_is_finite(self%shrinkage),'LDA controls must be finite')
    call require(self%tol>=0,'LDA tol must be nonnegative')
    call require(self%shrinkage>=0.and.self%shrinkage<=1,'invalid LDA shrinkage')
    call require(self%n_components>=0,'LDA n_components must be nonnegative')
    if(allocated(self%scalings_)) deallocate(self%scalings_)
    if(allocated(self%xbar_)) deallocate(self%xbar_)
    if(allocated(self%explained_variance_ratio_)) deallocate(self%explained_variance_ratio_)
    self%n_components_=0;self%within_rank_=0;self%between_rank_=0
    if(self%solver=='svd') then
      call require(self%shrinkage==0,'LDA SVD solver does not support shrinkage')
      call lda_svd_fit(self,x,y)
      return
    end if
    call class_statistics(self,x,y); p=size(x,2);allocate(pooled(p,p));pooled=0
    call require(self%n_components<=min(p,size(self%classes_)-1),'LDA n_components exceeds class/feature limit')
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
    do c=1,size(self%classes_)
      self%covariance_(:,:,c)=pooled
      call invert_covariance(pooled,self%precision_(:,:,c),self%logdet_(c))
    end do
    self%coef_=matmul(self%means_,self%precision_(:,:,1))
    self%intercept_=log(self%priors_)-0.5_dp*sum(self%coef_*self%means_,dim=2)
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
    call require(self%solver=='svd','LDA transform requires SVD solver')
    call require(allocated(self%scalings_),'LDA SVD is not fitted')
    call check_xy(transpose(x),size(self%scalings_,1))
    z=matmul(x-spread(self%xbar_,1,size(x,1)),self%scalings_(:,:self%n_components_))
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
    real(dp), allocatable :: z(:,:)
    call require(allocated(self%coef_),'LDA is not fitted')
    call check_xy(transpose(x),size(self%coef_,2))
    z=matmul(x,transpose(self%coef_))+spread(self%intercept_,1,size(x,1))
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
