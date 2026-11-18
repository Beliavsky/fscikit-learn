module sklearn_discriminant_analysis
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
    real(dp) :: shrinkage=0
    real(dp), allocatable :: coef_(:,:),intercept_(:)
  contains
    procedure :: fit=>lda_fit
    procedure :: decision_function=>lda_decision
  end type
contains
  subroutine class_statistics(self,x,y)
    class(quadratic_discriminant_analysis), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp) :: delta(size(x,2))
    integer :: c,i,j,n,p,k,nc
    call check_xy(x,size(y)); self%classes_=unique_labels(y); k=size(self%classes_)
    n=size(y); p=size(x,2); call require(k>=2,'discriminant analysis requires at least two classes')
    if(allocated(self%means_)) deallocate(self%means_,self%priors_,self%covariance_,self%precision_,self%logdet_)
    allocate(self%means_(k,p),self%priors_(k),self%covariance_(p,p,k),self%precision_(p,p,k),self%logdet_(k))
    self%means_=0; self%covariance_=0
    do c=1,k
      nc=count(y==self%classes_(c)); call require(nc>=2,'each discriminant class needs at least two samples')
      self%priors_(c)=real(nc,dp)/n
      do i=1,n
        if(y(i)==self%classes_(c)) self%means_(c,:)=self%means_(c,:)+x(i,:)
      end do
      self%means_(c,:)=self%means_(c,:)/nc
      do i=1,n
        if(y(i)/=self%classes_(c)) cycle
        delta=x(i,:)-self%means_(c,:)
        do j=1,p
          self%covariance_(:,j,c)=self%covariance_(:,j,c)+delta*delta(j)
        end do
      end do
      self%covariance_(:,:,c)=self%covariance_(:,:,c)/(nc-1)
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
    real(dp) :: pooled(size(x,2),size(x,2)),trace
    integer :: c,j,p,nc
    call require(self%shrinkage>=0.and.self%shrinkage<=1,'invalid LDA shrinkage')
    call class_statistics(self,x,y); p=size(x,2); pooled=0
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
