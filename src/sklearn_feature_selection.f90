module sklearn_feature_selection
  use sklearn_recursive_selection, only: rfe,rfecv,rfecv_results
  use sklearn_sequential_selection, only: sequential_feature_selector
  use sklearn_common
  implicit none
  private
  public :: rfe,rfecv,rfecv_results
  public :: sequential_feature_selector
  public :: variance_threshold,select_k_best,f_classif,f_regression,chi2
  type :: variance_threshold
    real(dp) :: threshold=0
    real(dp), allocatable :: variances_(:)
    logical, allocatable :: support_(:)
  contains
    procedure :: fit=>variance_fit
    procedure :: transform=>variance_transform
    procedure :: inverse_transform=>variance_inverse
    procedure :: get_support=>variance_support
  end type
  type :: select_k_best
    integer :: k=10
    real(dp), allocatable :: scores_(:)
    logical, allocatable :: support_(:)
  contains
    procedure :: fit_classification=>select_class_fit
    procedure :: fit_regression=>select_reg_fit
    generic :: fit=>fit_classification,fit_regression
    procedure :: transform=>select_transform
    procedure :: inverse_transform=>select_inverse
    procedure :: get_support=>select_support
  end type
contains
  function f_classif(x,y) result(scores)
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp) :: scores(size(x,2)),mean(size(x,2)),center(size(x,2)),between(size(x,2)),within(size(x,2))
    integer, allocatable :: classes(:)
    integer :: c,i,n,k
    call check_xy(x,size(y)); classes=unique_labels(y); k=size(classes); n=size(y)
    call require(k>1.and.n>k,'ANOVA requires multiple classes and residual degrees of freedom')
    mean=sum(x,dim=1)/n; between=0; within=0
    do c=1,k
      center=0
      do i=1,n
        if(y(i)==classes(c)) center=center+x(i,:)
      end do
      center=center/count(y==classes(c)); between=between+count(y==classes(c))*(center-mean)**2
      do i=1,n
        if(y(i)==classes(c)) within=within+(x(i,:)-center)**2
      end do
    end do
    scores=between/(k-1)/max(within/(n-k),tiny(1._dp))
  end function
  function f_regression(x,y) result(scores)
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp) :: scores(size(x,2)),xc(size(x,1)),yc(size(y)),r,den
    integer :: j,n
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    n=size(y); call require(n>2,'F regression requires n>2')
    yc=y-sum(y)/n
    do j=1,size(x,2)
      xc=x(:,j)-sum(x(:,j))/n; den=sqrt(sum(xc**2)*sum(yc**2)); r=0
      if(den>0) r=min(1._dp,(dot_product(xc,yc)/den)**2)
      scores(j)=r*real(n-2,dp)/max(1-r,epsilon(1._dp))
    end do
  end function
  function chi2(x,y) result(scores)
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp) :: scores(size(x,2)),observed(size(x,2)),expected(size(x,2)),total(size(x,2))
    integer, allocatable :: classes(:)
    integer :: c,i
    call check_xy(x,size(y)); call require(all(x>=0),'chi2 requires nonnegative X')
    classes=unique_labels(y); scores=0; total=sum(x,dim=1)
    do c=1,size(classes)
      observed=0
      do i=1,size(y)
        if(y(i)==classes(c)) observed=observed+x(i,:)
      end do
      expected=total*real(count(y==classes(c)),dp)/size(y)
      scores=scores+(observed-expected)**2/max(expected,tiny(1._dp))
    end do
  end function
  subroutine variance_fit(self,x)
    class(variance_threshold), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: j
    call check_x(x); call require(self%threshold>=0,'variance threshold must be nonnegative')
    self%variances_=sum(x,dim=1)*0
    do j=1,size(x,2)
      self%variances_(j)=sum((x(:,j)-sum(x(:,j))/size(x,1))**2)/size(x,1)
    end do
    self%support_=self%variances_>self%threshold
    call require(any(self%support_),'no feature meets variance threshold')
  end subroutine
  subroutine choose(self)
    class(select_k_best), intent(inout) :: self
    integer :: order(size(self%scores_))
    call require(self%k>=0.and.self%k<=size(self%scores_),'invalid k in select_k_best')
    order=argsort(self%scores_); self%support_=self%scores_<-huge(1._dp)
    self%support_(order(size(order)-self%k+1:))=.true.
  end subroutine
  subroutine select_class_fit(self,x,y,score_func)
    class(select_k_best), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    character(*), intent(in), optional :: score_func
    character(16) :: kind
    kind='f_classif'; if(present(score_func)) kind=score_func
    select case(trim(kind))
    case('f_classif'); self%scores_=f_classif(x,y)
    case('chi2'); self%scores_=chi2(x,y)
    case default; call require(.false.,'unknown classification feature score')
    end select
    call choose(self)
  end subroutine
  subroutine select_reg_fit(self,x,y)
    class(select_k_best), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    self%scores_=f_regression(x,y); call choose(self)
  end subroutine
  function subset(x,support) result(z)
    real(dp), intent(in) :: x(:,:)
    logical, intent(in) :: support(:)
    real(dp), allocatable :: z(:,:)
    integer :: indices(size(support)),j
    call check_xy(transpose(x),size(support)); indices=[(j,j=1,size(support))]
    z=x(:,pack(indices,support))
  end function
  function restore(x,support) result(z)
    real(dp), intent(in) :: x(:,:)
    logical, intent(in) :: support(:)
    real(dp), allocatable :: z(:,:)
    integer :: indices(size(support)),j
    call require(size(x,2)==count(support),'inverse feature count mismatch')
    allocate(z(size(x,1),size(support))); z=0; indices=[(j,j=1,size(support))]
    z(:,pack(indices,support))=x
  end function
  function variance_transform(self,x) result(z)
    class(variance_threshold), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call require(allocated(self%support_),'variance_threshold is not fitted'); z=subset(x,self%support_)
  end function
  function variance_inverse(self,x) result(z)
    class(variance_threshold), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call require(allocated(self%support_),'variance_threshold is not fitted'); z=restore(x,self%support_)
  end function
  function variance_support(self) result(mask)
    class(variance_threshold), intent(in) :: self
    logical, allocatable :: mask(:)
    call require(allocated(self%support_),'variance_threshold is not fitted'); mask=self%support_
  end function
  function select_transform(self,x) result(z)
    class(select_k_best), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call require(allocated(self%support_),'select_k_best is not fitted'); z=subset(x,self%support_)
  end function
  function select_inverse(self,x) result(z)
    class(select_k_best), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call require(allocated(self%support_),'select_k_best is not fitted'); z=restore(x,self%support_)
  end function
  function select_support(self) result(mask)
    class(select_k_best), intent(in) :: self
    logical, allocatable :: mask(:)
    call require(allocated(self%support_),'select_k_best is not fitted'); mask=self%support_
  end function
end module
