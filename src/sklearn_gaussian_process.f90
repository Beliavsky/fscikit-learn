module sklearn_gaussian_process
  use sklearn_gp_classification, only: gaussian_process_classifier
  use sklearn_common
  use sklearn_metrics, only: r2_score
  implicit none
  private
  public :: gaussian_process_classifier
  public :: gaussian_process_regressor
  type :: gaussian_process_regressor
    real(dp) :: alpha=1.e-10_dp,length_scale=1,amplitude=1,noise_level=0
    logical :: normalize_y=.false.
    real(dp) :: y_train_mean_=0,y_train_std_=1,log_marginal_likelihood_=0
    real(dp), allocatable :: x_train_(:,:),alpha_(:),l_(:,:)
  contains
    procedure :: fit=>gp_fit
    procedure :: predict=>gp_predict
    procedure :: predict_std=>gp_predict_std
    procedure :: predict_cov=>gp_predict_cov
    procedure :: score=>gp_score
  end type
contains
  function covariance(self,x,y) result(k)
    class(gaussian_process_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp) :: k(size(x,1),size(y,1))
    integer :: i,j
    call check_x(x); call check_x(y); call require(size(x,2)==size(y,2),'GP feature count mismatch')
    do j=1,size(y,1)
      do i=1,size(x,1)
        k(i,j)=self%amplitude*exp(-sum((x(i,:)-y(j,:))**2)/(2*self%length_scale**2))
      end do
    end do
  end function
  subroutine gp_fit(self,x,y)
    class(gaussian_process_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp) :: gram(size(y),size(y)),target(size(y))
    integer :: i,n
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1])); n=size(y)
    call require(self%alpha>=0.and.self%noise_level>=0.and.self%length_scale>0.and.self%amplitude>0, &
      'invalid Gaussian process kernel parameters')
    self%x_train_=x; self%y_train_mean_=0; self%y_train_std_=1
    if(self%normalize_y) then
      self%y_train_mean_=sum(y)/n; self%y_train_std_=sqrt(sum((y-self%y_train_mean_)**2)/n)
      if(self%y_train_std_<10*epsilon(1._dp)) self%y_train_std_=1
    end if
    target=(y-self%y_train_mean_)/self%y_train_std_
    gram=covariance(self,x,x)
    do i=1,n
      gram(i,i)=gram(i,i)+self%alpha+self%noise_level
    end do
    self%l_=cholesky_factor(gram)
    self%alpha_=solve_upper(transpose(self%l_),solve_lower(self%l_,target))
    self%log_marginal_likelihood_=-0.5_dp*dot_product(target,self%alpha_)-0.5_dp*n*log(2*acos(-1._dp))
    do i=1,n
      self%log_marginal_likelihood_=self%log_marginal_likelihood_-log(self%l_(i,i))
    end do
  end subroutine
  function gp_predict(self,x) result(y)
    class(gaussian_process_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: y(size(x,1))
    call require(allocated(self%alpha_),'Gaussian process is not fitted')
    y=matmul(covariance(self,x,self%x_train_),self%alpha_)*self%y_train_std_+self%y_train_mean_
  end function
  function gp_predict_cov(self,x) result(cov)
    class(gaussian_process_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: cov(size(x,1),size(x,1))
    real(dp), allocatable :: cross(:,:),v(:,:)
    integer :: j
    call require(allocated(self%alpha_),'Gaussian process is not fitted')
    cross=covariance(self,self%x_train_,x); allocate(v(size(cross,1),size(cross,2)))
    do j=1,size(x,1)
      v(:,j)=solve_lower(self%l_,cross(:,j))
    end do
    cov=covariance(self,x,x)-matmul(transpose(v),v)
    do j=1,size(x,1)
      cov(j,j)=max(0._dp,cov(j,j)+self%noise_level)
    end do
    cov=cov*self%y_train_std_**2
  end function
  function gp_predict_std(self,x) result(std)
    class(gaussian_process_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: std(size(x,1))
    real(dp), allocatable :: cross(:,:),v(:)
    integer :: j
    call require(allocated(self%alpha_),'Gaussian process is not fitted')
    cross=covariance(self,self%x_train_,x)
    do j=1,size(x,1)
      v=solve_lower(self%l_,cross(:,j))
      std(j)=sqrt(max(0._dp,self%amplitude+self%noise_level-dot_product(v,v)))*self%y_train_std_
    end do
  end function
  real(dp) function gp_score(self,x,y) result(v)
    class(gaussian_process_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    v=r2_score(y,self%predict(x))
  end function
end module
