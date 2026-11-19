module sklearn_naive_bayes
  use sklearn_categorical_nb, only: categorical_nb,categorical_table
  use sklearn_common
  use sklearn_metrics, only: accuracy_score
  implicit none
  private
  public :: categorical_nb,categorical_table
  public :: gaussian_nb, multinomial_nb, bernoulli_nb, complement_nb
  type :: gaussian_nb
    real(dp) :: var_smoothing=1.e-9_dp,epsilon_=0
    integer, allocatable :: classes_(:)
    real(dp), allocatable :: theta_(:,:),var_(:,:),class_prior_(:),class_count_(:)
  contains
    procedure :: fit=>gaussian_fit
    procedure :: predict_log_proba=>gaussian_log_proba
    procedure :: predict_proba=>gaussian_proba
    procedure :: predict=>gaussian_predict
    procedure :: score=>gaussian_score
  end type
  type :: multinomial_nb
    real(dp) :: alpha=1
    logical :: fit_prior=.true.
    integer, allocatable :: classes_(:)
    real(dp), allocatable :: feature_log_prob_(:,:),class_log_prior_(:),class_count_(:)
  contains
    procedure :: fit=>multinomial_fit
    procedure :: joint_log_likelihood=>multinomial_joint
    procedure :: predict_log_proba=>discrete_log_proba
    procedure :: predict_proba=>discrete_proba
    procedure :: predict=>discrete_predict
    procedure :: score=>discrete_score
  end type
  type, extends(multinomial_nb) :: bernoulli_nb
    real(dp) :: binarize_threshold=0
  contains
    procedure :: fit=>bernoulli_fit
    procedure :: joint_log_likelihood=>bernoulli_joint
  end type
  type, extends(multinomial_nb) :: complement_nb
    logical :: norm=.false.
  contains
    procedure :: fit=>complement_fit
    procedure :: joint_log_likelihood=>complement_joint
  end type
contains
  subroutine gaussian_fit(self,x,y)
    class(gaussian_nb), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp) :: mean(size(x,2)),variance(size(x,2))
    integer :: i,j,k,n,p
    call check_xy(x,size(y)); call require(self%var_smoothing>=0,'negative variance smoothing')
    self%classes_=unique_labels(y); k=size(self%classes_); n=size(y); p=size(x,2)
    if(allocated(self%theta_)) deallocate(self%theta_,self%var_,self%class_prior_,self%class_count_)
    allocate(self%theta_(k,p),self%var_(k,p),self%class_prior_(k),self%class_count_(k))
    mean=sum(x,dim=1)/n
    do j=1,p
      variance(j)=sum((x(:,j)-mean(j))**2)/n
    end do
    self%epsilon_=self%var_smoothing*maxval(variance)
    self%theta_=0; self%var_=0
    do i=1,k
      self%class_count_(i)=count(y==self%classes_(i))
      do j=1,n
        if(y(j)==self%classes_(i)) self%theta_(i,:)=self%theta_(i,:)+x(j,:)
      end do
      self%theta_(i,:)=self%theta_(i,:)/self%class_count_(i)
      do j=1,n
        if(y(j)==self%classes_(i)) self%var_(i,:)=self%var_(i,:)+(x(j,:)-self%theta_(i,:))**2
      end do
      self%var_(i,:)=self%var_(i,:)/self%class_count_(i)+self%epsilon_
    end do
    call require(all(self%var_>0),'Gaussian NB has zero variance; increase smoothing or remove constant data')
    self%class_prior_=self%class_count_/n
  end subroutine
  function gaussian_log_proba(self,x) result(logp)
    class(gaussian_nb), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: logp(:,:)
    integer :: i,c
    call require(allocated(self%theta_),'gaussian_nb is not fitted')
    call check_xy(transpose(x),size(self%theta_,2)); allocate(logp(size(x,1),size(self%classes_)))
    do i=1,size(x,1)
      do c=1,size(self%classes_)
        logp(i,c)=log(self%class_prior_(c))-0.5_dp*sum(log(2*acos(-1._dp)*self%var_(c,:)) &
          +(x(i,:)-self%theta_(c,:))**2/self%var_(c,:))
      end do
      logp(i,:)=logp(i,:)-logsumexp(logp(i,:))
    end do
  end function
  function gaussian_proba(self,x) result(p)
    class(gaussian_nb), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: p(:,:)
    p=exp(self%predict_log_proba(x))
  end function
  function gaussian_predict(self,x) result(y)
    class(gaussian_nb), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: y(size(x,1)),i
    real(dp), allocatable :: p(:,:)
    p=self%predict_log_proba(x)
    do i=1,size(y)
      y(i)=self%classes_(maxloc(p(i,:),dim=1))
    end do
  end function
  real(dp) function gaussian_score(self,x,y) result(v)
    class(gaussian_nb), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    v=accuracy_score(y,self%predict(x))
  end function
  subroutine discrete_counts(self,x,y,counts)
    class(multinomial_nb), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), allocatable, intent(out) :: counts(:,:)
    integer :: i,c,k
    call check_xy(x,size(y)); call require(all(x>=0),'discrete NB requires nonnegative data')
    call require(self%alpha>0,'NB alpha must be strictly positive')
    self%classes_=unique_labels(y); k=size(self%classes_)
    allocate(counts(k,size(x,2))); counts=0
    self%class_count_=spread(0._dp,1,k)
    do i=1,size(y)
      c=label_index(self%classes_,y(i)); counts(c,:)=counts(c,:)+x(i,:)
      self%class_count_(c)=self%class_count_(c)+1
    end do
    self%class_log_prior_=log(self%class_count_/size(y))
    if(.not.self%fit_prior) self%class_log_prior_=-log(real(k,dp))
  end subroutine
  subroutine multinomial_fit(self,x,y)
    class(multinomial_nb), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), allocatable :: counts(:,:)
    call discrete_counts(self,x,y,counts); counts=counts+self%alpha
    self%feature_log_prob_=log(counts/spread(sum(counts,dim=2),2,size(x,2)))
  end subroutine
  subroutine bernoulli_fit(self,x,y)
    class(bernoulli_nb), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), allocatable :: counts(:,:)
    call discrete_counts(self,merge(1._dp,0._dp,x>self%binarize_threshold),y,counts)
    self%feature_log_prob_=log((counts+self%alpha)/ &
      spread(self%class_count_+2*self%alpha,2,size(x,2)))
  end subroutine
  subroutine complement_fit(self,x,y)
    class(complement_nb), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), allocatable :: counts(:,:)
    call discrete_counts(self,x,y,counts)
    counts=spread(sum(counts,dim=1),1,size(counts,1))-counts+self%alpha
    self%feature_log_prob_=-log(counts/spread(sum(counts,dim=2),2,size(x,2)))
    if(self%norm) self%feature_log_prob_=self%feature_log_prob_/ &
      spread(sum(self%feature_log_prob_,dim=2),2,size(x,2))
  end subroutine
  function multinomial_joint(self,x) result(joint)
    class(multinomial_nb), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: joint(:,:)
    call require(allocated(self%feature_log_prob_),'NB is not fitted')
    call check_xy(transpose(x),size(self%feature_log_prob_,2))
    call require(all(x>=0),'multinomial NB requires nonnegative data')
    joint=matmul(x,transpose(self%feature_log_prob_))+spread(self%class_log_prior_,1,size(x,1))
  end function
  function bernoulli_joint(self,x) result(joint)
    class(bernoulli_nb), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: joint(:,:),negative(:,:),binary(:,:)
    call require(allocated(self%feature_log_prob_),'NB is not fitted')
    call check_xy(transpose(x),size(self%feature_log_prob_,2))
    binary=merge(1._dp,0._dp,x>self%binarize_threshold)
    negative=log(1-exp(self%feature_log_prob_))
    joint=matmul(binary,transpose(self%feature_log_prob_-negative))+ &
      spread(sum(negative,dim=2)+self%class_log_prior_,1,size(x,1))
  end function
  function complement_joint(self,x) result(joint)
    class(complement_nb), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: joint(:,:)
    call require(allocated(self%feature_log_prob_),'NB is not fitted')
    call check_xy(transpose(x),size(self%feature_log_prob_,2))
    call require(all(x>=0),'complement NB requires nonnegative data')
    joint=matmul(x,transpose(self%feature_log_prob_))
    if(size(self%classes_)==1) joint=joint+spread(self%class_log_prior_,1,size(x,1))
  end function
  function discrete_log_proba(self,x) result(logp)
    class(multinomial_nb), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: logp(:,:)
    integer :: i
    logp=self%joint_log_likelihood(x)
    do i=1,size(x,1)
      logp(i,:)=logp(i,:)-logsumexp(logp(i,:))
    end do
  end function
  function discrete_proba(self,x) result(p)
    class(multinomial_nb), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: p(:,:)
    p=exp(self%predict_log_proba(x))
  end function
  function discrete_predict(self,x) result(y)
    class(multinomial_nb), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: y(size(x,1)),i
    real(dp), allocatable :: p(:,:)
    p=self%joint_log_likelihood(x)
    do i=1,size(y)
      y(i)=self%classes_(maxloc(p(i,:),dim=1))
    end do
  end function
  real(dp) function discrete_score(self,x,y) result(v)
    class(multinomial_nb), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    v=accuracy_score(y,self%predict(x))
  end function
end module
