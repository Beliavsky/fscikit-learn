! Gaussian streaming/weighted updates follow sklearn 1.6.1 naive_bayes.py.
! BSD attribution: THIRD_PARTY_NOTICES.md and LICENSES/scikit-learn-COPYING.
module sklearn_naive_bayes
  use sklearn_categorical_nb, only: categorical_nb,categorical_table
  use sklearn_common
  use sklearn_metrics, only: accuracy_score
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value,ieee_quiet_nan,ieee_negative_inf,ieee_positive_inf
  implicit none
  private
  public :: categorical_nb,categorical_table
  public :: gaussian_nb, multinomial_nb, bernoulli_nb, complement_nb
  type :: gaussian_nb
    real(dp) :: var_smoothing=1.e-9_dp,epsilon_=0
    integer, allocatable :: classes_(:)
    real(dp), allocatable :: theta_(:,:),var_(:,:),class_prior_(:),class_count_(:)
    integer :: n_features_in_=0
    real(dp), allocatable :: priors(:)
  contains
    procedure :: fit=>gaussian_fit
    procedure :: partial_fit=>gaussian_partial_fit
    procedure :: predict_log_proba=>gaussian_log_proba
    procedure :: predict_proba=>gaussian_proba
    procedure :: predict=>gaussian_predict
    procedure :: score=>gaussian_score
  end type
  type :: multinomial_nb
    real(dp) :: alpha=1
    logical :: fit_prior=.true.,force_alpha=.true.
    real(dp), allocatable :: alpha_per_feature(:)
    integer :: n_features_in_=0
    real(dp), allocatable :: class_prior(:),feature_count_(:,:)
    integer, allocatable :: classes_(:)
    real(dp), allocatable :: feature_log_prob_(:,:),class_log_prior_(:),class_count_(:)
  contains
    procedure :: fit=>multinomial_fit
    procedure :: partial_fit=>discrete_partial_fit
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
    real(dp), allocatable :: feature_all_(:)
  contains
    procedure :: fit=>complement_fit
    procedure :: joint_log_likelihood=>complement_joint
  end type
contains
  subroutine gaussian_fit(self,x,y,sample_weight)
    class(gaussian_nb), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    call check_xy(x,size(y))
    if(allocated(self%classes_)) deallocate(self%classes_)
    if(allocated(self%theta_)) deallocate(self%theta_,self%var_,self%class_prior_,self%class_count_)
    call self%partial_fit(x,y,unique_labels(y),sample_weight)
  end subroutine

  subroutine gaussian_partial_fit(self,x,y,classes,sample_weight)
    class(gaussian_nb), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    integer, intent(in), optional :: classes(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp) :: weights(size(y)),mean(size(x,2)),variance(size(x,2)),new_mean(size(x,2)),new_var(size(x,2))
    real(dp) :: delta(size(x,2)),mass,old,total,ratio,overall,limit,max_feature
    integer, allocatable :: labels(:),rows(:)
    integer :: i,j,k,n,p
    logical :: first
    call check_xy(x,size(y)); n=size(y); p=size(x,2)
    call require(ieee_is_finite(self%var_smoothing),'Gaussian NB smoothing must be finite')
    call require(self%var_smoothing>=0,'negative variance smoothing')
    weights=1; limit=sqrt(huge(1._dp))/16
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'Gaussian NB sample weight length mismatch')
      call require(all(ieee_is_finite(sample_weight)),'Gaussian NB sample weights must be finite')
      call require(all(sample_weight>=0),'Gaussian NB sample weights must be nonnegative')
      call require(maxval(sample_weight)<=limit/n,'Gaussian NB weight mass exceeds working range')
      weights=sample_weight
    end if
    first=.not.allocated(self%classes_)
    if(first) then
      call require(present(classes),'first Gaussian NB partial_fit requires classes')
      call require(size(classes)>0,'Gaussian NB classes must not be empty')
      self%classes_=unique_labels(classes); k=size(self%classes_); self%n_features_in_=p
      if(allocated(self%theta_)) deallocate(self%theta_,self%var_,self%class_prior_,self%class_count_)
      allocate(self%theta_(k,p),self%var_(k,p),self%class_prior_(k),self%class_count_(k))
      self%theta_=0; self%var_=0; self%class_count_=0; self%class_prior_=0
      if(allocated(self%priors)) then
        call require(size(self%priors)==k,'Gaussian NB prior length mismatch')
        call require(all(ieee_is_finite(self%priors)),'Gaussian NB priors must be finite')
        call require(all(self%priors>=0),'Gaussian NB priors must be nonnegative')
        call require(maxval(self%priors)<=2,'Gaussian NB priors must sum to one')
        call require(abs(sum(self%priors)-1)<=1.e-8_dp+1.e-5_dp,'Gaussian NB priors must sum to one')
        self%class_prior_=self%priors
      end if
    else
      call require(p==self%n_features_in_,'Gaussian NB partial_fit feature count mismatch')
      if(present(classes)) then
        call require(size(classes)>0,'Gaussian NB classes must not be empty')
        labels=unique_labels(classes)
        call require(size(labels)==size(self%classes_),'Gaussian NB partial_fit classes changed')
        call require(all(labels==self%classes_),'Gaussian NB partial_fit classes changed')
      end if
    end if
    do i=1,n
      call require(any(self%classes_==y(i)),'Gaussian NB target is outside declared classes')
    end do
    overall=sum(self%class_count_)
    call require(sum(weights)<=limit-overall,'Gaussian NB cumulative weight mass exceeds working range')
    max_feature=maxval(abs(x))
    call require(max_feature<=huge(1._dp)/(16*real(n,dp)),'Gaussian NB feature mass exceeds working range')
    if(max_feature>1) call require(max(1._dp,sum(weights))<=huge(1._dp)/16/max_feature, &
      'Gaussian NB weighted feature mass exceeds working range')
    mean=sum(x,dim=1)/n
    do j=1,p
      call require(maxval(abs(x(:,j)-mean(j)))<=sqrt(huge(1._dp)/(16*real(n,dp))), &
        'Gaussian NB variance exceeds working range')
      variance(j)=sum((x(:,j)-mean(j))**2)/n
    end do
    ! sklearn uses unweighted batch variance for smoothing, even for weighted updates.
    mass=maxval(variance)
    if(mass>1._dp/16) call require(self%var_smoothing<=huge(1._dp)/16/mass, &
      'Gaussian NB smoothing exceeds working range')
    self%epsilon_=self%var_smoothing*mass
    ! Preserve the pinned source's subtraction of the CURRENT batch epsilon.
    if(.not.first) self%var_=self%var_-self%epsilon_
    do i=1,size(self%classes_)
      rows=pack([(j,j=1,n)],y==self%classes_(i)); if(size(rows)==0) cycle
      mass=sum(weights(rows)); old=self%class_count_(i)
      ! Source np.isclose(n_new,0) skips moment updates, but still counts tiny weights.
      if(.not.present(sample_weight).or.mass>1.e-8_dp) then
        new_mean=matmul(weights(rows),x(rows,:))/mass
        do j=1,p
          call require(maxval(abs(x(rows,j)-new_mean(j)))<=sqrt(huge(1._dp)/(4*real(n,dp))), &
            'Gaussian NB variance exceeds working range')
          new_var(j)=sum((weights(rows)/mass)*(x(rows,j)-new_mean(j))**2)
        end do
        if(old==0) then
          self%theta_(i,:)=new_mean; self%var_(i,:)=new_var
        else
          total=old+mass; ratio=mass/total; delta=self%theta_(i,:)-new_mean
          call require(maxval(abs(delta))<=sqrt(huge(1._dp)/16),'Gaussian NB mean separation exceeds working range')
          self%theta_(i,:)=(old/total)*self%theta_(i,:)+ratio*new_mean
          self%var_(i,:)=(old/total)*self%var_(i,:)+ratio*new_var+(old/total)*ratio*delta**2
        end if
      end if
      self%class_count_(i)=old+mass
    end do
    self%var_=self%var_+self%epsilon_
    call require(all(ieee_is_finite(self%var_)),'Gaussian NB variance exceeds working range')
    call require(maxval(abs(self%var_))<=huge(1._dp)/8,'Gaussian NB variance exceeds working range')
    if(.not.allocated(self%priors)) then
      overall=sum(self%class_count_)
      if(overall==0) then
        self%class_prior_=ieee_value(0._dp,ieee_quiet_nan)
      else
        self%class_prior_=self%class_count_/overall
      end if
    end if
  end subroutine

  function gaussian_joint(self,x) result(joint)
    class(gaussian_nb), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: joint(:,:)
    real(dp) :: normalizer,distance,term,difference,variance
    integer :: i,c,j
    logical :: undefined
    call require(allocated(self%theta_),'gaussian_nb is not fitted')
    call check_xy(transpose(x),size(self%theta_,2)); allocate(joint(size(x,1),size(self%classes_)))
    do c=1,size(self%classes_)
      undefined=.not.ieee_is_finite(self%class_prior_(c))
      if(any(self%var_(c,:)<=0)) undefined=.true.
      if(undefined) then
        joint(:,c)=ieee_value(0._dp,ieee_quiet_nan); cycle
      end if
      if(self%class_prior_(c)==0) then
        joint(:,c)=ieee_value(0._dp,ieee_negative_inf); cycle
      end if
      normalizer=-.5_dp*sum(log(2*acos(-1._dp)*self%var_(c,:)))
      do i=1,size(x,1)
        distance=0
        do j=1,size(x,2)
          variance=self%var_(c,j)
          ! Avoid overflow while retaining the source's infinite tail likelihood.
          if(abs(x(i,j))/2+abs(self%theta_(c,j))/2>huge(1._dp)/2) then
            distance=ieee_value(0._dp,ieee_negative_inf); exit
          end if
          difference=x(i,j)-self%theta_(c,j)
          if(abs(difference)>sqrt(huge(1._dp))*sqrt(min(1._dp,variance))) then
            distance=ieee_value(0._dp,ieee_negative_inf); exit
          end if
          term=difference**2/variance
          if(term>huge(1._dp)-distance) then
            distance=ieee_value(0._dp,ieee_negative_inf); exit
          end if
          distance=distance+term
        end do
        if(distance<0) then
          joint(i,c)=distance
        else
          joint(i,c)=log(self%class_prior_(c))+(normalizer-.5_dp*distance)
        end if
      end do
    end do
  end function

  function gaussian_log_proba(self,x) result(logp)
    class(gaussian_nb), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: logp(:,:)
    integer :: i
    logp=gaussian_joint(self,x)
    do i=1,size(x,1)
      if(any(ieee_is_nan(logp(i,:)))) then
        logp(i,:)=ieee_value(0._dp,ieee_quiet_nan)
      else if(maxval(logp(i,:))==ieee_value(0._dp,ieee_negative_inf)) then
        logp(i,:)=ieee_value(0._dp,ieee_quiet_nan)
      else
        logp(i,:)=logp(i,:)-logsumexp(logp(i,:))
      end if
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
    integer :: y(size(x,1)),i,c,index
    real(dp), allocatable :: p(:,:)
    p=gaussian_joint(self,x)
    do i=1,size(y)
      index=0
      do c=1,size(self%classes_)
        if(ieee_is_nan(p(i,c))) then
          index=c; exit
        end if
      end do
      if(index==0) index=maxloc(p(i,:),dim=1)
      y(i)=self%classes_(index)
    end do
  end function
  real(dp) function gaussian_score(self,x,y) result(v)
    class(gaussian_nb), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    v=accuracy_score(y,self%predict(x))
  end function
  function discrete_alpha(self,p) result(alpha)
    class(multinomial_nb), intent(in) :: self
    integer, intent(in) :: p
    real(dp) :: alpha(p)
    if(allocated(self%alpha_per_feature)) then
      select type(self)
      class is(bernoulli_nb)
        call require(.false.,'Bernoulli NB requires scalar alpha')
      end select
      call require(size(self%alpha_per_feature)==p,'NB alpha_per_feature length mismatch')
      call require(all(ieee_is_finite(self%alpha_per_feature)),'NB alpha must be finite')
      call require(all(self%alpha_per_feature>=0),'NB alpha must be nonnegative')
      alpha=self%alpha_per_feature
    else
      call require(ieee_is_finite(self%alpha),'NB alpha must be finite')
      call require(self%alpha>=0,'NB alpha must be nonnegative')
      alpha=self%alpha
    end if
    if(.not.self%force_alpha) alpha=max(alpha,1.e-10_dp)
  end function
  subroutine discrete_controls(self,x)
    class(multinomial_nb), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: alpha(size(x,2))
    call check_x(x);alpha=discrete_alpha(self,size(x,2))
    select type(self)
    class is(bernoulli_nb)
      call require(ieee_is_finite(self%binarize_threshold),'Bernoulli NB threshold must be finite')
      call require(self%binarize_threshold>=0,'Bernoulli NB threshold must be nonnegative')
    class default
      call require(all(x>=0),'discrete NB requires nonnegative data')
    end select
  end subroutine
  elemental real(dp) function discrete_log(value) result(logged)
    real(dp), intent(in) :: value
    if(value==0) then
      logged=ieee_value(0._dp,ieee_negative_inf)
    else
      logged=log(value)
    end if
  end function
  subroutine discrete_update(self)
    class(multinomial_nb), intent(inout) :: self
    real(dp), allocatable :: counts(:,:),alpha(:)
    real(dp) :: total,denominator
    integer :: c,k,p,j
    k=size(self%classes_);p=self%n_features_in_;alpha=discrete_alpha(self,p)
    call require(all(self%feature_count_<=spread(huge(1._dp)-alpha,1,k)), &
      'NB smoothed count exceeds numeric range')
    select type(self)
    class is(bernoulli_nb)
      call require(alpha(1)<=huge(1._dp)/2,'NB smoothed count exceeds numeric range')
      call require(all(self%class_count_<=huge(1._dp)-2*alpha(1)),'NB smoothed count exceeds numeric range')
      counts=self%feature_count_+spread(alpha,1,k)
      if(allocated(self%feature_log_prob_)) deallocate(self%feature_log_prob_)
      allocate(self%feature_log_prob_(k,p))
      do c=1,k
        denominator=self%class_count_(c)+2*alpha(1)
        if(denominator==0) then
          self%feature_log_prob_(c,:)=ieee_value(0._dp,ieee_quiet_nan)
        else
          self%feature_log_prob_(c,:)=discrete_log(counts(c,:))-log(denominator)
        end if
      end do
    class is(complement_nb)
      call require(maxval(self%feature_count_)<=huge(1._dp)/k,'NB feature total exceeds numeric range')
      self%feature_all_=sum(self%feature_count_,dim=1)
      call require(all(self%feature_all_<=huge(1._dp)-alpha),'NB smoothed count exceeds numeric range')
      ! Match source addition before subtraction, including lost tiny smoothing.
      counts=spread(self%feature_all_+alpha,1,k)-self%feature_count_
      call require(maxval(counts)<=huge(1._dp)/p,'NB smoothed total exceeds numeric range')
      if(allocated(self%feature_log_prob_)) deallocate(self%feature_log_prob_)
      allocate(self%feature_log_prob_(k,p))
      do c=1,k
        denominator=sum(counts(c,:))
        if(denominator==0) then
          self%feature_log_prob_(c,:)=ieee_value(0._dp,ieee_quiet_nan)
        else
          self%feature_log_prob_(c,:)=-discrete_log(counts(c,:)/denominator)
        end if
        if(.not.self%norm) cycle
        if(any(ieee_is_nan(self%feature_log_prob_(c,:)))) cycle
        if(any(.not.ieee_is_finite(self%feature_log_prob_(c,:)))) then
          do j=1,p
            if(ieee_is_finite(self%feature_log_prob_(c,j))) then
              self%feature_log_prob_(c,j)=0
            else
              self%feature_log_prob_(c,j)=ieee_value(0._dp,ieee_quiet_nan)
            end if
          end do
        else
          denominator=sum(self%feature_log_prob_(c,:))
          if(denominator==0) then
            self%feature_log_prob_(c,:)=ieee_value(0._dp,ieee_quiet_nan)
          else
            self%feature_log_prob_(c,:)=self%feature_log_prob_(c,:)/denominator
          end if
        end if
      end do
    class default
      counts=self%feature_count_+spread(alpha,1,k)
      call require(maxval(counts)<=huge(1._dp)/p,'NB smoothed total exceeds numeric range')
      if(allocated(self%feature_log_prob_)) deallocate(self%feature_log_prob_)
      allocate(self%feature_log_prob_(k,p))
      do c=1,k
        denominator=sum(counts(c,:))
        if(denominator==0) then
          self%feature_log_prob_(c,:)=ieee_value(0._dp,ieee_quiet_nan)
        else
          self%feature_log_prob_(c,:)=discrete_log(counts(c,:))-log(denominator)
        end if
      end do
    end select
    if(allocated(self%class_prior)) then
      call require(size(self%class_prior)==k,'NB class_prior must match class count')
      call require(all(ieee_is_finite(self%class_prior)),'NB class_prior must be finite')
      call require(all(self%class_prior>=0),'NB class_prior must be nonnegative')
      self%class_log_prior_=discrete_log(self%class_prior)
    else if(.not.self%fit_prior) then
      self%class_log_prior_=spread(-log(real(k,dp)),1,k)
    else
      call require(maxval(self%class_count_)<=huge(1._dp)/k,'NB class total exceeds numeric range')
      total=sum(self%class_count_)
      if(total==0) then
        self%class_log_prior_=spread(ieee_value(0._dp,ieee_quiet_nan),1,k)
      else
        self%class_log_prior_=discrete_log(self%class_count_)-log(total)
      end if
    end if
  end subroutine
  subroutine discrete_partial_fit(self,x,y,classes,sample_weight)
    class(multinomial_nb), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    integer, optional, intent(in) :: classes(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer, allocatable :: labels(:)
    real(dp), allocatable :: data(:,:),weights(:),contribution(:)
    integer :: i,c,k,p
    call check_xy(x,size(y));call discrete_controls(self,x);p=size(x,2)
    if(.not.allocated(self%classes_)) then
      call require(present(classes),'NB first partial_fit requires classes')
      self%classes_=unique_labels(classes);k=size(self%classes_);self%n_features_in_=p
      allocate(self%feature_count_(k,p),self%class_count_(k));self%feature_count_=0;self%class_count_=0
    else
      if(present(classes)) then
        labels=unique_labels(classes)
        call require(size(labels)==size(self%classes_),'NB classes changed between partial_fit calls')
        call require(all(labels==self%classes_),'NB classes changed between partial_fit calls')
      end if
      call require(p==self%n_features_in_,'NB feature count changed between partial_fit calls')
    end if
    weights=spread(1._dp,1,size(y))
    if(present(sample_weight)) then
      call require(size(sample_weight)==size(y),'NB sample weight length mismatch')
      call require(all(ieee_is_finite(sample_weight)),'NB sample weights must be finite')
      call require(all(sample_weight>=0),'NB sample weights must be nonnegative')
      weights=sample_weight
    end if
    data=x
    select type(self)
    class is(bernoulli_nb)
      data=merge(1._dp,0._dp,x>self%binarize_threshold)
    end select
    do i=1,size(y)
      c=label_index(self%classes_,y(i));call require(c>0,'NB label is absent from declared classes')
      if(weights(i)>1) call require(all(data(i,:)<=huge(1._dp)/weights(i)),'NB weighted count exceeds numeric range')
      contribution=weights(i)*data(i,:)
      call require(all(self%feature_count_(c,:)<=huge(1._dp)-contribution),'NB accumulated count exceeds numeric range')
      call require(self%class_count_(c)<=huge(1._dp)-weights(i),'NB accumulated class count exceeds numeric range')
      self%feature_count_(c,:)=self%feature_count_(c,:)+contribution
      self%class_count_(c)=self%class_count_(c)+weights(i)
    end do
    call discrete_update(self)
  end subroutine
  subroutine multinomial_fit(self,x,y,sample_weight)
    class(multinomial_nb), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    if(allocated(self%classes_)) deallocate(self%classes_)
    if(allocated(self%feature_count_)) deallocate(self%feature_count_)
    if(allocated(self%class_count_)) deallocate(self%class_count_)
    if(allocated(self%feature_log_prob_)) deallocate(self%feature_log_prob_)
    if(allocated(self%class_log_prior_)) deallocate(self%class_log_prior_)
    select type(self)
    class is(complement_nb)
      if(allocated(self%feature_all_)) deallocate(self%feature_all_)
    end select
    self%n_features_in_=0
    call self%partial_fit(x,y,unique_labels(y),sample_weight)
  end subroutine
  subroutine bernoulli_fit(self,x,y,sample_weight)
    class(bernoulli_nb), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    call multinomial_fit(self,x,y,sample_weight)
  end subroutine
  subroutine complement_fit(self,x,y,sample_weight)
    class(complement_nb), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    call multinomial_fit(self,x,y,sample_weight)
  end subroutine
  elemental real(dp) function discrete_add(a,b) result(value)
    real(dp), intent(in) :: a,b
    if(ieee_is_nan(a).or.ieee_is_nan(b)) then
      value=ieee_value(0._dp,ieee_quiet_nan)
    else if(.not.ieee_is_finite(a).and..not.ieee_is_finite(b)) then
      if(sign(1._dp,a)/=sign(1._dp,b)) then
        value=ieee_value(0._dp,ieee_quiet_nan)
      else
        value=a
      end if
    else
      value=a+b
    end if
  end function
  function discrete_dot(x,weights) result(joint)
    real(dp), intent(in) :: x(:,:),weights(:,:)
    real(dp), allocatable :: joint(:,:)
    real(dp) :: value
    logical :: positive,negative,undefined
    integer :: i,c,j
    if(all(ieee_is_finite(weights))) then
      joint=matmul(x,transpose(weights));return
    end if
    allocate(joint(size(x,1),size(weights,1)))
    do c=1,size(weights,1)
      if(any(ieee_is_nan(weights(c,:)))) then
        joint(:,c)=ieee_value(0._dp,ieee_quiet_nan);cycle
      end if
      do i=1,size(x,1)
        value=0;positive=.false.;negative=.false.;undefined=.false.
        do j=1,size(x,2)
          if(ieee_is_finite(weights(c,j))) then
            value=value+x(i,j)*weights(c,j)
          else if(x(i,j)==0) then
            undefined=.true.
          else if(weights(c,j)>0) then
            positive=.true.
          else
            negative=.true.
          end if
        end do
        if(undefined.or.(positive.and.negative)) then
          joint(i,c)=ieee_value(0._dp,ieee_quiet_nan)
        else if(positive) then
          joint(i,c)=ieee_value(0._dp,ieee_positive_inf)
        else if(negative) then
          joint(i,c)=ieee_value(0._dp,ieee_negative_inf)
        else
          joint(i,c)=value
        end if
      end do
    end do
  end function
  function multinomial_joint(self,x) result(joint)
    class(multinomial_nb), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: joint(:,:)
    call require(allocated(self%feature_log_prob_),'NB is not fitted')
    call check_xy(transpose(x),size(self%feature_log_prob_,2))
    call require(all(x>=0),'multinomial NB requires nonnegative data')
    joint=discrete_add(discrete_dot(x,self%feature_log_prob_),spread(self%class_log_prior_,1,size(x,1)))
  end function
  function bernoulli_joint(self,x) result(joint)
    class(bernoulli_nb), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: joint(:,:),negative(:,:),binary(:,:),weights(:,:),base(:)
    integer :: c
    call require(allocated(self%feature_log_prob_),'NB is not fitted')
    call check_xy(transpose(x),size(self%feature_log_prob_,2))
    binary=merge(1._dp,0._dp,x>self%binarize_threshold)
    allocate(negative(size(self%classes_),size(x,2)),weights(size(self%classes_),size(x,2)))
    base=spread(ieee_value(0._dp,ieee_quiet_nan),1,size(self%classes_));weights=ieee_value(0._dp,ieee_quiet_nan)
    do c=1,size(self%classes_)
      if(any(ieee_is_nan(self%feature_log_prob_(c,:)))) cycle
      negative(c,:)=1-exp(self%feature_log_prob_(c,:))
      ! A unit feature probability makes source's dense likelihood NaN for all rows.
      if(any(negative(c,:)<=0)) cycle
      negative(c,:)=log(negative(c,:))
      weights(c,:)=self%feature_log_prob_(c,:)-negative(c,:)
      base(c)=discrete_add(sum(negative(c,:)),self%class_log_prior_(c))
    end do
    joint=discrete_add(discrete_dot(binary,weights),spread(base,1,size(x,1)))
  end function
  function complement_joint(self,x) result(joint)
    class(complement_nb), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: joint(:,:)
    call require(allocated(self%feature_log_prob_),'NB is not fitted')
    call check_xy(transpose(x),size(self%feature_log_prob_,2))
    call require(all(x>=0),'complement NB requires nonnegative data')
    joint=discrete_dot(x,self%feature_log_prob_)
    if(size(self%classes_)==1) joint=discrete_add(joint,spread(self%class_log_prior_,1,size(x,1)))
  end function
  function discrete_log_proba(self,x) result(logp)
    class(multinomial_nb), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: logp(:,:)
    integer :: i
    logp=self%joint_log_likelihood(x)
    do i=1,size(x,1)
      if(any(ieee_is_nan(logp(i,:)))) then
        logp(i,:)=ieee_value(0._dp,ieee_quiet_nan)
      else if(any(logp(i,:)==ieee_value(0._dp,ieee_positive_inf))) then
        where(logp(i,:)==ieee_value(0._dp,ieee_positive_inf))
          logp(i,:)=ieee_value(0._dp,ieee_quiet_nan)
        elsewhere
          logp(i,:)=ieee_value(0._dp,ieee_negative_inf)
        end where
      else if(all(logp(i,:)==ieee_value(0._dp,ieee_negative_inf))) then
        logp(i,:)=ieee_value(0._dp,ieee_quiet_nan)
      else
        logp(i,:)=logp(i,:)-logsumexp(logp(i,:))
      end if
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
    integer :: y(size(x,1)),i,c,index
    real(dp), allocatable :: p(:,:)
    p=self%joint_log_likelihood(x)
    do i=1,size(y)
      index=0
      do c=1,size(self%classes_)
        if(ieee_is_nan(p(i,c))) then
          index=c;exit
        end if
      end do
      if(index==0) index=maxloc(p(i,:),dim=1)
      y(i)=self%classes_(index)
    end do
  end function
  real(dp) function discrete_score(self,x,y) result(v)
    class(multinomial_nb), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    v=accuracy_score(y,self%predict(x))
  end function
end module
