! CategoricalNB contracts: sklearn 1.6.1 naive_bayes.py.
! BSD-3-Clause attribution retained in LICENSES/scikit-learn-COPYING.
module sklearn_categorical_nb
  use sklearn_common, only: dp,require,check_x,check_xy,unique_labels,label_index,logsumexp
  use sklearn_metrics, only: accuracy_score
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value,ieee_negative_inf,ieee_quiet_nan
  implicit none
  private
  public :: categorical_nb,categorical_table
  ! Each feature has its own (class,category) table; category zero is column one.
  type :: categorical_table
    real(dp), allocatable :: values(:,:)
  end type
  type :: categorical_nb
    real(dp) :: alpha=1
    logical :: fit_prior=.true.,force_alpha=.true.
    integer :: min_categories=0,n_features_in_=0
    integer, allocatable :: min_categories_per_feature(:),classes_(:),n_categories_(:)
    real(dp), allocatable :: class_prior(:),class_count_(:),class_log_prior_(:)
    type(categorical_table), allocatable :: category_count_(:),feature_log_prob_(:)
  contains
    procedure :: fit=>categorical_fit
    procedure :: partial_fit=>categorical_partial_fit
    procedure :: joint_log_likelihood=>categorical_joint
    procedure :: predict_log_proba=>categorical_log_proba
    procedure :: predict_proba=>categorical_proba
    procedure :: predict=>categorical_predict
    procedure :: score=>categorical_score
  end type
contains
  subroutine categorical_check(x)
    real(dp), intent(in) :: x(:,:)
    call check_x(x)
    call require(all(x>=0),'CategoricalNB categories must be nonnegative')
    call require(all(x<=real(huge(0)-1,dp)),'CategoricalNB category index overflow')
    call require(all(x==aint(x)),'CategoricalNB requires integer-valued categories')
  end subroutine

  subroutine categorical_init(self,p,classes)
    class(categorical_nb), intent(inout) :: self
    integer, intent(in) :: p,classes(:)
    integer :: j
    call require(size(classes)>0,'CategoricalNB classes must not be empty')
    self%classes_=unique_labels(classes); self%n_features_in_=p
    self%class_count_=spread(0._dp,1,size(self%classes_)); self%n_categories_=spread(0,1,p)
    if(allocated(self%category_count_)) deallocate(self%category_count_,self%feature_log_prob_)
    allocate(self%category_count_(p),self%feature_log_prob_(p))
    do j=1,p
      allocate(self%category_count_(j)%values(size(self%classes_),0))
    end do
  end subroutine

  subroutine categorical_fit(self,x,y,sample_weight)
    class(categorical_nb), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    call check_xy(x,size(y)); call categorical_check(x)
    call categorical_init(self,size(x,2),y)
    call categorical_update(self,x,y,sample_weight)
  end subroutine

  subroutine categorical_partial_fit(self,x,y,classes,sample_weight)
    class(categorical_nb), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    integer, intent(in), optional :: classes(:)
    real(dp), intent(in), optional :: sample_weight(:)
    integer, allocatable :: sorted(:)
    call check_xy(x,size(y)); call categorical_check(x)
    if(.not.allocated(self%classes_)) then
      call require(present(classes),'CategoricalNB first partial_fit requires classes')
      call categorical_init(self,size(x,2),classes)
    else if(present(classes)) then
      call require(size(classes)>0,'CategoricalNB classes must not be empty')
      sorted=unique_labels(classes)
      call require(size(sorted)==size(self%classes_),'CategoricalNB partial_fit classes changed')
      call require(all(sorted==self%classes_),'CategoricalNB partial_fit classes changed')
    end if
    call categorical_update(self,x,y,sample_weight)
  end subroutine

  subroutine categorical_update(self,x,y,sample_weight)
    class(categorical_nb), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: w(:),expanded(:,:),smooth(:,:)
    real(dp) :: smoothing,total,denominator
    integer, allocatable :: encoded(:)
    integer :: n,p,c,j,i,k,width,oldwidth
    call require(size(x,2)==self%n_features_in_,'CategoricalNB feature count mismatch')
    call require(ieee_is_finite(self%alpha),'nonfinite CategoricalNB alpha')
    call require(self%alpha>=0,'CategoricalNB alpha must be nonnegative')
    smoothing=self%alpha
    if(.not.self%force_alpha) smoothing=max(smoothing,1.e-10_dp)
    call require(self%min_categories>=0,'CategoricalNB min_categories must be nonnegative')
    n=size(y); p=size(x,2); c=size(self%classes_)
    if(allocated(self%min_categories_per_feature)) then
      call require(size(self%min_categories_per_feature)==p,'CategoricalNB min_categories shape mismatch')
      call require(all(self%min_categories_per_feature>0),'CategoricalNB per-feature minima must be positive')
    end if
    encoded=[(label_index(self%classes_,y(i)),i=1,n)]
    call require(all(encoded>0),'CategoricalNB target not in classes')
    w=spread(1._dp,1,n)
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'CategoricalNB sample weight count mismatch')
      call require(all(ieee_is_finite(sample_weight)),'nonfinite CategoricalNB sample weights')
      call require(all(sample_weight>=0),'negative CategoricalNB sample weights')
      w=sample_weight
    end if
    call require(maxval(w)<=huge(1._dp)/(4*real(n,dp)),'CategoricalNB weight mass exceeds numeric range')
    total=sum(self%class_count_)
    call require(sum(w)<=huge(1._dp)/4-total,'CategoricalNB cumulative weight exceeds numeric range')
    do i=1,n
      k=encoded(i); self%class_count_(k)=self%class_count_(k)+w(i)
    end do
    do j=1,p
      oldwidth=size(self%category_count_(j)%values,2)
      width=max(oldwidth,int(maxval(x(:,j)))+1,self%min_categories)
      if(allocated(self%min_categories_per_feature)) width=max(width,self%min_categories_per_feature(j))
      if(width>oldwidth) then
        allocate(expanded(c,width)); expanded=0
        expanded(:,:oldwidth)=self%category_count_(j)%values
        call move_alloc(expanded,self%category_count_(j)%values)
      end if
      self%n_categories_(j)=max(int(maxval(x(:,j)))+1,self%min_categories)
      if(allocated(self%min_categories_per_feature)) &
        self%n_categories_(j)=max(self%n_categories_(j),self%min_categories_per_feature(j))
      do i=1,n
        k=int(x(i,j))+1
        self%category_count_(j)%values(encoded(i),k)=self%category_count_(j)%values(encoded(i),k)+w(i)
      end do
      call require(smoothing<=huge(1._dp)/(4*real(width,dp)), &
        'CategoricalNB smoothing exceeds numeric range')
      smooth=self%category_count_(j)%values+smoothing
      if(allocated(self%feature_log_prob_(j)%values)) deallocate(self%feature_log_prob_(j)%values)
      allocate(self%feature_log_prob_(j)%values(c,width))
      do k=1,c
        denominator=sum(smooth(k,:))
        if(denominator==0) then
          self%feature_log_prob_(j)%values(k,:)=ieee_value(0._dp,ieee_quiet_nan)
        else
          self%feature_log_prob_(j)%values(k,:)=categorical_log(smooth(k,:))-log(denominator)
        end if
      end do
    end do
    if(allocated(self%class_prior)) then
      call require(size(self%class_prior)==c,'CategoricalNB class prior count mismatch')
      call require(all(ieee_is_finite(self%class_prior)),'nonfinite CategoricalNB class prior')
      call require(all(self%class_prior>=0),'negative CategoricalNB class prior')
      self%class_log_prior_=categorical_log(self%class_prior)
    else if(self%fit_prior) then
      total=sum(self%class_count_)
      if(total==0) then
        self%class_log_prior_=spread(ieee_value(0._dp,ieee_quiet_nan),1,c)
      else
        self%class_log_prior_=categorical_log(self%class_count_)-log(total)
      end if
    else
      self%class_log_prior_=spread(-log(real(c,dp)),1,c)
    end if
  end subroutine

  elemental real(dp) function categorical_log(value) result(logged)
    real(dp), intent(in) :: value
    if(value==0) then
      logged=ieee_value(0._dp,ieee_negative_inf)
    else
      logged=log(value)
    end if
  end function

  function categorical_joint(self,x) result(scores)
    class(categorical_nb), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: scores(:,:)
    integer :: i,j
    call require(allocated(self%class_count_),'CategoricalNB is not fitted')
    call categorical_check(x)
    call require(size(x,2)==self%n_features_in_,'CategoricalNB feature count mismatch')
    allocate(scores(size(x,1),size(self%classes_))); scores=0
    do j=1,size(x,2)
      call require(all(x(:,j)<size(self%category_count_(j)%values,2)),'CategoricalNB unseen category exceeds fitted range')
      do i=1,size(x,1)
        scores(i,:)=scores(i,:)+self%feature_log_prob_(j)%values(:,int(x(i,j))+1)
      end do
    end do
    scores=scores+spread(self%class_log_prior_,1,size(x,1))
  end function

  function categorical_log_proba(self,x) result(scores)
    class(categorical_nb), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: scores(:,:)
    integer :: i
    scores=self%joint_log_likelihood(x)
    do i=1,size(x,1)
      if(any(ieee_is_nan(scores(i,:)))) then
        scores(i,:)=ieee_value(0._dp,ieee_quiet_nan)
      else if(maxval(scores(i,:))==ieee_value(0._dp,ieee_negative_inf)) then
        scores(i,:)=ieee_value(0._dp,ieee_quiet_nan)
      else
        scores(i,:)=scores(i,:)-logsumexp(scores(i,:))
      end if
    end do
  end function

  function categorical_proba(self,x) result(scores)
    class(categorical_nb), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: scores(:,:)
    scores=exp(self%predict_log_proba(x))
  end function

  function categorical_predict(self,x) result(y)
    class(categorical_nb), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: y(:)
    integer :: i,k,index
    real(dp), allocatable :: scores(:,:)
    scores=self%joint_log_likelihood(x); allocate(y(size(x,1)))
    do i=1,size(y)
      index=0
      do k=1,size(self%classes_)
        if(ieee_is_nan(scores(i,k))) then
          index=k; exit
        end if
      end do
      if(index==0) index=maxloc(scores(i,:),dim=1)
      y(i)=self%classes_(index)
    end do
  end function

  real(dp) function categorical_score(self,x,y) result(score)
    class(categorical_nb), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    score=accuracy_score(y,self%predict(x))
  end function
end module
