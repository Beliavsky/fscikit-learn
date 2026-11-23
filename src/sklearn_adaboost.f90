! SAMME contracts: sklearn 1.6.1 ensemble/_weight_boosting.py (BSD).
! Weighted CART and NumPy 2.2.6 reduction association; retained BSD notices.
module sklearn_adaboost
  use sklearn_common, only: dp,require,check_x,check_xy,unique_labels
  use sklearn_tree, only: decision_tree_classifier
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_ieee_math, only: ieee_difference,ieee_product,ieee_quotient,ieee_argmax
  use, intrinsic :: iso_fortran_env, only: real32
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value,ieee_quiet_nan, &
    ieee_positive_inf,ieee_negative_inf
  implicit none
  private
  public :: adaboost_classifier
  type :: adaboost_classifier
    integer :: n_estimators=50,n_estimators_=0,n_classes_=0,n_features_in_=0
    real(dp) :: learning_rate=1
    integer, allocatable :: classes_(:),stump_feature_(:),stump_left_(:),stump_right_(:)
    real(dp), allocatable :: stump_threshold_(:),estimator_weights_(:),estimator_errors_(:),feature_importances_(:)
    integer :: random_state=0,max_depth=1,min_samples_split=2,min_samples_leaf=1,max_features=0,tree_random_state=0
    character(16) :: criterion='gini',splitter='best'
    real(dp) :: min_impurity_decrease=0
    type(decision_tree_classifier) :: estimator_
    type(decision_tree_classifier), allocatable :: estimators_(:)
  contains
    procedure :: fit=>ada_fit
    procedure :: decision_function=>ada_decision
    procedure :: predict=>ada_predict
    procedure :: predict_proba=>ada_proba
    procedure :: predict_log_proba=>ada_log_proba
    procedure :: score=>ada_score
    procedure :: staged_decision_function=>ada_staged_decision
    procedure :: staged_predict=>ada_staged_predict
    procedure :: staged_predict_proba=>ada_staged_proba
    procedure :: staged_score=>ada_staged_score
  end type
contains
  real(dp) function plus(a,b) result(value)
    real(dp), intent(in) :: a,b
    value=ieee_difference(a,-b)
  end function
  recursive real(dp) function numpy_sum(values) result(total)
    real(dp), intent(in) :: values(:)
    real(dp) :: partial(8)
    integer :: n,i,j,whole,split
    n=size(values);total=0
    if(n==0) return
    if(n<8) then
      total=-0._dp
      do i=1,n
        total=plus(total,values(i))
      end do
    else if(n<=128) then
      partial=values(:8);whole=n-mod(n,8)
      do i=9,whole,8
        do j=1,8
          partial(j)=plus(partial(j),values(i+j-1))
        end do
      end do
      total=plus(plus(plus(partial(1),partial(2)),plus(partial(3),partial(4))), &
        plus(plus(partial(5),partial(6)),plus(partial(7),partial(8))))
      do i=whole+1,n
        total=plus(total,values(i))
      end do
    else
      split=n/2;split=split-mod(split,8)
      total=plus(numpy_sum(values(:split)),numpy_sum(values(split+1:)))
    end if
  end function
  real(dp) function quiet_exp(value) result(answer)
    real(dp), intent(in) :: value
    if(ieee_is_nan(value)) then
      answer=value
    else if(value>log(huge(value))) then
      answer=ieee_value(0._dp,ieee_positive_inf)
    else
      answer=exp(value)
    end if
  end function
  subroutine ada_fit(self,x,y,sample_weight)
    class(adaboost_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    type(legacy_random_state) :: rng
    real(dp), allocatable :: data(:,:),w(:)
    integer, allocatable :: prediction(:)
    logical, allocatable :: positive(:),incorrect(:)
    real(dp) :: error,weight,total,delta,log_weight
    integer :: n,p,k,i,t,left,right
    call check_xy(x,size(y));n=size(y);p=size(x,2)
    call require(self%n_estimators>0,'AdaBoost n_estimators must be positive')
    call require(self%random_state>=0,'AdaBoost random_state must be nonnegative')
    call require(ieee_is_finite(self%learning_rate),'nonfinite AdaBoost learning rate')
    call require(self%learning_rate>0,'AdaBoost learning rate must be positive')
    call require(self%max_depth>=1.and.self%min_samples_split>=2.and.self%min_samples_leaf>=1, &
      'invalid AdaBoost CART size controls')
    call require(self%max_features>=0.and.self%max_features<=p,'invalid AdaBoost CART max_features')
    call require(any(self%criterion==['gini    ','entropy ','log_loss']),'invalid AdaBoost CART criterion')
    call require(self%splitter=='best'.or.self%splitter=='random','invalid AdaBoost CART splitter')
    call require(ieee_is_finite(self%min_impurity_decrease),'AdaBoost CART impurity decrease must be finite')
    call require(self%min_impurity_decrease>=0,'negative AdaBoost CART impurity decrease')
    call require(maxval(abs(x))<=real(huge(1._real32),dp),'AdaBoost input exceeds float32 range')
    self%classes_=unique_labels(y);k=size(self%classes_);w=spread(1._dp,1,n)
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'AdaBoost sample weight count mismatch')
      call require(all(ieee_is_finite(sample_weight)),'nonfinite AdaBoost sample weights')
      call require(all(sample_weight>=0),'negative AdaBoost sample weights');w=sample_weight
    end if
    call require(maxval(w)>0,'AdaBoost needs positive total sample weight')
    total=numpy_sum(w)
    if(.not.ieee_is_finite(total)) then
      ! Preserve the existing native weight-scale correction when the source
      ! normalization would turn every finite weight into zero.
      w=w/maxval(w);total=numpy_sum(w)
    end if
    w=w/total;positive=w>0;data=real(real(x,real32),dp)
    if(allocated(self%stump_feature_)) deallocate(self%stump_feature_,self%stump_left_,self%stump_right_, &
      self%stump_threshold_,self%estimator_weights_,self%estimator_errors_,self%feature_importances_)
    if(allocated(self%estimators_)) deallocate(self%estimators_)
    allocate(self%estimators_(self%n_estimators))
    self%estimator_=decision_tree_classifier()
    self%estimator_%max_depth=self%max_depth;self%estimator_%min_samples_split=self%min_samples_split
    self%estimator_%min_samples_leaf=self%min_samples_leaf;self%estimator_%max_features=self%max_features
    self%estimator_%criterion=self%criterion;self%estimator_%random_thresholds=self%splitter=='random'
    self%estimator_%min_impurity_decrease=self%min_impurity_decrease
    self%estimator_%random_state=self%tree_random_state
    allocate(self%stump_feature_(self%n_estimators),self%stump_left_(self%n_estimators), &
      self%stump_right_(self%n_estimators),self%stump_threshold_(self%n_estimators))
    self%stump_feature_=0;self%stump_left_=self%classes_(1);self%stump_right_=self%classes_(1);self%stump_threshold_=0
    self%estimator_weights_=spread(0._dp,1,self%n_estimators)
    self%estimator_errors_=spread(1._dp,1,self%n_estimators);self%feature_importances_=spread(0._dp,1,p)
    self%n_estimators_=0;self%n_features_in_=p;self%n_classes_=k
    call rng%seed(self%random_state)
    do t=1,self%n_estimators
      where(positive) w=max(w,epsilon(1._dp))
      where(.not.positive) w=0
      self%estimators_(t)=self%estimator_
      self%estimators_(t)%random_state=rng%integer_index(2147483647)-1
      call self%estimators_(t)%fit(data,y,w)
      prediction=self%estimators_(t)%predict(data);incorrect=prediction/=y
      error=numpy_sum(merge(w,0._dp,incorrect))/numpy_sum(w)
      if(error>0) then
        if(error>=1-1._dp/k) then
          call require(t>1,'AdaBoost weak learner is no better than random guessing');exit
        end if
        weight=ieee_product(self%learning_rate,log((1-error)/error)+log(real(k-1,dp)))
      else
        weight=1
      end if
      self%stump_feature_(t)=self%estimators_(t)%nodes_(1)%feature
      self%stump_threshold_(t)=self%estimators_(t)%nodes_(1)%threshold
      left=self%estimators_(t)%nodes_(1)%left;right=self%estimators_(t)%nodes_(1)%right
      if(left==0) then
        left=1;right=1
      end if
      self%stump_left_(t)=self%classes_(ieee_argmax(self%estimators_(t)%nodes_(left)%probability))
      self%stump_right_(t)=self%classes_(ieee_argmax(self%estimators_(t)%nodes_(right)%probability))
      self%estimator_weights_(t)=weight;self%estimator_errors_(t)=error;self%n_estimators_=t
      if(error<=0) exit
      if(t<self%n_estimators) then
        do i=1,n
          log_weight=ieee_value(0._dp,ieee_negative_inf)
          if(w(i)>0) log_weight=log(w(i))
          delta=ieee_product(weight,merge(1._dp,0._dp,incorrect(i)))
          delta=ieee_product(delta,merge(1._dp,0._dp,w(i)>0))
          w(i)=quiet_exp(plus(log_weight,delta))
        end do
      end if
      total=numpy_sum(w)
      if(.not.ieee_is_finite(total)) exit
      if(total<=0) exit
      if(t<self%n_estimators) w=w/total
    end do
    self%estimators_=self%estimators_(:self%n_estimators_)
    do t=1,self%n_estimators_
      do i=1,p
        self%feature_importances_(i)=plus(self%feature_importances_(i), &
          ieee_product(self%estimator_weights_(t),self%estimators_(t)%feature_importances_(i)))
      end do
    end do
    total=numpy_sum(self%estimator_weights_)
    do i=1,p
      self%feature_importances_(i)=ieee_quotient(self%feature_importances_(i),total)
    end do
  end subroutine

  subroutine prediction_data(self,x)
    class(adaboost_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    call require(self%n_estimators_>0,'AdaBoost is not fitted');call check_x(x)
    call require(size(x,2)==self%n_features_in_,'AdaBoost feature count mismatch')
    if(self%n_classes_>1) then
      call require(maxval(abs(x))<=real(huge(1._real32),dp),'AdaBoost input exceeds float32 range')
    end if
  end subroutine
  subroutine accumulate_votes(self,x,t,full,first)
    class(adaboost_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: t
    real(dp), intent(inout) :: full(:,:)
    logical, intent(in) :: first
    integer, allocatable :: prediction(:)
    real(dp) :: contribution
    integer :: c,i
    prediction=self%estimators_(t)%predict(x)
    do c=1,self%n_classes_
      do i=1,size(x,1)
        if(prediction(i)==self%classes_(c)) then
          contribution=self%estimator_weights_(t)
        else
          contribution=ieee_product(-1._dp/(self%n_classes_-1),self%estimator_weights_(t))
        end if
        if(first) then
          full(i,c)=contribution
        else
          full(i,c)=plus(full(i,c),contribution)
        end if
      end do
    end do
  end subroutine
  function normalize_votes(self,full,total) result(scores)
    class(adaboost_classifier), intent(in) :: self
    real(dp), intent(in) :: full(:,:),total
    real(dp), allocatable :: scores(:,:)
    integer :: c,i
    if(self%n_classes_==2) then
      allocate(scores(size(full,1),1))
      do i=1,size(full,1)
        scores(i,1)=ieee_difference(ieee_quotient(full(i,2),total),ieee_quotient(full(i,1),total))
      end do
    else
      allocate(scores(size(full,1),size(full,2)))
      do c=1,size(full,2)
        do i=1,size(full,1)
          scores(i,c)=ieee_quotient(full(i,c),total)
        end do
      end do
    end if
  end function
  function ada_scores(self,x) result(scores)
    class(adaboost_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: scores(:,:),full(:,:)
    integer :: t
    call prediction_data(self,x)
    if(self%n_classes_==1) then
      allocate(scores(size(x,1),1));scores=0;return
    end if
    allocate(full(size(x,1),self%n_classes_));full=0
    do t=1,self%n_estimators_
      call accumulate_votes(self,x,t,full,.false.)
    end do
    scores=normalize_votes(self,full,numpy_sum(self%estimator_weights_))
  end function

  function ada_decision(self,x) result(scores)
    class(adaboost_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: scores(:,:)
    scores=ada_scores(self,x)
  end function
  function labels_from_scores(self,scores) result(y)
    class(adaboost_classifier), intent(in) :: self
    real(dp), intent(in) :: scores(:,:)
    integer, allocatable :: y(:)
    integer :: i
    if(self%n_classes_==2) then
      allocate(y(size(scores,1)));y=self%classes_(1)
      do i=1,size(y)
        if(ieee_is_nan(scores(i,1))) cycle
        if(scores(i,1)>0) y(i)=self%classes_(2)
      end do
    else
      allocate(y(size(scores,1)))
      do i=1,size(y)
        y(i)=self%classes_(ieee_argmax(scores(i,:)))
      end do
    end if
  end function
  function proba_from_scores(self,scores) result(prob)
    class(adaboost_classifier), intent(in) :: self
    real(dp), intent(in) :: scores(:,:)
    real(dp), allocatable :: prob(:,:)
    real(dp) :: maximum,total
    integer :: i
    if(self%n_classes_==1) then
      allocate(prob(size(scores,1),1));prob=1;return
    else if(self%n_classes_==2) then
      allocate(prob(size(scores,1),2));prob(:,1)=-scores(:,1)/2;prob(:,2)=scores(:,1)/2
    else
      prob=scores/(self%n_classes_-1)
    end if
    do i=1,size(prob,1)
      if(any(ieee_is_nan(prob(i,:)))) then
        prob(i,:)=ieee_value(0._dp,ieee_quiet_nan)
      else
        maximum=maxval(prob(i,:));prob(i,:)=exp(prob(i,:)-maximum)
        total=numpy_sum(prob(i,:));prob(i,:)=prob(i,:)/total
      end if
    end do
  end function
  function ada_predict(self,x) result(y)
    class(adaboost_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: y(:)
    y=labels_from_scores(self,self%decision_function(x))
  end function
  function ada_proba(self,x) result(prob)
    class(adaboost_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: prob(:,:)
    call require(self%n_estimators_>0,'AdaBoost is not fitted')
    if(self%n_classes_==1) then
      allocate(prob(size(x,1),1));prob=1
    else
      prob=proba_from_scores(self,self%decision_function(x))
    end if
  end function
  function ada_log_proba(self,x) result(prob)
    class(adaboost_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: prob(:,:)
    integer :: i,j
    prob=self%predict_proba(x)
    do j=1,size(prob,2)
      do i=1,size(prob,1)
        if(ieee_is_nan(prob(i,j))) cycle
        if(prob(i,j)==0) then
          prob(i,j)=ieee_value(0._dp,ieee_negative_inf)
        else
          prob(i,j)=log(prob(i,j))
        end if
      end do
    end do
  end function
  real(dp) function accuracy(y,prediction,sample_weight) result(value)
    integer, intent(in) :: y(:),prediction(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    real(dp), allocatable :: terms(:)
    real(dp) :: total
    integer :: i
    call require(size(y)==size(prediction),'AdaBoost score target count mismatch')
    if(present(sample_weight)) then
      call require(size(sample_weight)==size(y),'AdaBoost score weight count mismatch')
      total=numpy_sum(sample_weight)
      if(ieee_is_nan(total)) then
        value=total;return
      end if
      call require(total/=0,'AdaBoost score weights sum to zero')
      allocate(terms(size(y)))
      do i=1,size(y)
        terms(i)=ieee_product(sample_weight(i),merge(1._dp,0._dp,y(i)==prediction(i)))
      end do
      value=ieee_quotient(numpy_sum(terms),total)
    else
      value=real(count(y==prediction),dp)/size(y)
    end if
  end function
  real(dp) function ada_score(self,x,y,sample_weight) result(value)
    class(adaboost_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    value=accuracy(y,self%predict(x),sample_weight)
  end function

  function ada_staged_decision(self,x) result(scores)
    class(adaboost_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: scores(:,:,:),full(:,:)
    real(dp) :: total
    integer :: t,channels
    call prediction_data(self,x);channels=self%n_classes_
    if(channels==2) channels=1
    allocate(scores(size(x,1),channels,self%n_estimators_));scores=0
    if(self%n_classes_==1) return
    allocate(full(size(x,1),self%n_classes_));full=0;total=0
    do t=1,self%n_estimators_
      call accumulate_votes(self,x,t,full,t==1);total=plus(total,self%estimator_weights_(t))
      scores(:,:,t)=normalize_votes(self,full,total)
    end do
  end function
  function ada_staged_predict(self,x) result(y)
    class(adaboost_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: y(:,:)
    real(dp), allocatable :: scores(:,:,:)
    integer :: t
    scores=self%staged_decision_function(x); allocate(y(size(x,1),self%n_estimators_))
    do t=1,self%n_estimators_
      y(:,t)=labels_from_scores(self,scores(:,:,t))
    end do
  end function
  function ada_staged_proba(self,x) result(prob)
    class(adaboost_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: prob(:,:,:),scores(:,:,:)
    integer :: t
    scores=self%staged_decision_function(x); allocate(prob(size(x,1),self%n_classes_,self%n_estimators_))
    do t=1,self%n_estimators_
      prob(:,:,t)=proba_from_scores(self,scores(:,:,t))
    end do
  end function
  function ada_staged_score(self,x,y,sample_weight) result(values)
    class(adaboost_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    real(dp), allocatable :: values(:)
    integer, allocatable :: predictions(:,:)
    integer :: t
    predictions=self%staged_predict(x);allocate(values(self%n_estimators_))
    do t=1,self%n_estimators_
      values(t)=accuracy(y,predictions(:,t),sample_weight)
    end do
  end function
end module
