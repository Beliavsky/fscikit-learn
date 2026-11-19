! SAMME contracts: sklearn 1.6.1 ensemble/_weight_boosting.py (BSD).
! Weighted Gini stumps are implemented natively; sklearn notice retained.
module sklearn_adaboost
  use sklearn_common, only: dp,require,check_x,check_xy,unique_labels,label_index,argsort
  use sklearn_metrics, only: accuracy_score
  use, intrinsic :: iso_fortran_env, only: real32
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: adaboost_classifier
  type :: adaboost_classifier
    integer :: n_estimators=50,n_estimators_=0,n_classes_=0,n_features_in_=0
    real(dp) :: learning_rate=1
    integer, allocatable :: classes_(:),stump_feature_(:),stump_left_(:),stump_right_(:)
    real(dp), allocatable :: stump_threshold_(:),estimator_weights_(:),estimator_errors_(:),feature_importances_(:)
  contains
    procedure :: fit=>ada_fit
    procedure :: decision_function=>ada_decision
    procedure :: predict=>ada_predict
    procedure :: predict_proba=>ada_proba
    procedure :: score=>ada_score
    procedure :: staged_decision_function=>ada_staged_decision
    procedure :: staged_predict=>ada_staged_predict
    procedure :: staged_predict_proba=>ada_staged_proba
  end type
contains
  subroutine fit_stump(x,y,w,k,feature,threshold,left_class,right_class)
    real(dp), intent(in) :: x(:,:),w(:)
    integer, intent(in) :: y(:),k
    integer, intent(out) :: feature,left_class,right_class
    real(dp), intent(out) :: threshold
    real(dp) :: total(k),left(k),right(k),wl,wr,best,gain,cut
    integer, allocatable :: members(:),order(:),sorted(:)
    integer :: i,j,n,lc,rc
    total=0
    do i=1,size(y)
      total(y(i))=total(y(i))+w(i)
    end do
    feature=0; threshold=0; left_class=maxloc(total,dim=1); right_class=left_class
    best=sum(total**2)/sum(w)
    members=pack([(i,i=1,size(y))],w>0); n=size(members)
    do j=1,size(x,2)
      order=argsort(x(members,j)); sorted=members(order); left=0; right=total; wl=0; wr=sum(w)
      do i=1,n-1
        left(y(sorted(i)))=left(y(sorted(i)))+w(sorted(i))
        right(y(sorted(i)))=right(y(sorted(i)))-w(sorted(i))
        wl=wl+w(sorted(i)); wr=wr-w(sorted(i))
        if(x(sorted(i+1),j)<=x(sorted(i),j)+1.e-7_dp) cycle
        if(wl<=0.or.wr<=0) cycle
        gain=sum(left**2)/wl+sum(right**2)/wr
        if(gain<=best+epsilon(1._dp)*max(1._dp,abs(best))) cycle
        best=gain; cut=x(sorted(i),j)/2+x(sorted(i+1),j)/2
        if(cut==x(sorted(i+1),j)) cut=x(sorted(i),j)
        feature=j; threshold=cut; lc=maxloc(left,dim=1); rc=maxloc(right,dim=1)
        left_class=lc; right_class=rc
      end do
    end do
  end subroutine

  subroutine ada_fit(self,x,y,sample_weight)
    class(adaboost_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: data(:,:),w(:),logs(:)
    integer, allocatable :: encoded(:),prediction(:)
    logical, allocatable :: positive(:),incorrect(:)
    real(dp) :: error,weight,scale
    integer :: n,p,k,i,t,feature,lc,rc
    real(dp) :: threshold
    call check_xy(x,size(y)); n=size(y); p=size(x,2)
    call require(self%n_estimators>0,'AdaBoost n_estimators must be positive')
    call require(ieee_is_finite(self%learning_rate),'nonfinite AdaBoost learning rate')
    call require(self%learning_rate>0,'AdaBoost learning rate must be positive')
    call require(maxval(abs(x))<=real(huge(1._real32),dp),'AdaBoost input exceeds float32 range')
    self%classes_=unique_labels(y); k=size(self%classes_)
    call require(k>=2,'AdaBoost requires at least two classes')
    encoded=[(label_index(self%classes_,y(i)),i=1,n)]; w=spread(1._dp,1,n)
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'AdaBoost sample weight count mismatch')
      call require(all(ieee_is_finite(sample_weight)),'nonfinite AdaBoost sample weights')
      call require(all(sample_weight>=0),'negative AdaBoost sample weights'); w=sample_weight
    end if
    scale=maxval(w); call require(scale>0,'AdaBoost needs positive total sample weight')
    w=w/scale; w=w/sum(w); positive=w>0; data=real(real(x,real32),dp)
    if(allocated(self%stump_feature_)) deallocate(self%stump_feature_,self%stump_left_,self%stump_right_, &
      self%stump_threshold_,self%estimator_weights_,self%estimator_errors_,self%feature_importances_)
    allocate(self%stump_feature_(self%n_estimators),self%stump_left_(self%n_estimators), &
      self%stump_right_(self%n_estimators),self%stump_threshold_(self%n_estimators))
    self%stump_feature_=0; self%stump_left_=1; self%stump_right_=1; self%stump_threshold_=0
    self%estimator_weights_=spread(0._dp,1,self%n_estimators)
    self%estimator_errors_=spread(1._dp,1,self%n_estimators); self%feature_importances_=spread(0._dp,1,p)
    self%n_estimators_=0; self%n_features_in_=p; self%n_classes_=k
    do t=1,self%n_estimators
      where(positive) w=max(w,epsilon(1._dp))
      call fit_stump(data,encoded,w,k,feature,threshold,lc,rc)
      prediction=spread(lc,1,n)
      if(feature>0) prediction=merge(lc,rc,data(:,feature)<=threshold)
      incorrect=prediction/=encoded; error=sum(pack(w,incorrect))/sum(w)
      if(error>=1-1._dp/k) then
        call require(t>1,'AdaBoost stump is no better than random guessing'); exit
      end if
      weight=1
      if(error>0) weight=self%learning_rate*(log((1-error)/error)+log(real(k-1,dp)))
      call require(ieee_is_finite(weight),'AdaBoost estimator weight overflow')
      self%stump_feature_(t)=feature; self%stump_threshold_(t)=threshold
      self%stump_left_(t)=self%classes_(lc); self%stump_right_(t)=self%classes_(rc)
      self%estimator_weights_(t)=weight; self%estimator_errors_(t)=error; self%n_estimators_=t
      if(feature>0) self%feature_importances_(feature)=self%feature_importances_(feature)+weight
      if(error<=0.or.t==self%n_estimators) exit
      ! Equivalent normalized log update, avoiding exponential overflow and log(0).
      logs=spread(-huge(1._dp),1,n)
      where(positive) logs=log(max(w,tiny(1._dp)))+merge(weight,0._dp,incorrect)
      scale=maxval(logs); w=0
      where(positive) w=exp(logs-scale)
      w=w/sum(w)
    end do
    self%feature_importances_=self%feature_importances_/sum(self%estimator_weights_)
  end subroutine

  function ada_scores(self,x,limit) result(scores)
    class(adaboost_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: limit
    real(dp), allocatable :: scores(:,:),data(:,:)
    integer, allocatable :: prediction(:)
    integer :: t,c,f,channels
    call require(self%n_estimators_>0,'AdaBoost is not fitted'); call check_x(x)
    call require(size(x,2)==self%n_features_in_,'AdaBoost feature count mismatch')
    call require(maxval(abs(x))<=real(huge(1._real32),dp),'AdaBoost input exceeds float32 range')
    data=real(real(x,real32),dp); channels=self%n_classes_
    if(channels==2) channels=1
    allocate(scores(size(x,1),channels)); scores=0
    do t=1,limit
      f=self%stump_feature_(t); prediction=spread(self%stump_left_(t),1,size(x,1))
      if(f>0) prediction=merge(self%stump_left_(t),self%stump_right_(t),data(:,f)<=self%stump_threshold_(t))
      if(self%n_classes_==2) then
        scores(:,1)=scores(:,1)+2*self%estimator_weights_(t)*merge(1._dp,-1._dp,prediction==self%classes_(2))
      else
        do c=1,channels
          scores(:,c)=scores(:,c)+self%estimator_weights_(t)* &
            merge(1._dp,-1._dp/(channels-1),prediction==self%classes_(c))
        end do
      end if
    end do
    scores=scores/sum(self%estimator_weights_(:limit))
  end function

  function ada_decision(self,x) result(scores)
    class(adaboost_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: scores(:,:)
    scores=ada_scores(self,x,self%n_estimators_)
  end function
  function labels_from_scores(self,scores) result(y)
    class(adaboost_classifier), intent(in) :: self
    real(dp), intent(in) :: scores(:,:)
    integer, allocatable :: y(:)
    if(self%n_classes_==2) then
      y=self%classes_(merge(2,1,scores(:,1)>0))
    else
      y=self%classes_(maxloc(scores,dim=2))
    end if
  end function
  function proba_from_scores(self,scores) result(prob)
    class(adaboost_classifier), intent(in) :: self
    real(dp), intent(in) :: scores(:,:)
    real(dp), allocatable :: prob(:,:)
    if(self%n_classes_==2) then
      allocate(prob(size(scores,1),2)); prob(:,1)=-scores(:,1)/2; prob(:,2)=scores(:,1)/2
    else
      prob=scores/(self%n_classes_-1)
    end if
    prob=exp(prob-spread(maxval(prob,dim=2),2,size(prob,2)))
    prob=prob/spread(sum(prob,dim=2),2,size(prob,2))
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
    prob=proba_from_scores(self,self%decision_function(x))
  end function
  real(dp) function ada_score(self,x,y) result(value)
    class(adaboost_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    value=accuracy_score(y,self%predict(x))
  end function
  function ada_staged_decision(self,x) result(scores)
    class(adaboost_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: scores(:,:,:),current(:,:)
    integer :: t
    current=self%decision_function(x); allocate(scores(size(x,1),size(current,2),self%n_estimators_))
    do t=1,self%n_estimators_
      scores(:,:,t)=ada_scores(self,x,t)
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
end module
