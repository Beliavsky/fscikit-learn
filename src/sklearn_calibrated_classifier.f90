! Adapted from sklearn 1.6.1 calibration.py; retained BSD attribution in
! THIRD_PARTY_NOTICES.md and LICENSES/scikit-learn-COPYING.
module sklearn_calibrated_classifier
  use sklearn_common, only: dp,require,check_xy,check_x,unique_labels,sigmoid
  use sklearn_metrics, only: accuracy_score
  use sklearn_linear_model, only: logistic_regression
  use sklearn_naive_bayes, only: gaussian_nb
  use sklearn_neighbors, only: kneighbors_classifier
  use sklearn_isotonic, only: isotonic_regression
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: calibrated_classifier_cv
  type :: calibration_map
    real(dp) :: a_=0,b_=0
    logical :: isotonic_=.false.
    type(isotonic_regression) :: isotonic_model_
  end type
  type :: calibrated_pair
    class(*), allocatable :: estimator_
    type(calibration_map), allocatable :: calibrators_(:)
  end type
  type :: calibrated_classifier_cv
    character(8) :: method='sigmoid'
    integer :: cv=5,n_features_in_=0
    logical :: ensemble=.true.,prefit=.false.
    integer, allocatable :: classes_(:),fold_id_(:)
    type(calibrated_pair), allocatable :: calibrated_classifiers_(:)
    logical, private :: fitted_=.false.
  contains
    procedure :: fit=>calibrated_fit
    procedure :: predict_proba=>calibrated_proba
    procedure :: predict=>calibrated_predict
    procedure :: score=>calibrated_score
  end type
contains
  subroutine fit_base(model,x,y,sample_weight)
    class(*), intent(inout) :: model
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    select type(model)
    type is(logistic_regression)
      call model%fit(x,y,sample_weight)
    type is(gaussian_nb)
      call model%fit(x,y,sample_weight)
    type is(kneighbors_classifier)
      call model%fit(x,y)
    class default
      call require(.false.,'unsupported calibrated classifier estimator')
    end select
  end subroutine

  function base_labels(model) result(labels)
    class(*), intent(in) :: model
    integer, allocatable :: labels(:)
    select type(model)
    type is(logistic_regression)
      call require(allocated(model%classes_),'prefit calibration estimator is not fitted')
      labels=model%classes_
    type is(gaussian_nb)
      call require(allocated(model%classes_),'prefit calibration estimator is not fitted')
      labels=model%classes_
    type is(kneighbors_classifier)
      call require(allocated(model%classes_),'prefit calibration estimator is not fitted')
      labels=model%classes_
    class default
      call require(.false.,'unsupported calibrated classifier estimator')
    end select
  end function

  function response(model,x) result(values)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:,:),prob(:,:)
    integer, allocatable :: labels(:)
    labels=base_labels(model)
    select type(model)
    type is(logistic_regression)
      values=model%decision_function(x)
    type is(gaussian_nb)
      prob=model%predict_proba(x)
    type is(kneighbors_classifier)
      prob=model%predict_proba(x)
    end select
    if(allocated(prob)) then
      if(size(labels)==2) then
        values=prob(:,2:2)
      else
        values=prob
      end if
    end if
    call require(all(ieee_is_finite(values)),'calibration responses must be finite')
  end function

  subroutine sigmoid_fit(map,x,y,weights)
    type(calibration_map), intent(inout) :: map
    real(dp), intent(in) :: x(:),y(:),weights(:)
    real(dp) :: f(size(x)),target(size(x)),w(size(x)),theta(2),trial(2),g(2),h(2,2),step(2)
    real(dp) :: prior0,prior1,scale,loss,next_loss,rate,det,trace,directional,next_g(2),next_h(2,2)
    integer :: it,j
    prior0=sum(weights,mask=y<=0); prior1=sum(weights,mask=y>0)
    target=merge((prior1+1)/(prior1+2),1/(prior0+2),y>0)
    scale=1; if(maxval(abs(x))>=30) scale=maxval(abs(x))
    f=x/scale; w=weights/max(1._dp,sum(weights))
    theta=[0._dp,log(prior0+1)-log(prior1+1)]
    ! Damped two-variable Newton minimizes exactly the source Platt objective.
    do it=1,300
      call evaluate(theta,loss,g,h)
      if(maxval(abs(g))<=1.e-11_dp) exit
      trace=h(1,1)+h(2,2); det=h(1,1)*h(2,2)-h(1,2)**2
      if(det>1.e-12_dp*h(1,1)*h(2,2)) then
        step=[h(2,2)*g(1)-h(1,2)*g(2),h(1,1)*g(2)-h(1,2)*g(1)]/det
      else
        ! Moore-Penrose Newton direction for constant/rank-one score designs.
        step=g/max(trace,1.e-16_dp)
      end if
      step=step/max(1._dp,maxval(abs(step))/100)
      directional=dot_product(g,step); rate=1
      do j=1,60
        trial=theta-rate*step
        call evaluate(trial,next_loss,next_g,next_h)
        if(next_loss<=loss-1.e-4_dp*rate*directional) exit
        ! Near the optimum, summed loss can round upward by an ulp even while
        ! the Newton step reduces the independently evaluated gradient.
        if(next_loss<=loss+8*epsilon(1._dp)*(1+abs(loss)).and.maxval(abs(next_g))<maxval(abs(g))) exit
        rate=rate/2
      end do
      theta=trial
    end do
    call evaluate(theta,loss,g,h)
    call require(maxval(abs(g))<=2.e-9_dp,'sigmoid calibration failed to converge')
    map%a_=theta(1)/scale; map%b_=theta(2)
  contains
    subroutine evaluate(coefficients,value,gradient,hessian)
      real(dp), intent(in) :: coefficients(2)
      real(dp), intent(out) :: value
      real(dp), intent(out), optional :: gradient(2),hessian(2,2)
      real(dp) :: raw,p,d,v,design(2)
      integer :: i
      value=0
      if(present(gradient)) gradient=0
      if(present(hessian)) hessian=0
      do i=1,size(f)
        raw=-(coefficients(1)*f(i)+coefficients(2)); p=sigmoid(raw)
        value=value+w(i)*(max(raw,0._dp)-target(i)*raw+log(1+exp(-abs(raw))))
        if(present(gradient)) then
          design=[f(i),1._dp]; d=w(i)*(target(i)-p); gradient=gradient+d*design
          v=w(i)*p*(1-p); hessian(:,1)=hessian(:,1)+v*design*f(i)
          hessian(:,2)=hessian(:,2)+v*design
        end if
      end do
    end subroutine
  end subroutine

  subroutine fit_maps(pair,scores,y,classes,method,weights)
    type(calibrated_pair), intent(inout) :: pair
    real(dp), intent(in) :: scores(:,:),weights(:)
    integer, intent(in) :: y(:),classes(:)
    character(*), intent(in) :: method
    real(dp) :: target(size(y))
    integer :: c,k,label
    k=size(classes)
    allocate(pair%calibrators_(size(scores,2)))
    do c=1,size(scores,2)
      label=classes(c); if(k==2) label=classes(2)
      target=merge(1._dp,0._dp,y==label)
      pair%calibrators_(c)%isotonic_=method=='isotonic'
      if(method=='isotonic') then
        pair%calibrators_(c)%isotonic_model_%out_of_bounds='clip'
        ! Binary targets bound the exact PAV solution; contain rounding at 0/1.
        pair%calibrators_(c)%isotonic_model_%y_min=0
        pair%calibrators_(c)%isotonic_model_%y_max=1
        call pair%calibrators_(c)%isotonic_model_%fit(scores(:,c),target,weights)
      else
        call sigmoid_fit(pair%calibrators_(c),scores(:,c),target,weights)
      end if
    end do
  end subroutine

  function stratified_folds(y,cv) result(folds)
    integer, intent(in) :: y(:),cv
    integer :: folds(size(y)),classes(size(y)),nclasses,i,c,offset,quota,f,j,used
    ! Encode classes by first appearance, as sklearn StratifiedKFold does.
    nclasses=0
    do i=1,size(y)
      if(any(classes(:nclasses)==y(i))) cycle
      nclasses=nclasses+1; classes(nclasses)=y(i)
    end do
    offset=0
    do c=1,nclasses
      call require(count(y==classes(c))>=cv,'calibration CV needs at least cv samples in each class')
      used=0; j=1
      do f=0,cv-1
        quota=0
        do i=offset,offset+count(y==classes(c))-1
          if(mod(i,cv)==f) quota=quota+1
        end do
        do while(quota>0)
          if(y(j)==classes(c)) then
            folds(j)=f+1; used=used+1; quota=quota-1
          end if
          j=j+1
        end do
      end do
      offset=offset+used
    end do
  end function

  subroutine calibrated_fit(self,x,y,estimator,sample_weight)
    class(calibrated_classifier_cv), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    class(*), intent(in) :: estimator
    real(dp), intent(in), optional :: sample_weight(:)
    class(*), allocatable :: model
    real(dp) :: weights(size(y))
    real(dp), allocatable :: scores(:,:),oof(:,:)
    integer :: ids(size(y)),i,n,k,q,f,pair_count
    integer, allocatable :: train(:),test(:)
    self%fitted_=.false.; call check_xy(x,size(y)); n=size(y)
    call require(self%method=='sigmoid'.or.self%method=='isotonic','unknown calibration method')
    select type(estimator)
    type is(logistic_regression)
    type is(gaussian_nb)
    type is(kneighbors_classifier)
    class default
      call require(.false.,'unsupported calibrated classifier estimator')
    end select
    weights=1
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'calibration sample weight length mismatch')
      call require(all(ieee_is_finite(sample_weight)),'calibration sample weights must be finite')
      call require(all(sample_weight>=0),'calibration sample weights must be nonnegative')
      call require(maxval(sample_weight)<=huge(1._dp)/(16*real(n,dp)),'calibration weight mass exceeds working range')
      weights=sample_weight
    end if
    if(self%prefit) then
      self%classes_=base_labels(estimator)
    else
      self%classes_=unique_labels(y)
    end if
    k=size(self%classes_); call require(k>=2,'calibration requires at least two estimator classes')
    self%n_features_in_=size(x,2); ids=[(i,i=1,n)]
    if(allocated(self%calibrated_classifiers_)) deallocate(self%calibrated_classifiers_)
    if(allocated(self%fold_id_)) deallocate(self%fold_id_)
    if(self%prefit) then
      allocate(self%calibrated_classifiers_(1))
      allocate(self%calibrated_classifiers_(1)%estimator_,source=estimator)
      scores=response(estimator,x)
      call fit_maps(self%calibrated_classifiers_(1),scores,y,self%classes_,self%method,weights)
    else
      call require(self%cv>=2.and.self%cv<=n,'calibration cv must lie between two and the sample count')
      self%fold_id_=stratified_folds(y,self%cv)
      pair_count=1; if(self%ensemble) pair_count=self%cv
      allocate(self%calibrated_classifiers_(pair_count)); q=k; if(k==2) q=1
      allocate(oof(n,q))
      do f=1,self%cv
        train=pack(ids,self%fold_id_/=f); test=pack(ids,self%fold_id_==f)
        allocate(model,source=estimator)
        if(present(sample_weight)) then
          call fit_base(model,x(train,:),y(train),weights(train))
        else
          call fit_base(model,x(train,:),y(train))
        end if
        scores=response(model,x(test,:))
        if(self%ensemble) then
          call move_alloc(model,self%calibrated_classifiers_(f)%estimator_)
          call fit_maps(self%calibrated_classifiers_(f),scores,y(test),self%classes_,self%method,weights(test))
        else
          oof(test,:)=scores; deallocate(model)
        end if
      end do
      if(.not.self%ensemble) then
        allocate(self%calibrated_classifiers_(1)%estimator_,source=estimator)
        call fit_base(self%calibrated_classifiers_(1)%estimator_,x,y,sample_weight)
        call fit_maps(self%calibrated_classifiers_(1),oof,y,self%classes_,self%method,weights)
      end if
    end if
    self%fitted_=.true.
  end subroutine

  function map_predict(map,x) result(values)
    type(calibration_map), intent(in) :: map
    real(dp), intent(in) :: x(:)
    real(dp), allocatable :: values(:)
    real(dp) :: raw
    integer :: i
    if(map%isotonic_) then
      values=map%isotonic_model_%predict(x)
    else
      allocate(values(size(x)))
      do i=1,size(x)
        if(abs(map%a_)>1) then
          if(abs(x(i))>huge(1._dp)/(2*abs(map%a_))) then
            values(i)=merge(0._dp,1._dp,sign(1._dp,map%a_)==sign(1._dp,x(i))); cycle
          end if
        end if
        raw=-(map%a_*x(i)+map%b_); values(i)=sigmoid(raw)
      end do
    end if
  end function

  function calibrated_proba(self,x) result(prob)
    class(calibrated_classifier_cv), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: prob(:,:),scores(:,:),current(:,:)
    real(dp) :: total
    integer :: f,c,i,k
    call require(self%fitted_,'calibrated classifier is not fitted')
    call check_x(x); call require(size(x,2)==self%n_features_in_,'calibration feature count mismatch')
    k=size(self%classes_); allocate(prob(size(x,1),k),current(size(x,1),k)); prob=0
    do f=1,size(self%calibrated_classifiers_)
      scores=response(self%calibrated_classifiers_(f)%estimator_,x); current=0
      do c=1,size(scores,2)
        i=c; if(k==2) i=2
        current(:,i)=map_predict(self%calibrated_classifiers_(f)%calibrators_(c),scores(:,c))
      end do
      if(k==2) then
        current(:,1)=1-current(:,2)
      else
        do i=1,size(x,1)
          total=sum(current(i,:))
          if(total==0) then
            current(i,:)=1._dp/k
          else
            current(i,:)=current(i,:)/total
          end if
        end do
      end if
      where(current>1.and.current<=1+1.e-5_dp) current=1
      prob=prob+current
    end do
    prob=prob/size(self%calibrated_classifiers_)
  end function

  function calibrated_predict(self,x) result(labels)
    class(calibrated_classifier_cv), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: labels(size(x,1))
    real(dp), allocatable :: prob(:,:)
    prob=self%predict_proba(x); labels=self%classes_(maxloc(prob,dim=2))
  end function

  real(dp) function calibrated_score(self,x,y) result(value)
    class(calibrated_classifier_cv), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    value=accuracy_score(y,self%predict(x))
  end function
end module
