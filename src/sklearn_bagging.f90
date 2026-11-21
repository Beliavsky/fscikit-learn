! Bagging: sklearn 1.6.1 ensemble/_bagging.py and _base.py (BSD-3-Clause).
! Dense native adaptation; see THIRD_PARTY_NOTICES.md and LICENSES.
module sklearn_bagging
  use sklearn_common, only: dp,require,check_xy,check_x,unique_labels,label_index,dense_svd
  use sklearn_linear_model, only: linear_regression,ridge
  use sklearn_tree, only: decision_tree_classifier,decision_tree_regressor
  use sklearn_base_dispatch, only: train_classifier,train_regressor,labels,regression,probabilities, &
    base_weighted,base_supported,base_proba,base_classes,seed_base
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_bagging_sampling, only: sample_indices
  use sklearn_ieee_math, only: ieee_difference,ieee_product,ieee_quotient,ieee_total,ieee_mean,ieee_argmax
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value,ieee_quiet_nan
  implicit none
  private
  public :: bagging_classifier,bagging_regressor
  type :: bagging_member
    class(*), allocatable :: estimator
  end type
  type :: index_vector
    integer, allocatable :: values(:)
  end type
  type :: bagging_storage
    integer :: n_estimators=10,max_samples=0,max_features=0,random_state=0
    real(dp) :: max_samples_fraction=0,max_features_fraction=0
    logical :: bootstrap=.true.,bootstrap_features=.false.,warm_start=.false.,oob_score=.false.
    integer :: n_features_in_=0,n_estimators_=0,max_samples_=0,max_features_=0
    real(dp) :: oob_score_=0
    class(*), allocatable :: estimator_
    type(bagging_member), allocatable :: estimators_(:)
    type(index_vector), allocatable :: estimators_features_(:),estimators_samples_(:)
    integer, allocatable :: seeds_(:)
  end type
  type, extends(bagging_storage) :: bagging_classifier
    integer, allocatable :: classes_(:)
    integer :: n_classes_=0
    real(dp), allocatable :: oob_decision_function_(:,:)
  contains
    procedure :: fit=>classifier_fit
    procedure :: predict=>classifier_predict
    procedure :: predict_proba=>classifier_probability
    procedure :: score=>classifier_score
  end type
  type, extends(bagging_storage) :: bagging_regressor
    real(dp), allocatable :: oob_prediction_(:)
  contains
    procedure :: fit=>regressor_fit
    procedure :: predict=>regressor_predict
    procedure :: score=>regressor_score
  end type
contains
  logical function consumes_weights(model) result(value)
    class(*), intent(in) :: model
    value=base_weighted(model)
    select type(model)
    type is(linear_regression)
      value=.true.
    type is(ridge)
      value=.true.
    end select
  end function

  subroutine weighted_linear(model,x,y,w,alpha)
    class(linear_regression), intent(inout) :: model
    real(dp), intent(in) :: x(:,:),y(:),w(:),alpha
    real(dp) :: xc(size(x,1),size(x,2)),yc(size(y)),xm(size(x,2)),ym
    real(dp) :: u(size(x,1),size(x,2)),s(size(x,2)),v(size(x,2),size(x,2)),projection(size(x,2))
    real(dp) :: mass,scale,cutoff,limit
    integer :: i,j
    call require(ieee_is_finite(alpha),'bagging ridge alpha must be finite')
    call require(alpha>=0,'bagging ridge alpha must be nonnegative')
    call require(all(w>=0),'bagging linear weights must be nonnegative')
    mass=ieee_total(w)
    call require(ieee_is_finite(mass),'bagging linear weight sum must be finite')
    xm=0; ym=0
    if(model%fit_intercept) then
      call require(mass>0,'bagging linear weights must have a positive sum')
      do j=1,size(x,2)
        xm(j)=ieee_mean(x(:,j),w)
      end do
      ym=ieee_mean(y,w)
    end if
    call require(all(ieee_is_finite(xm)).and.ieee_is_finite(ym),'bagging linear weighted centering overflow')
    do i=1,size(y)
      scale=sqrt(w(i)); yc(i)=ieee_product(ieee_difference(y(i),ym),scale)
      do j=1,size(x,2)
        xc(i,j)=ieee_product(ieee_difference(x(i,j),xm(j)),scale)
      end do
    end do
    call require(all(ieee_is_finite(xc)).and.all(ieee_is_finite(yc)),'bagging linear rescaling overflow')
    limit=sqrt(huge(1._dp))/sqrt(real(max(size(x,1),size(x,2)),dp))/4
    call require(maxval(abs(xc))<=limit.and.maxval(abs(yc))<=limit,'bagging linear design exceeds working range')
    call dense_svd(xc,s,u,v)
    cutoff=epsilon(1._dp)*max(size(x,1),size(x,2))*maxval(s)
    model%rank_=count(s>cutoff); projection=matmul(transpose(u),yc)
    do j=1,size(s)
      if(s(j)>cutoff.or.alpha>0) then
        projection(j)=ieee_quotient(ieee_product(projection(j),s(j)), &
          ieee_difference(ieee_product(s(j),s(j)),-alpha))
      else
        projection(j)=0
      end if
    end do
    model%coef_=matmul(v,projection); model%intercept_=ym-dot_product(xm,model%coef_)
  end subroutine

  subroutine fit_weighted_regression(model,x,y,w)
    class(*), intent(inout) :: model
    real(dp), intent(in) :: x(:,:),y(:),w(:)
    select type(model)
    type is(linear_regression)
      call weighted_linear(model,x,y,w,0._dp)
    type is(ridge)
      call weighted_linear(model,x,y,w,model%alpha)
    class default
      call train_regressor(model,x,y,w)
    end select
  end subroutine

  subroutine controls(self,n,p,m,nf,old)
    class(bagging_storage), intent(in) :: self
    integer, intent(in) :: n,p
    integer, intent(out) :: m,nf,old
    call require(self%n_estimators>0,'bagging n_estimators must be positive')
    call require(self%random_state>=0,'bagging random_state must be nonnegative')
    call require(self%max_samples>=0.and.self%max_features>=0,'bagging sampling counts must be nonnegative')
    call require(ieee_is_finite(self%max_samples_fraction).and.ieee_is_finite(self%max_features_fraction), &
      'bagging sampling fractions must be finite')
    call require(self%max_samples_fraction>=0.and.self%max_samples_fraction<=1.and. &
      self%max_features_fraction>=0.and.self%max_features_fraction<=1,'bagging sampling fractions must be in [0,1]')
    call require(.not.(self%max_samples>0.and.self%max_samples_fraction>0).and. &
      .not.(self%max_features>0.and.self%max_features_fraction>0),'bagging sampling count/fraction conflict')
    m=n; nf=p
    if(self%max_samples>0) m=self%max_samples
    if(self%max_samples_fraction>0) m=int(self%max_samples_fraction*n)
    if(self%max_features>0) nf=self%max_features
    if(self%max_features_fraction>0) nf=max(1,int(self%max_features_fraction*p))
    call require(m<=n.and.nf<=p,'bagging sampling counts exceed population')
    call require(self%bootstrap.or..not.self%oob_score,'bagging OOB scoring requires bootstrap')
    call require(.not.(self%warm_start.and.self%oob_score),'bagging OOB scoring forbids warm_start')
    old=0
    if(self%warm_start.and.allocated(self%estimators_)) then
      old=size(self%estimators_)
      call require(self%n_estimators>=old,'bagging warm_start cannot reduce n_estimators')
      call require(p==self%n_features_in_,'bagging warm_start feature count mismatch')
    end if
  end subroutine

  subroutine fit_members(self,x,yr,yc,estimator,sample_weight)
    class(bagging_storage), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),yr(:)
    integer, intent(in) :: yc(:)
    class(*), intent(in), optional :: estimator
    real(dp), intent(in), optional :: sample_weight(:)
    type(decision_tree_classifier) :: default_c
    type(decision_tree_regressor) :: default_r
    type(bagging_member), allocatable :: members(:)
    type(index_vector), allocatable :: features(:),samples(:)
    integer, allocatable :: seeds(:)
    real(dp), allocatable :: w(:),current(:),design(:,:)
    type(legacy_random_state) :: rng,draw_rng
    integer :: n,p,m,nf,old,b,i,seed
    logical :: classifier,weighted
    n=size(x,1); p=size(x,2); classifier=size(yc)>0
    call controls(self,n,p,m,nf,old)
    if(allocated(self%estimator_)) deallocate(self%estimator_)
    if(present(estimator)) then
      allocate(self%estimator_,source=estimator)
    else if(classifier) then
      allocate(self%estimator_,source=default_c)
    else
      allocate(self%estimator_,source=default_r)
    end if
    call require(base_supported(self%estimator_,classifier),'unsupported native bagging base')
    weighted=consumes_weights(self%estimator_)
    allocate(w(n),current(n)); w=1
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'bagging sample weight length mismatch')
      call require(all(ieee_is_finite(sample_weight)),'bagging sample weights must be finite')
      call require(weighted,'bagging base has no native weighted fit'); w=sample_weight
    end if
    allocate(members(self%n_estimators),features(self%n_estimators),samples(self%n_estimators),seeds(self%n_estimators))
    if(old>0) then
      members(:old)=self%estimators_; features(:old)=self%estimators_features_
      samples(:old)=self%estimators_samples_; seeds(:old)=self%seeds_
    end if
    call rng%seed(self%random_state)
    do b=1,old
      seed=rng%integer_index(2147483647)-1
    end do
    do b=old+1,self%n_estimators
      seed=rng%integer_index(2147483647)-1; seeds(b)=seed
      allocate(members(b)%estimator,source=self%estimator_)
      call seed_base(members(b)%estimator,seed)
      call draw_rng%seed(seed)
      features(b)%values=sample_indices(draw_rng,p,nf,self%bootstrap_features)
      samples(b)%values=sample_indices(draw_rng,n,m,self%bootstrap)
      if(weighted) then
        current=0
        if(self%bootstrap) then
          do i=1,m
            current(samples(b)%values(i))=current(samples(b)%values(i))+1
          end do
          do i=1,n
            current(i)=ieee_product(current(i),w(i))
          end do
        else
          current(samples(b)%values)=w(samples(b)%values)
        end if
        call require(all(ieee_is_finite(current)),'bagging bootstrap weight overflow')
        design=x(:,features(b)%values)
        if(classifier) then
          call train_classifier(members(b)%estimator,design,yc,current)
        else
          call fit_weighted_regression(members(b)%estimator,design,yr,current)
        end if
      else
        design=x(samples(b)%values,features(b)%values)
        if(classifier) then
          call train_classifier(members(b)%estimator,design,yc(samples(b)%values))
        else
          call train_regressor(members(b)%estimator,design,yr(samples(b)%values))
        end if
      end if
    end do
    call move_alloc(members,self%estimators_); call move_alloc(features,self%estimators_features_)
    call move_alloc(samples,self%estimators_samples_); call move_alloc(seeds,self%seeds_)
    self%n_features_in_=p; self%n_estimators_=self%n_estimators
    self%max_samples_=m; self%max_features_=nf
  end subroutine

  subroutine classifier_fit(self,x,y,estimator,sample_weight)
    class(bagging_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    class(*), intent(in), optional :: estimator
    real(dp), intent(in), optional :: sample_weight(:)
    integer, allocatable :: encoded(:),classes(:)
    integer :: i
    call check_xy(x,size(y)); classes=unique_labels(y)
    if(self%warm_start.and.allocated(self%classes_)) then
      call require(size(classes)==size(self%classes_),'bagging warm_start class count mismatch')
      call require(all(classes==self%classes_),'bagging warm_start classes changed')
    end if
    self%classes_=classes; self%n_classes_=size(classes)
    encoded=[(label_index(classes,y(i))-1,i=1,size(y))]
    call fit_members(self,x,[real(dp)::],encoded,estimator,sample_weight)
    if(allocated(self%oob_decision_function_)) deallocate(self%oob_decision_function_)
    if(self%oob_score) call classifier_oob(self,x,encoded)
  end subroutine

  subroutine regressor_fit(self,x,y,estimator,sample_weight)
    class(bagging_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    class(*), intent(in), optional :: estimator
    real(dp), intent(in), optional :: sample_weight(:)
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    call fit_members(self,x,y,[integer::],estimator,sample_weight)
    if(allocated(self%oob_prediction_)) deallocate(self%oob_prediction_)
    if(self%oob_score) call regressor_oob(self,x,y)
  end subroutine

  subroutine query(self,x)
    class(bagging_storage), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    call require(allocated(self%estimators_),'bagging estimator is not fitted')
    call require(size(x,1)>0.and.size(x,2)>0,'bagging queries must be nonempty')
    call require(all(ieee_is_finite(x).or.ieee_is_nan(x)),'bagging queries contain infinity')
    call require(size(x,2)==self%n_features_in_,'bagging query feature count mismatch')
  end subroutine

  function member_probability(model,x,k,oob) result(value)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: k
    logical, intent(in), optional :: oob
    real(dp), allocatable :: value(:,:),raw(:,:)
    integer, allocatable :: classes(:),predicted(:)
    integer :: j,i
    allocate(value(size(x,1),k)); value=0
    if(base_proba(model)) then
      raw=probabilities(model,x); classes=base_classes(model)
      call require(size(raw,2)==size(classes),'bagging base probability shape mismatch')
      if(present(oob)) then
        if(oob) then
          ! Literal sklearn 1.6.1 OOB addition uses NumPy broadcasting.
          call require(size(raw,2)==k.or.size(raw,2)==1,'bagging OOB probability class shape mismatch')
          do j=1,k
            value(:,j)=raw(:,min(j,size(raw,2)))
          end do
          return
        end if
      end if
      call require(all(classes>=0).and.all(classes<k),'bagging base returned an invalid encoded class')
      do j=1,size(classes)
        value(:,classes(j)+1)=raw(:,j)
      end do
    else
      predicted=labels(model,x)
      call require(all(predicted>=0).and.all(predicted<k),'bagging base returned an invalid encoded class')
      do i=1,size(predicted)
        value(i,predicted(i)+1)=1
      end do
    end if
  end function

  function classifier_probability(self,x) result(value)
    class(bagging_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: value(:,:),raw(:,:)
    integer :: b,i,j
    call query(self,x); allocate(value(size(x,1),self%n_classes_)); value=0
    do b=1,size(self%estimators_)
      raw=member_probability(self%estimators_(b)%estimator,x(:,self%estimators_features_(b)%values),self%n_classes_)
      do j=1,size(value,2)
        do i=1,size(value,1)
          value(i,j)=ieee_difference(value(i,j),-raw(i,j))
        end do
      end do
    end do
    value=value/size(self%estimators_)
  end function

  function classifier_predict(self,x) result(value)
    class(bagging_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: value(:)
    real(dp), allocatable :: response(:,:)
    integer :: i
    response=self%predict_proba(x); allocate(value(size(x,1)))
    do i=1,size(value)
      value(i)=self%classes_(ieee_argmax(response(i,:)))
    end do
  end function

  function regressor_predict(self,x) result(value)
    class(bagging_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: value(:),raw(:)
    integer :: b,i
    call query(self,x); allocate(value(size(x,1))); value=0
    do b=1,size(self%estimators_)
      raw=regression(self%estimators_(b)%estimator,x(:,self%estimators_features_(b)%values))
      do i=1,size(value)
        value(i)=ieee_difference(value(i),-raw(i))
      end do
    end do
    value=value/size(self%estimators_)
  end function

  subroutine classifier_oob(self,x,y)
    class(bagging_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    logical :: mask(size(y))
    integer, allocatable :: ids(:)
    real(dp), allocatable :: raw(:,:),totals(:,:)
    real(dp) :: mass
    integer :: b,i,j,correct
    allocate(totals(size(y),self%n_classes_)); totals=0
    do b=1,size(self%estimators_)
      mask=.true.; mask(self%estimators_samples_(b)%values)=.false.
      ids=pack([(i,i=1,size(y))],mask)
      if(size(ids)==0) cycle
      raw=member_probability(self%estimators_(b)%estimator,x(ids,self%estimators_features_(b)%values), &
        self%n_classes_,.true.)
      do j=1,self%n_classes_
        do i=1,size(ids)
          totals(ids(i),j)=ieee_difference(totals(ids(i),j),-raw(i,j))
        end do
      end do
    end do
    allocate(self%oob_decision_function_(size(y),self%n_classes_)); correct=0
    do i=1,size(y)
      if(ieee_argmax(totals(i,:))-1==y(i)) correct=correct+1
      mass=ieee_total(totals(i,:))
      do j=1,self%n_classes_
        self%oob_decision_function_(i,j)=ieee_quotient(totals(i,j),mass)
      end do
    end do
    self%oob_score_=real(correct,dp)/size(y)
  end subroutine

  subroutine regressor_oob(self,x,y)
    class(bagging_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    logical :: mask(size(y))
    integer, allocatable :: ids(:)
    integer :: counts(size(y)),b,i
    real(dp), allocatable :: raw(:)
    allocate(self%oob_prediction_(size(y))); self%oob_prediction_=0; counts=0
    do b=1,size(self%estimators_)
      mask=.true.; mask(self%estimators_samples_(b)%values)=.false.
      ids=pack([(i,i=1,size(y))],mask)
      if(size(ids)==0) cycle
      raw=regression(self%estimators_(b)%estimator,x(ids,self%estimators_features_(b)%values))
      do i=1,size(ids)
        self%oob_prediction_(ids(i))=ieee_difference(self%oob_prediction_(ids(i)),-raw(i))
        counts(ids(i))=counts(ids(i))+1
      end do
    end do
    self%oob_prediction_=self%oob_prediction_/max(counts,1)
    self%oob_score_=regression_score(y,self%oob_prediction_)
  end subroutine

  subroutine score_weights(w,n)
    real(dp), intent(in), optional :: w(:)
    integer, intent(in) :: n
    real(dp) :: mass
    if(.not.present(w)) return
    call require(size(w)==n,'bagging score weight length mismatch')
    call require(all(ieee_is_finite(w)),'bagging score weights must be finite')
    mass=ieee_total(w)
    if(.not.ieee_is_nan(mass)) call require(mass/=0,'bagging score weights must have a nonzero sum')
  end subroutine

  real(dp) function classifier_score(self,x,y,sample_weight) result(value)
    class(bagging_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    integer, allocatable :: predicted(:)
    predicted=self%predict(x)
    call require(size(y)==size(predicted),'bagging score target length mismatch')
    call score_weights(sample_weight,size(y))
    value=ieee_mean(merge(1._dp,0._dp,predicted==y),sample_weight)
  end function

  real(dp) function regression_score(y,predicted,w) result(value)
    real(dp), intent(in) :: y(:),predicted(:)
    real(dp), intent(in), optional :: w(:)
    real(dp) :: residual(size(y)),centered(size(y)),center,num,den,difference
    integer :: i
    call require(size(y)==size(predicted),'bagging score target length mismatch')
    call require(all(ieee_is_finite(y)),'bagging score targets must be finite')
    call require(all(ieee_is_finite(predicted)),'bagging score predictions must be finite')
    if(size(y)<2) then
      if(present(w)) then
        call require(size(w)==size(y),'bagging score weight length mismatch')
        call require(all(ieee_is_finite(w)),'bagging score weights must be finite')
      end if
      value=ieee_value(0._dp,ieee_quiet_nan); return
    end if
    call score_weights(w,size(y)); center=ieee_mean(y,w)
    do i=1,size(y)
      difference=ieee_difference(y(i),predicted(i)); residual(i)=ieee_product(difference,difference)
      difference=ieee_difference(y(i),center); centered(i)=ieee_product(difference,difference)
    end do
    num=ieee_mean(residual,w); den=ieee_mean(centered,w)
    value=ieee_value(0._dp,ieee_quiet_nan)
    if(ieee_is_nan(num).or.ieee_is_nan(den)) return
    value=1
    if(num/=0) then
      value=0
      if(den/=0) value=ieee_difference(1._dp,ieee_quotient(num,den))
    end if
  end function

  real(dp) function regressor_score(self,x,y,sample_weight) result(value)
    class(bagging_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: predicted(:)
    predicted=self%predict(x); value=regression_score(y,predicted,sample_weight)
  end function
end module
