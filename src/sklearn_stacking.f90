! Stacking: sklearn 1.6.1 ensemble/_stacking.py and cross_val_predict (BSD-3-Clause).
! Dense native adaptation; retained notices are in THIRD_PARTY_NOTICES.md and LICENSES.
module sklearn_stacking
  use sklearn_sgd_clone, only: reset_sgd_fit
  use sklearn_common, only: dp,require,check_xy,check_x,unique_labels,label_index
  use sklearn_voting, only: stacking_estimator=>voting_estimator
  use sklearn_base_dispatch, only: base_supported,base_weighted,base_proba,base_decision,base_classes, &
    train_classifier,train_regressor,labels,regression,probabilities,decisions
  use sklearn_model_selection, only: kfold,index_split,ridge_cv
  use sklearn_linear_model, only: logistic_regression,linear_regression,ridge
  use sklearn_tree, only: decision_tree_classifier,decision_tree_regressor
  use sklearn_weighted_linear, only: weighted_linear
  use sklearn_composite_scores, only: score_weights,regression_score
  use sklearn_ieee_math, only: ieee_mean
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan
  implicit none
  private
  public :: stacking_classifier,stacking_regressor,stacking_estimator
  type :: model_slot
    class(*), allocatable :: estimator
  end type
  type :: stacking_storage
    integer :: cv=5,n_features_in_=0,n_features_out_=0
    logical :: passthrough=.false.,prefit=.false.
    type(stacking_estimator), allocatable :: estimators_(:)
    class(*), allocatable :: final_estimator_
    integer, allocatable :: active_indices_(:),n_feature_outs_(:)
    character(20), allocatable :: stack_method_(:)
    real(dp), allocatable :: training_meta_features_(:,:)
    type(index_split), allocatable :: cv_splits_(:)
    type(model_slot), allocatable :: cv_estimators_(:,:)
    logical, private :: fitted_=.false.,passthrough_fit_=.false.
  end type
  type, extends(stacking_storage) :: stacking_classifier
    character(20) :: stack_method='auto'
    integer, allocatable :: classes_(:)
    integer :: n_classes_=0
  contains
    procedure :: fit=>classifier_fit
    procedure :: transform=>classifier_transform
    procedure :: fit_transform=>classifier_fit_transform
    procedure :: predict=>classifier_predict
    procedure :: predict_proba=>classifier_proba
    procedure :: decision_function=>classifier_decision
    procedure :: score=>classifier_score
  end type
  type, extends(stacking_storage) :: stacking_regressor
  contains
    procedure :: fit=>regressor_fit
    procedure :: transform=>regressor_transform
    procedure :: fit_transform=>regressor_fit_transform
    procedure :: predict=>regressor_predict
    procedure :: score=>regressor_score
  end type
contains
  subroutine validate_names(estimators,active)
    type(stacking_estimator), intent(in) :: estimators(:)
    integer, allocatable, intent(out) :: active(:)
    integer :: i,j
    call require(size(estimators)>0,'stacking requires at least one estimator')
    do i=1,size(estimators)
      call require(len_trim(estimators(i)%name)>0,'stacking estimator name must be nonempty')
      call require(index(estimators(i)%name,'__')==0,'stacking estimator name cannot contain __')
      call require(.not.any(trim(estimators(i)%name)==[character(20)::'estimators','final_estimator','cv', &
        'stack_method','passthrough','prefit','n_jobs','verbose']),'stacking estimator name conflicts with a parameter')
      do j=1,i-1
        call require(estimators(i)%name/=estimators(j)%name,'stacking estimator names must be unique')
      end do
      if(.not.estimators(i)%drop) call require(allocated(estimators(i)%estimator),'active stacking estimator is missing')
    end do
    active=pack([(i,i=1,size(estimators))],.not.estimators%drop)
    call require(size(active)>0,'all stacking estimators are dropped')
  end subroutine

  logical function supports_weights(model) result(ok)
    class(*), intent(in) :: model
    ok=base_weighted(model)
    select type(model)
    type is(linear_regression)
      ok=.true.
    type is(ridge)
      ok=.true.
    end select
  end function

  subroutine fit_base(model,x,yc,yr,classifier,w)
    class(*), intent(inout) :: model
    real(dp), intent(in) :: x(:,:),yr(:)
    integer, intent(in) :: yc(:)
    logical, intent(in) :: classifier
    real(dp), intent(in), optional :: w(:)
    real(dp) :: ones(size(x,1))
    call require(base_supported(model,classifier),'unsupported native stacking base')
    call reset_sgd_fit(model)
    if(present(w)) call require(supports_weights(model),'stacking base has no native weighted fit')
    ! Trees use the source float32/RNG builder even for an omitted weight vector.
    ones=1
    select type(model)
    class is(decision_tree_classifier)
      if(present(w)) then
        call model%fit(x,yc,w)
      else
        call model%fit(x,yc,ones)
      end if
      return
    class is(decision_tree_regressor)
      if(present(w)) then
        call model%fit(x,yr,w)
      else
        call model%fit(x,yr,ones)
      end if
      return
    type is(linear_regression)
      if(present(w)) then
        call weighted_linear(model,x,yr,w,0._dp); return
      end if
    type is(ridge)
      if(present(w)) then
        call weighted_linear(model,x,yr,w,model%alpha); return
      end if
    end select
    if(classifier) then
      call train_classifier(model,x,yc,w)
    else
      call train_regressor(model,x,yr,w)
    end if
  end subroutine

  function method_for(model,requested) result(method)
    class(*), intent(in) :: model
    character(*), intent(in) :: requested
    character(20) :: method
    method=requested
    if(requested=='auto') then
      method='predict'
      if(base_decision(model)) method='decision_function'
      if(base_proba(model)) method='predict_proba'
    end if
    call require(any(method==[character(20)::'predict','predict_proba','decision_function']), &
      'unknown stacking response method')
    if(method=='predict_proba') call require(base_proba(model),'stacking base has no predict_proba')
    if(method=='decision_function') call require(base_decision(model),'stacking base has no decision_function')
  end function

  function response(model,x,method,k,enforce) result(value)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    character(*), intent(in) :: method
    integer, intent(in) :: k
    logical, intent(in) :: enforce
    real(dp), allocatable :: value(:,:),raw(:,:),aligned(:,:)
    integer, allocatable :: classes(:),predicted(:)
    integer :: j
    if(k==0) then
      allocate(raw(size(x,1),1)); raw(:,1)=regression(model,x)
    else if(method=='predict') then
      predicted=labels(model,x); allocate(raw(size(x,1),1)); raw(:,1)=real(predicted,dp)
    else
      if(method=='predict_proba') then
        raw=probabilities(model,x)
      else
        raw=decisions(model,x)
      end if
      if(enforce) then
        classes=base_classes(model)
        if(size(classes)/=k) then
          if(method=='decision_function') then
            call require(size(classes)>2,'stacking decision_function fold has too few classes')
            call require(size(raw,2)==size(classes),'stacking fold decision output shape is irregular')
          end if
          call require(all(classes>=0).and.all(classes<k),'stacking fold returned an invalid class')
          allocate(aligned(size(x,1),k)); aligned=0
          if(method=='decision_function') aligned=-huge(1._dp)
          do j=1,size(classes)
            aligned(:,classes(j)+1)=raw(:,j)
          end do
          call move_alloc(aligned,raw)
        end if
      end if
    end if
    if(method=='predict_proba'.and.k==2) then
      value=raw(:,2:)
    else
      call move_alloc(raw,value)
    end if
  end function

  function default_folds(n,cv,y) result(folds)
    integer, intent(in) :: n,cv,y(:)
    type(index_split), allocatable :: folds(:)
    type(kfold) :: kf
    integer, allocatable :: classes(:),counts(:),assignment(:),members(:)
    integer :: i,j,c,offset,position,width
    call require(cv>=2.and.cv<=n,'invalid stacking cv count')
    if(size(y)==0) then
      kf%n_splits=cv; folds=kf%split(n); return
    end if
    classes=[integer::]
    do i=1,n
      if(.not.any(classes==y(i))) classes=[classes,y(i)]
    end do
    counts=[(count(y==classes(i)),i=1,size(classes))]
    call require(.not.all(counts<cv),'stacking cv exceeds every class count')
    allocate(folds(cv),assignment(n)); offset=0
    do c=1,size(classes)
      members=pack([(i,i=1,n)],y==classes(c)); position=1
      do j=1,cv
        width=count(modulo(offset+[(i,i=0,counts(c)-1)],cv)==j-1)
        assignment(members(position:position+width-1))=j; position=position+width
      end do
      offset=offset+counts(c)
    end do
    do j=1,cv
      folds(j)%test=pack([(i,i=1,n)],assignment==j)
      folds(j)%train=pack([(i,i=1,n)],assignment/=j)
    end do
  end function

  subroutine validate_folds(folds,n)
    type(index_split), intent(in) :: folds(:)
    integer, intent(in) :: n
    integer :: seen(n),j,i
    call require(size(folds)>0,'stacking needs at least one CV split'); seen=0
    do j=1,size(folds)
      call require(allocated(folds(j)%train).and.allocated(folds(j)%test),'stacking split indices are missing')
      call require(size(folds(j)%train)>0.and.size(folds(j)%test)>0,'stacking split indices must be nonempty')
      call require(all(folds(j)%train>=1).and.all(folds(j)%train<=n).and. &
        all(folds(j)%test>=1).and.all(folds(j)%test<=n),'stacking split index is out of range')
      do i=1,size(folds(j)%test)
        seen(folds(j)%test(i))=seen(folds(j)%test(i))+1
      end do
    end do
    call require(all(seen==1),'stacking test indices must partition all rows exactly once')
  end subroutine

  subroutine fit_stack(self,x,yc,yr,estimators,final_estimator,w,splits,requested,k)
    class(stacking_storage), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),yr(:)
    integer, intent(in) :: yc(:),k
    type(stacking_estimator), intent(in) :: estimators(:)
    class(*), intent(in), optional :: final_estimator
    real(dp), intent(in), optional :: w(:)
    type(index_split), intent(in), optional :: splits(:)
    character(*), intent(in) :: requested
    type(logistic_regression) :: default_c
    type(ridge_cv) :: default_r
    type(stacking_estimator) :: prototype
    real(dp), allocatable :: raw(:,:),meta(:,:),fold_w(:)
    integer :: b,f,first,last,n,total
    logical :: classifier
    self%fitted_=.false.; n=size(x,1); classifier=k>0
    call validate_names(estimators,self%active_indices_)
    if(present(w)) then
      call require(size(w)==n,'stacking sample weight length mismatch')
      call require(all(ieee_is_finite(w)),'stacking sample weights must be finite')
    end if
    if(allocated(self%estimators_)) deallocate(self%estimators_)
    if(allocated(self%final_estimator_)) deallocate(self%final_estimator_)
    if(allocated(self%stack_method_)) deallocate(self%stack_method_)
    if(allocated(self%cv_estimators_)) deallocate(self%cv_estimators_)
    if(allocated(self%cv_splits_)) deallocate(self%cv_splits_)
    allocate(self%estimators_(size(self%active_indices_)),self%stack_method_(size(self%active_indices_)))
    self%n_feature_outs_=self%active_indices_*0
    do b=1,size(self%estimators_)
      self%estimators_(b)=estimators(self%active_indices_(b))
      call require(base_supported(self%estimators_(b)%estimator,classifier),'unsupported native stacking base')
      if(.not.self%prefit) call fit_base(self%estimators_(b)%estimator,x,yc,yr,classifier,w)
      if(classifier) then
        self%stack_method_(b)=method_for(self%estimators_(b)%estimator,requested)
      else
        self%stack_method_(b)='predict'
      end if
      raw=response(self%estimators_(b)%estimator,x,self%stack_method_(b),k,.false.)
      self%n_feature_outs_(b)=size(raw,2)
    end do
    total=sum(self%n_feature_outs_); self%passthrough_fit_=self%passthrough
    if(self%passthrough_fit_) total=total+size(x,2)
    call require(total>0,'stacking meta design has no features')
    allocate(meta(n,total)); meta=0
    if(.not.self%prefit) then
      if(present(splits)) then
        self%cv_splits_=splits
      else
        self%cv_splits_=default_folds(n,self%cv,yc)
      end if
      call validate_folds(self%cv_splits_,n)
      allocate(self%cv_estimators_(size(self%cv_splits_),size(self%estimators_)))
    else
      call require(.not.present(splits),'stacking prefit cannot accept CV splits')
    end if
    first=1
    do b=1,size(self%estimators_)
      last=first+self%n_feature_outs_(b)-1
      if(self%prefit) then
        raw=response(self%estimators_(b)%estimator,x,self%stack_method_(b),k,.false.)
        meta(:,first:last)=raw
      else
        ! A scalar descriptor also avoids compiler issues with complex polymorphic SOURCE expressions.
        prototype=estimators(self%active_indices_(b))
        do f=1,size(self%cv_splits_)
          associate(plan=>self%cv_splits_(f))
            allocate(self%cv_estimators_(f,b)%estimator,source=prototype%estimator)
            if(present(w)) then
              fold_w=w(plan%train)
              if(classifier) then
                call fit_base(self%cv_estimators_(f,b)%estimator,x(plan%train,:),yc(plan%train),[real(dp)::],.true.,fold_w)
              else
                call fit_base(self%cv_estimators_(f,b)%estimator,x(plan%train,:),[integer::],yr(plan%train),.false.,fold_w)
              end if
            else
              if(classifier) then
                call fit_base(self%cv_estimators_(f,b)%estimator,x(plan%train,:),yc(plan%train),[real(dp)::],.true.)
              else
                call fit_base(self%cv_estimators_(f,b)%estimator,x(plan%train,:),[integer::],yr(plan%train),.false.)
              end if
            end if
            raw=response(self%cv_estimators_(f,b)%estimator,x(plan%test,:),self%stack_method_(b),k,classifier)
            call require(size(raw,2)==self%n_feature_outs_(b),'stacking fold response width mismatch')
            meta(plan%test,first:last)=raw
          end associate
        end do
      end if
      first=last+1
    end do
    if(self%passthrough_fit_) meta(:,first:)=x
    call check_x(meta)
    call require(maxval(abs(meta))<=sqrt(huge(1._dp))/sqrt(real(max(n,total),dp))/4, &
      'stacking meta features exceed working range')
    if(present(final_estimator)) then
      allocate(self%final_estimator_,source=final_estimator)
    else if(classifier) then
      default_c%tol=1.e-10_dp; default_c%max_iter=500
      allocate(self%final_estimator_,source=default_c)
    else
      default_r%cv=0; allocate(self%final_estimator_,source=default_r)
    end if
    call fit_base(self%final_estimator_,meta,yc,yr,classifier,w)
    self%training_meta_features_=meta; self%n_features_in_=size(x,2); self%n_features_out_=total
    self%fitted_=.true.
  end subroutine

  subroutine classifier_fit(self,x,y,estimators,final_estimator,sample_weight,splits)
    class(stacking_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    type(stacking_estimator), intent(in) :: estimators(:)
    class(*), intent(in), optional :: final_estimator
    real(dp), intent(in), optional :: sample_weight(:)
    type(index_split), intent(in), optional :: splits(:)
    integer, allocatable :: encoded(:)
    integer :: i
    call check_xy(x,size(y)); self%classes_=unique_labels(y); self%n_classes_=size(self%classes_)
    encoded=[(label_index(self%classes_,y(i))-1,i=1,size(y))]
    call fit_stack(self,x,encoded,[real(dp)::],estimators,final_estimator,sample_weight,splits, &
      self%stack_method,self%n_classes_)
  end subroutine

  subroutine regressor_fit(self,x,y,estimators,final_estimator,sample_weight,splits)
    class(stacking_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    type(stacking_estimator), intent(in) :: estimators(:)
    class(*), intent(in), optional :: final_estimator
    real(dp), intent(in), optional :: sample_weight(:)
    type(index_split), intent(in), optional :: splits(:)
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    call fit_stack(self,x,[integer::],y,estimators,final_estimator,sample_weight,splits,'predict',0)
  end subroutine

  function transform(self,x,k) result(value)
    class(stacking_storage), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: k
    real(dp), allocatable :: value(:,:),raw(:,:)
    integer :: b,first,last
    call require(self%fitted_,'stacking estimator is not fitted')
    call require(size(x,1)>0.and.size(x,2)>0,'stacking queries must be nonempty')
    call require(all(ieee_is_finite(x).or.ieee_is_nan(x)),'stacking queries contain infinity')
    call require(size(x,2)==self%n_features_in_,'stacking query feature count mismatch')
    allocate(value(size(x,1),self%n_features_out_)); first=1
    do b=1,size(self%estimators_)
      raw=response(self%estimators_(b)%estimator,x,self%stack_method_(b),k,.false.)
      call require(size(raw,2)==self%n_feature_outs_(b),'stacking transform response width mismatch')
      last=first+self%n_feature_outs_(b)-1; value(:,first:last)=raw; first=last+1
    end do
    if(self%passthrough_fit_) value(:,first:)=x
  end function
  function classifier_transform(self,x) result(value)
    class(stacking_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: value(:,:)
    value=transform(self,x,self%n_classes_)
  end function
  function regressor_transform(self,x) result(value)
    class(stacking_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: value(:,:)
    value=transform(self,x,0)
  end function
  function classifier_predict(self,x) result(value)
    class(stacking_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: value(:),encoded(:)
    real(dp), allocatable :: meta(:,:)
    meta=self%transform(x); encoded=labels(self%final_estimator_,meta)
    call require(all(encoded>=0).and.all(encoded<self%n_classes_),'stacking final estimator returned an invalid class')
    value=self%classes_(encoded+1)
  end function
  function classifier_proba(self,x) result(value)
    class(stacking_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: value(:,:),meta(:,:)
    meta=self%transform(x)
    call require(base_proba(self%final_estimator_),'stacking final estimator has no predict_proba')
    value=probabilities(self%final_estimator_,meta)
  end function
  function classifier_decision(self,x) result(value)
    class(stacking_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: value(:,:),meta(:,:)
    meta=self%transform(x)
    call require(base_decision(self%final_estimator_),'stacking final estimator has no decision_function')
    value=decisions(self%final_estimator_,meta)
  end function
  function regressor_predict(self,x) result(value)
    class(stacking_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: value(:),meta(:,:)
    meta=self%transform(x); value=regression(self%final_estimator_,meta)
  end function
  real(dp) function classifier_score(self,x,y,sample_weight) result(value)
    class(stacking_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    integer, allocatable :: predicted(:)
    predicted=self%predict(x); call require(size(y)==size(predicted),'stacking score target length mismatch')
    call score_weights(sample_weight,size(y),'stacking')
    value=ieee_mean(merge(1._dp,0._dp,predicted==y),sample_weight)
  end function
  real(dp) function regressor_score(self,x,y,sample_weight) result(value)
    class(stacking_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: predicted(:)
    predicted=self%predict(x); value=regression_score(y,predicted,sample_weight,'stacking')
  end function
  function classifier_fit_transform(self,x,y,estimators,final_estimator,sample_weight,splits) result(value)
    class(stacking_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    type(stacking_estimator), intent(in) :: estimators(:)
    class(*), intent(in), optional :: final_estimator
    real(dp), intent(in), optional :: sample_weight(:)
    type(index_split), intent(in), optional :: splits(:)
    real(dp), allocatable :: value(:,:)
    call self%fit(x,y,estimators,final_estimator,sample_weight,splits); value=self%transform(x)
  end function
  function regressor_fit_transform(self,x,y,estimators,final_estimator,sample_weight,splits) result(value)
    class(stacking_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    type(stacking_estimator), intent(in) :: estimators(:)
    class(*), intent(in), optional :: final_estimator
    real(dp), intent(in), optional :: sample_weight(:)
    type(index_split), intent(in), optional :: splits(:)
    real(dp), allocatable :: value(:,:)
    call self%fit(x,y,estimators,final_estimator,sample_weight,splits); value=self%transform(x)
  end function
end module
