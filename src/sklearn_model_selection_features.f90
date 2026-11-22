! SelectFromModel contracts: sklearn 1.6.1 feature_selection/_from_model.py
! and _base.py (BSD-3-Clause). Native adapters are original MIT contributions.
module sklearn_model_selection_features
  use sklearn_common, only: dp,require,check_x,check_xy,argsort
  use sklearn_feature_extraction, only: feature_string
  use sklearn_linear_model, only: linear_regression,ridge,elastic_net,lasso,elastic_net_cv,lasso_cv, &
    logistic_regression,sgd_regressor,sgd_classifier,multi_task_elastic_net,multi_task_lasso, &
    multi_task_elastic_net_cv,multi_task_lasso_cv,perceptron,passive_aggressive_classifier,ridge_classifier
  use sklearn_tree, only: decision_tree_regressor,decision_tree_classifier
  use sklearn_ensemble, only: random_forest_regressor,random_forest_classifier, &
    gradient_boosting_regressor,gradient_boosting_classifier
  use sklearn_base_dispatch, only: train_regressor,train_classifier,base_supported,base_fitted
  use sklearn_weighted_linear, only: weighted_linear
  use sklearn_ieee_math, only: ieee_mean,ieee_product
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan
  implicit none
  private
  public :: select_from_model
  abstract interface
    subroutine model_importance_function(model,values,matrix)
      import dp
      class(*), intent(in) :: model
      real(dp), allocatable, intent(out) :: values(:,:)
      logical, intent(out) :: matrix
    end subroutine
    integer function feature_cap_function(x) result(n)
      import dp
      real(dp), intent(in) :: x(:,:)
    end function
  end interface
  type :: select_from_model
    class(*), allocatable :: estimator,estimator_
    logical :: prefit=.false.
    character(80) :: threshold='auto'
    real(dp) :: numeric_threshold=0,norm_order=1
    character(24) :: importance_getter='auto'
    integer :: max_features=-1,max_features_=-1,n_features_in_=0
    procedure(model_importance_function), pointer, nopass :: importance_func=>null()
    procedure(feature_cap_function), pointer, nopass :: max_features_func=>null()
    type(feature_string), allocatable :: feature_names_in_(:)
  contains
    procedure :: fit_prefit,fit_regression,fit_classification,fit_multioutput
    generic :: fit=>fit_prefit,fit_regression,fit_classification,fit_multioutput
    procedure :: partial_fit_regression,partial_fit_classification
    generic :: partial_fit=>partial_fit_regression,partial_fit_classification
    procedure :: fit_transform_regression,fit_transform_classification,fit_transform_prefit,fit_transform_multioutput
    generic :: fit_transform=>fit_transform_regression,fit_transform_classification, &
      fit_transform_prefit,fit_transform_multioutput
    procedure :: transform, inverse_transform,get_support,get_support_indices,get_feature_names_out
    procedure :: threshold_value,feature_importances
  end type
contains
  subroutine controls(self)
    class(select_from_model), intent(in) :: self
    call require(.not.ieee_is_nan(self%norm_order),'SelectFromModel norm_order is NaN')
    if(ieee_is_finite(self%norm_order)) then
      call require(self%norm_order/=0,'SelectFromModel norm_order must be nonzero')
      call require(modulo(self%norm_order,1._dp)==0,'SelectFromModel norm_order must be integral or infinite')
    end if
    call require(self%max_features>=-1,'SelectFromModel max_features must be nonnegative or -1')
    call require(any(self%importance_getter==[character(24)::'auto','coef_','feature_importances_','callback']), &
      'SelectFromModel invalid importance_getter')
    if(self%importance_getter=='callback') then
      call require(associated(self%importance_func),'SelectFromModel importance callback is unassociated')
    else
      call require(.not.associated(self%importance_func),'SelectFromModel set importance_getter=callback for callback')
    end if
  end subroutine
  subroutine check_prefitted(model)
    class(*), intent(in) :: model
    logical :: fitted
    fitted=base_fitted(model)
    select type(model)
    class is(linear_regression)
      fitted=allocated(model%coef_)
    class is(multi_task_elastic_net)
      fitted=allocated(model%coef_)
    type is(sgd_regressor)
      fitted=allocated(model%coef_)
    class is(decision_tree_regressor)
      fitted=allocated(model%nodes_)
    class is(random_forest_regressor)
      fitted=allocated(model%estimators_)
    class is(random_forest_classifier)
      fitted=allocated(model%estimators_)
    type is(gradient_boosting_regressor)
      fitted=allocated(model%estimators_)
    type is(gradient_boosting_classifier)
      fitted=allocated(model%estimators_)
    end select
    call require(fitted,'SelectFromModel prefit estimator is not fitted or unsupported')
  end subroutine
  subroutine raw_importances(self,model,values,matrix)
    class(select_from_model), intent(in) :: self
    class(*), intent(in) :: model
    real(dp), allocatable, intent(out) :: values(:,:)
    logical, intent(out) :: matrix
    real(dp), allocatable :: vector(:)
    logical :: coefficients
    matrix=.false.;coefficients=.true.
    if(self%importance_getter=='callback') then
      call self%importance_func(model,values,matrix)
    else
      select type(model)
      class is(linear_regression)
        if(allocated(model%coef_)) vector=model%coef_
      class is(logistic_regression)
        if(allocated(model%coef_)) values=model%coef_
        matrix=.true.
      type is(sgd_regressor)
        if(allocated(model%coef_)) vector=model%coef_
      type is(sgd_classifier)
        if(allocated(model%coef_)) values=model%coef_
        matrix=.true.
      class is(multi_task_elastic_net)
        if(allocated(model%coef_)) values=model%coef_
        matrix=.true.
      type is(perceptron)
        if(allocated(model%coef_)) values=model%coef_
        matrix=.true.
      type is(passive_aggressive_classifier)
        if(allocated(model%coef_)) values=model%coef_
        matrix=.true.
      class is(ridge_classifier)
        if(allocated(model%coef_)) values=model%coef_
        matrix=.true.
      class is(decision_tree_regressor)
        if(allocated(model%feature_importances_)) vector=model%feature_importances_
        coefficients=.false.
      class is(decision_tree_classifier)
        if(allocated(model%feature_importances_)) vector=model%feature_importances_
        coefficients=.false.
      class is(random_forest_regressor)
        if(allocated(model%feature_importances_)) vector=model%feature_importances_
        coefficients=.false.
      class is(random_forest_classifier)
        if(allocated(model%feature_importances_)) vector=model%feature_importances_
        coefficients=.false.
      type is(gradient_boosting_regressor)
        if(allocated(model%feature_importances_)) vector=model%feature_importances_
        coefficients=.false.
      type is(gradient_boosting_classifier)
        if(allocated(model%feature_importances_)) vector=model%feature_importances_
        coefficients=.false.
      class default
        call require(.false.,'SelectFromModel unsupported importance estimator; use a callback')
      end select
      if(self%importance_getter=='coef_') &
        call require(coefficients,'SelectFromModel estimator has no coef_')
      if(self%importance_getter=='feature_importances_') &
        call require(.not.coefficients,'SelectFromModel estimator has no feature_importances_')
      if(allocated(vector)) values=reshape(vector,[1,size(vector)])
    end if
    call require(allocated(values),'SelectFromModel importance estimator is not fitted')
    call require(size(values,1)>0.and.size(values,2)>0,'SelectFromModel importances must be nonempty')
    call require(all(ieee_is_finite(values)),'SelectFromModel importances must be finite')
    if(.not.matrix) call require(size(values,1)==1,'SelectFromModel vector importances must have one row')
  end subroutine
  function feature_importances(self) result(scores)
    class(select_from_model), intent(in) :: self
    real(dp), allocatable :: scores(:),values(:,:)
    logical :: matrix
    real(dp) :: order,scale,total
    integer :: j
    call controls(self)
    if(allocated(self%estimator_)) then
      call raw_importances(self,self%estimator_,values,matrix)
    else
      call require(self%prefit,'SelectFromModel is not fitted')
      call require(allocated(self%estimator),'SelectFromModel estimator is unallocated')
      call check_prefitted(self%estimator)
      call raw_importances(self,self%estimator,values,matrix)
    end if
    if(self%n_features_in_>0) call require(size(values,2)==self%n_features_in_, &
      'SelectFromModel importance feature count mismatch')
    allocate(scores(size(values,2)));values=abs(values);order=self%norm_order
    if(.not.matrix) then
      scores=values(1,:);return
    end if
    do j=1,size(scores)
      if(.not.ieee_is_finite(order)) then
        if(order>0) then
          scores(j)=maxval(values(:,j))
        else
          scores(j)=minval(values(:,j))
        end if
      else if(order<0.and.any(values(:,j)==0)) then
        scores(j)=0
      else
        if(order>0) then
          scale=maxval(values(:,j))
        else
          scale=minval(values(:,j))
        end if
        scores(j)=0
        if(scale>0) then
          ! Scaling avoids intermediate overflow in positive/negative p-norms.
          if(order>0) then
            total=sum((values(:,j)/scale)**order)
          else
            total=sum((scale/values(:,j))**(-order))
          end if
          scores(j)=ieee_product(scale,total**(1/order))
        end if
      end if
    end do
    call require(all(ieee_is_finite(scores)),'SelectFromModel norm overflow')
  end function
  logical function l1_default(model) result(l1)
    class(*), intent(in) :: model
    l1=.false.
    select type(model)
    type is(lasso)
      l1=.true.
    type is(lasso_cv)
      l1=.true.
    class is(elastic_net_cv)
      l1=abs(model%l1_ratio_-1)<=1.001e-5_dp
    class is(elastic_net)
      l1=abs(model%l1_ratio-1)<=1.001e-5_dp
    type is(multi_task_lasso)
      l1=.true.
    type is(multi_task_lasso_cv)
      l1=.true.
    class is(multi_task_elastic_net_cv)
      l1=abs(model%l1_ratio_-1)<=1.001e-5_dp
    class is(multi_task_elastic_net)
      l1=abs(model%l1_ratio-1)<=1.001e-5_dp
    type is(sgd_regressor)
      l1=model%penalty=='l1'
    type is(sgd_classifier)
      l1=model%penalty=='l1'
    end select
  end function
  real(dp) function calculate_threshold(self,scores) result(value)
    class(select_from_model), intent(in) :: self
    real(dp), intent(in) :: scores(:)
    character(:), allocatable :: reference,text,factor_text
    real(dp) :: factor
    integer :: star,status,p
    integer, allocatable :: indices(:)
    text=trim(self%threshold)
    if(text=='value') then
      call require(.not.ieee_is_nan(self%numeric_threshold),'SelectFromModel numeric threshold is NaN')
      value=self%numeric_threshold;return
    end if
    if(text=='auto') then
      if(allocated(self%estimator_)) then
        if(l1_default(self%estimator_)) then
          value=1.e-5_dp;return
        end if
      else if(allocated(self%estimator)) then
        if(l1_default(self%estimator)) then
          value=1.e-5_dp;return
        end if
      end if
      text='mean'
    end if
    factor=1;reference=text;star=index(text,'*')
    if(star>0) then
      call require(star>1.and.star<len(text),'SelectFromModel malformed scaled threshold')
      factor_text=trim(adjustl(text(:star-1)))
      call require(scan(factor_text,' '//achar(9)//',/dD')==0,'SelectFromModel invalid threshold scale')
      read(factor_text,*,iostat=status) factor
      call require(status==0,'SelectFromModel invalid threshold scale')
      call require(ieee_is_finite(factor),'SelectFromModel threshold scale must be finite')
      reference=trim(adjustl(text(star+1:)))
    end if
    select case(reference)
    case('mean')
      value=ieee_mean(scores)
    case('median')
      indices=argsort(scores);p=size(scores)
      value=scores(indices((p+1)/2))
      if(mod(p,2)==0) value=ieee_mean([value,scores(indices(p/2+1))])
    case default
      call require(.false.,'SelectFromModel threshold must be auto, value, mean, median or a scaled mean/median')
    end select
    value=ieee_product(factor,value)
  end function
  real(dp) function threshold_value(self) result(value)
    class(select_from_model), intent(in) :: self
    real(dp), allocatable :: scores(:)
    scores=self%feature_importances();value=calculate_threshold(self,scores)
  end function
  function get_support(self) result(mask)
    class(select_from_model), intent(in) :: self
    logical, allocatable :: mask(:)
    real(dp), allocatable :: scores(:)
    integer, allocatable :: indices(:)
    integer :: cap
    scores=self%feature_importances();allocate(mask(size(scores)));mask=.true.
    cap=self%max_features
    if(allocated(self%estimator_)) then
      cap=self%max_features_
    else
      call require(.not.associated(self%max_features_func),'SelectFromModel callable max_features requires fit')
    end if
    if(self%max_features/=-1.or.associated(self%max_features_func)) then
      call require(cap>=0.and.cap<=size(scores),'SelectFromModel max_features exceeds feature count')
      indices=argsort(-scores);mask=.false.;mask(indices(:cap))=.true.
    end if
    where(scores<calculate_threshold(self,scores)) mask=.false.
  end function
  function get_support_indices(self) result(indices)
    class(select_from_model), intent(in) :: self
    integer, allocatable :: indices(:)
    logical, allocatable :: mask(:)
    integer :: i
    mask=self%get_support();indices=pack([(i,i=1,size(mask))],mask)
  end function
  subroutine prepare(self,x,estimator,feature_names,incremental)
    class(select_from_model), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    class(*), optional, intent(in) :: estimator
    type(feature_string), optional, intent(in) :: feature_names(:)
    logical, optional, intent(in) :: incremental
    logical :: first
    integer :: i
    type(feature_string), allocatable :: names(:)
    call check_x(x);call controls(self)
    first=.true.
    if(present(incremental)) then
      if(incremental) first=.not.allocated(self%estimator_)
    end if
    if(.not.first) then
      call require(.not.present(estimator),'SelectFromModel cannot replace estimator during partial_fit')
      call require(size(x,2)==self%n_features_in_,'SelectFromModel partial_fit feature count mismatch')
      if(present(feature_names)) names=self%get_feature_names_out(feature_names)
      return
    end if
    if(present(estimator)) then
      if(allocated(self%estimator)) deallocate(self%estimator)
      allocate(self%estimator,source=estimator)
    end if
    call require(allocated(self%estimator),'SelectFromModel estimator is unallocated')
    if(allocated(self%estimator_)) deallocate(self%estimator_)
    allocate(self%estimator_,source=self%estimator)
    if(self%prefit) then
      call check_prefitted(self%estimator_)
    else
      call reset_incremental(self%estimator_)
    end if
    self%n_features_in_=size(x,2);self%max_features_=self%max_features
    if(associated(self%max_features_func)) self%max_features_=self%max_features_func(x)
    call require(self%max_features_>=-1.and.self%max_features_<=size(x,2), &
      'SelectFromModel max_features exceeds feature count')
    if(associated(self%max_features_func)) call require(self%max_features_>=0, &
      'SelectFromModel max_features callback must return a nonnegative count')
    if(allocated(self%feature_names_in_)) deallocate(self%feature_names_in_)
    if(present(feature_names)) then
      call require(size(feature_names)==size(x,2),'SelectFromModel feature name count mismatch')
      do i=1,size(feature_names)
        call require(allocated(feature_names(i)%text),'SelectFromModel feature name is unallocated')
      end do
      self%feature_names_in_=feature_names
    end if
  end subroutine
  subroutine fit_prefit(self,x,estimator,feature_names)
    class(select_from_model), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    class(*), optional, intent(in) :: estimator
    type(feature_string), optional, intent(in) :: feature_names(:)
    call require(self%prefit,'SelectFromModel requires targets unless prefit=true')
    call prepare(self,x,estimator,feature_names)
  end subroutine
  logical function is_classifier(model) result(classifier)
    class(*), intent(in) :: model
    classifier=base_supported(model,.true.)
    select type(model)
    class is(random_forest_classifier)
      classifier=.true.
    type is(gradient_boosting_classifier)
      classifier=.true.
    end select
  end function
  subroutine fit_regressor_model(model,x,y,sample_weight)
    class(*), intent(inout) :: model
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    real(dp) :: weights(size(y))
    select type(model)
    type is(linear_regression)
      call weighted_fit(0._dp)
    type is(ridge)
      call weighted_fit(model%alpha)
    class is(random_forest_regressor)
      call require(.not.present(sample_weight),'SelectFromModel base has no native weighted fit')
      call model%fit(x,y)
    type is(gradient_boosting_regressor)
      call require(.not.present(sample_weight),'SelectFromModel base has no native weighted fit')
      call model%fit(x,y)
    class default
      call train_regressor(model,x,y,sample_weight)
    end select
  contains
    subroutine weighted_fit(alpha)
      real(dp), intent(in) :: alpha
      weights=1
      if(present(sample_weight)) then
        call require(size(sample_weight)==size(y),'SelectFromModel weight count mismatch')
        weights=sample_weight
      end if
      call require(all(ieee_is_finite(weights)),'SelectFromModel weights must be finite')
      select type(model)
      class is(linear_regression)
        call weighted_linear(model,x,y,weights,alpha)
      end select
    end subroutine
  end subroutine
  subroutine fit_classifier_model(model,x,y,sample_weight)
    class(*), intent(inout) :: model
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    select type(model)
    class is(random_forest_classifier)
      call require(.not.present(sample_weight),'SelectFromModel base has no native weighted fit')
      call model%fit(x,y)
    type is(gradient_boosting_classifier)
      call require(.not.present(sample_weight),'SelectFromModel base has no native weighted fit')
      call model%fit(x,y)
    class default
      call train_classifier(model,x,y,sample_weight)
    end select
  end subroutine
  subroutine reset_incremental(model)
    class(*), intent(inout) :: model
    type(sgd_regressor) :: reg
    type(sgd_classifier) :: clf
    select type(model)
    type is(sgd_regressor)
      reg%loss=model%loss;reg%penalty=model%penalty;reg%learning_rate=model%learning_rate
      reg%alpha=model%alpha;reg%eta0=model%eta0;reg%power_t=model%power_t;reg%epsilon=model%epsilon;reg%tol=model%tol
      reg%max_iter=model%max_iter;reg%n_iter_no_change=model%n_iter_no_change;reg%average=model%average
      reg%random_state=model%random_state;reg%fit_intercept=model%fit_intercept;reg%shuffle=model%shuffle
      model=reg
    type is(sgd_classifier)
      clf%loss=model%loss;clf%penalty=model%penalty;clf%learning_rate=model%learning_rate
      clf%alpha=model%alpha;clf%eta0=model%eta0;clf%power_t=model%power_t;clf%tol=model%tol
      clf%max_iter=model%max_iter;clf%n_iter_no_change=model%n_iter_no_change;clf%average=model%average
      clf%random_state=model%random_state;clf%fit_intercept=model%fit_intercept;clf%shuffle=model%shuffle
      clf%class_weight_balanced=model%class_weight_balanced
      model=clf
    end select
  end subroutine
  subroutine fit_regression(self,x,y,estimator,sample_weight,feature_names)
    class(select_from_model), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    class(*), optional, intent(in) :: estimator
    real(dp), optional, intent(in) :: sample_weight(:)
    type(feature_string), optional, intent(in) :: feature_names(:)
    call prepare(self,x,estimator,feature_names)
    if(.not.self%prefit) then
      call check_xy(x,size(y));call require(all(ieee_is_finite(y)),'SelectFromModel target must be finite')
      if(is_classifier(self%estimator_)) then
        call require(all(y>=-real(huge(1),dp)-1).and.all(y<=real(huge(1),dp)), &
          'SelectFromModel classifier label exceeds integer range')
        call require(all(modulo(y,1._dp)==0),'SelectFromModel classifier targets must be integral')
        call fit_classifier_model(self%estimator_,x,int(y),sample_weight)
      else
        call fit_regressor_model(self%estimator_,x,y,sample_weight)
      end if
    end if
  end subroutine
  subroutine fit_classification(self,x,y,estimator,sample_weight,feature_names)
    class(select_from_model), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    class(*), optional, intent(in) :: estimator
    real(dp), optional, intent(in) :: sample_weight(:)
    type(feature_string), optional, intent(in) :: feature_names(:)
    call prepare(self,x,estimator,feature_names)
    if(.not.self%prefit) then
      call check_xy(x,size(y))
      if(is_classifier(self%estimator_)) then
        call fit_classifier_model(self%estimator_,x,y,sample_weight)
      else
        call fit_regressor_model(self%estimator_,x,real(y,dp),sample_weight)
      end if
    end if
  end subroutine
  subroutine fit_multioutput(self,x,y,estimator,feature_names)
    class(select_from_model), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    class(*), optional, intent(in) :: estimator
    type(feature_string), optional, intent(in) :: feature_names(:)
    call prepare(self,x,estimator,feature_names)
    if(.not.self%prefit) then
      select type(model=>self%estimator_)
      class is(multi_task_elastic_net)
        ! A copied warm-start state must not alter sklearn clone semantics.
        if(allocated(model%coef_)) deallocate(model%coef_)
        call model%fit(x,y)
      class default
        call require(.false.,'SelectFromModel matrix targets require a native multi-task estimator')
      end select
    end if
  end subroutine
  subroutine partial_fit_regression(self,x,y,sample_weight,feature_names)
    class(select_from_model), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    type(feature_string), optional, intent(in) :: feature_names(:)
    call prepare(self,x,feature_names=feature_names,incremental=.true.)
    if(.not.self%prefit) then
      select type(model=>self%estimator_)
      type is(sgd_regressor)
        call model%partial_fit(x,y,sample_weight)
      class default
        call require(.false.,'SelectFromModel partial_fit requires a native SGD estimator')
      end select
    end if
  end subroutine
  subroutine partial_fit_classification(self,x,y,classes,sample_weight,feature_names)
    class(select_from_model), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    integer, optional, intent(in) :: classes(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    type(feature_string), optional, intent(in) :: feature_names(:)
    call prepare(self,x,feature_names=feature_names,incremental=.true.)
    if(.not.self%prefit) then
      select type(model=>self%estimator_)
      type is(sgd_classifier)
        call model%partial_fit(x,y,classes,sample_weight)
      class default
        call require(.false.,'SelectFromModel partial_fit requires a native SGD estimator')
      end select
    end if
  end subroutine
  function transform(self,x) result(z)
    class(select_from_model), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    integer, allocatable :: indices(:)
    logical, allocatable :: mask(:)
    integer :: i
    call check_x(x);mask=self%get_support()
    call require(size(x,2)==size(mask),'SelectFromModel transform feature count mismatch')
    indices=pack([(i,i=1,size(mask))],mask);z=x(:,indices)
  end function
  function inverse_transform(self,x) result(z)
    class(select_from_model), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    integer, allocatable :: indices(:)
    logical, allocatable :: mask(:)
    integer :: i
    call check_x(x)
    mask=self%get_support();indices=pack([(i,i=1,size(mask))],mask)
    call require(size(x,2)==size(indices),'SelectFromModel inverse feature count mismatch')
    allocate(z(size(x,1),size(mask)));z=0;z(:,indices)=x
  end function
  function get_feature_names_out(self,input_features) result(names)
    class(select_from_model), intent(in) :: self
    type(feature_string), optional, intent(in) :: input_features(:)
    type(feature_string), allocatable :: names(:),input(:)
    logical, allocatable :: mask(:)
    character(32) :: number
    integer :: i
    call require(allocated(self%estimator_),'SelectFromModel feature names require fit')
    mask=self%get_support();allocate(input(size(mask)))
    if(present(input_features)) then
      call require(size(input_features)==size(mask),'SelectFromModel feature name count mismatch')
      do i=1,size(mask)
        call require(allocated(input_features(i)%text),'SelectFromModel feature name is unallocated')
        if(allocated(self%feature_names_in_)) then
          call require(len(input_features(i)%text)==len(self%feature_names_in_(i)%text), &
            'SelectFromModel input feature names differ')
          call require(input_features(i)%text==self%feature_names_in_(i)%text, &
            'SelectFromModel input feature names differ')
        end if
      end do
      input=input_features
    else if(allocated(self%feature_names_in_)) then
      input=self%feature_names_in_
    else
      do i=1,size(mask)
        write(number,'(i0)') i-1;input(i)%text='x'//trim(number)
      end do
    end if
    names=pack(input,mask)
  end function
  function fit_transform_prefit(self,x,estimator,feature_names) result(z)
    class(select_from_model), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    class(*), optional, intent(in) :: estimator
    type(feature_string), optional, intent(in) :: feature_names(:)
    real(dp), allocatable :: z(:,:)
    call self%fit_prefit(x,estimator,feature_names);z=self%transform(x)
  end function
  function fit_transform_regression(self,x,y,estimator,sample_weight,feature_names) result(z)
    class(select_from_model), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    class(*), optional, intent(in) :: estimator
    real(dp), optional, intent(in) :: sample_weight(:)
    type(feature_string), optional, intent(in) :: feature_names(:)
    real(dp), allocatable :: z(:,:)
    call self%fit_regression(x,y,estimator,sample_weight,feature_names);z=self%transform(x)
  end function
  function fit_transform_classification(self,x,y,estimator,sample_weight,feature_names) result(z)
    class(select_from_model), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    class(*), optional, intent(in) :: estimator
    real(dp), optional, intent(in) :: sample_weight(:)
    type(feature_string), optional, intent(in) :: feature_names(:)
    real(dp), allocatable :: z(:,:)
    call self%fit_classification(x,y,estimator,sample_weight,feature_names);z=self%transform(x)
  end function
  function fit_transform_multioutput(self,x,y,estimator,feature_names) result(z)
    class(select_from_model), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    class(*), optional, intent(in) :: estimator
    type(feature_string), optional, intent(in) :: feature_names(:)
    real(dp), allocatable :: z(:,:)
    call self%fit_multioutput(x,y,estimator,feature_names);z=self%transform(x)
  end function
end module
