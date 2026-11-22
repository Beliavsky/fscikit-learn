! TransformedTargetRegressor: sklearn 1.6.1 compose/_target.py (BSD-3-Clause).
! See THIRD_PARTY_NOTICES.md. Native clone/storage adapters are original MIT code.
module sklearn_compose
  use sklearn_common, only: dp,require,check_x,check_xy
  use sklearn_function_transformer, only: function_transformer,matrix_function,matrices_close
  use sklearn_preprocessing, only: standard_scaler,minmax_scaler,maxabs_scaler,robust_scaler
  use sklearn_power_transform, only: power_transformer
  use sklearn_quantile_transform, only: quantile_transformer
  use sklearn_linear_model, only: linear_regression,ridge,elastic_net,lasso
  use sklearn_neighbors, only: kneighbors_regressor
  use sklearn_kernel_ridge, only: kernel_ridge
  use sklearn_dummy, only: dummy_regressor
  use sklearn_base_dispatch, only: base_supported,train_regressor,regression
  use sklearn_weighted_linear, only: weighted_linear
  use sklearn_composite_scores, only: regression_score
  use sklearn_ieee_math, only: ieee_mean
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: transformed_target_regressor,target_regressor_slot,target_regressor_bundle
  type :: target_regressor_slot
    class(*), allocatable :: model_
  end type
  type :: target_regressor_bundle
    type(target_regressor_slot), allocatable :: estimators_(:)
  end type
  type :: transformed_target_regressor
    class(*), allocatable :: regressor,transformer,regressor_,transformer_
    procedure(matrix_function), pointer, nopass :: func=>null(),inverse_func=>null()
    logical :: check_inverse=.true.,inverse_checked_=.false.,inverse_consistent_=.true.
    integer :: training_dim_=0,n_outputs_=0,n_transformed_outputs_=0,n_features_in_=0
  contains
    procedure, private :: fit_vector,fit_matrix,score_vector,score_matrix
    generic :: fit=>fit_vector,fit_matrix
    generic :: score=>score_vector,score_matrix
    procedure :: predict=>target_predict
    procedure :: predict_matrix=>target_predict_matrix
    procedure :: transformed_predict=>target_transformed_predict
  end type
contains
  subroutine transformer_fit(model,y)
    class(*), intent(inout) :: model
    real(dp), intent(in) :: y(:,:)
    select type(model)
    type is(function_transformer)
      call model%fit(y)
    type is(standard_scaler)
      call model%fit(y)
    type is(minmax_scaler)
      call model%fit(y)
    type is(maxabs_scaler)
      call model%fit(y)
    type is(robust_scaler)
      call model%fit(y)
    type is(power_transformer)
      call model%fit(y)
    type is(quantile_transformer)
      call model%fit(y)
    class default
      call require(.false.,'unsupported TransformedTargetRegressor transformer')
    end select
  end subroutine
  function target_map(model,y,inverse) result(z)
    class(*), intent(in) :: model
    real(dp), intent(in) :: y(:,:)
    logical, intent(in) :: inverse
    real(dp), allocatable :: z(:,:)
    select type(model)
    type is(function_transformer)
      if(inverse) then
        z=model%inverse_transform(y)
      else
        z=model%transform(y)
      end if
    type is(standard_scaler)
      if(inverse) then
        z=model%inverse_transform(y)
      else
        z=model%transform(y)
      end if
    type is(minmax_scaler)
      if(inverse) then
        z=model%inverse_transform(y)
      else
        z=model%transform(y)
      end if
    type is(maxabs_scaler)
      if(inverse) then
        z=model%inverse_transform(y)
      else
        z=model%transform(y)
      end if
    type is(robust_scaler)
      if(inverse) then
        z=model%inverse_transform(y)
      else
        z=model%transform(y)
      end if
    type is(power_transformer)
      if(inverse) then
        z=model%inverse_transform(y)
      else
        z=model%transform(y)
      end if
    type is(quantile_transformer)
      if(inverse) then
        z=model%inverse_transform(y)
      else
        z=model%transform(y)
      end if
    class default
      call require(.false.,'unsupported TransformedTargetRegressor transformer')
    end select
  end function
  subroutine prepare_transformer(self,y)
    class(transformed_target_regressor), intent(inout) :: self
    real(dp), intent(in) :: y(:,:)
    type(function_transformer) :: fresh
    real(dp), allocatable :: z(:,:),roundtrip(:,:)
    integer :: step
    if(allocated(self%transformer)) then
      call require(.not.associated(self%func).and..not.associated(self%inverse_func), &
        'TransformedTargetRegressor cannot combine transformer and functions')
      if(allocated(self%transformer_)) deallocate(self%transformer_)
      select type(prototype=>self%transformer)
      type is(function_transformer)
        fresh%func=>prototype%func;fresh%inverse_func=>prototype%inverse_func;fresh%names_func=>prototype%names_func
        fresh%validate=prototype%validate;fresh%check_inverse=prototype%check_inverse
        fresh%feature_names_out=prototype%feature_names_out
        allocate(self%transformer_,source=fresh)
      class default
        allocate(self%transformer_,source=self%transformer)
      end select
    else
      call require(associated(self%func).eqv.associated(self%inverse_func), &
        'TransformedTargetRegressor requires both func and inverse_func')
      fresh%func=>self%func;fresh%inverse_func=>self%inverse_func
      fresh%validate=.true.;fresh%check_inverse=self%check_inverse
      if(allocated(self%transformer_)) deallocate(self%transformer_)
      allocate(self%transformer_,source=fresh)
    end if
    call transformer_fit(self%transformer_,y)
    self%inverse_checked_=.false.;self%inverse_consistent_=.true.
    if(self%check_inverse) then
      step=max(1,size(y,1)/10)
      z=target_map(self%transformer_,y(1:size(y,1):step,:),.false.)
      roundtrip=target_map(self%transformer_,z,.true.)
      self%inverse_checked_=.true.
      self%inverse_consistent_=matrices_close(y(1:size(y,1):step,:),roundtrip,1.e-5_dp,1.e-8_dp)
    end if
  end subroutine
  logical function independent_targets(model) result(ok)
    class(*), intent(in) :: model
    ok=.true.
    select type(model)
    type is(linear_regression)
    type is(ridge)
    type is(elastic_net)
    type is(lasso)
    type is(kneighbors_regressor)
    type is(kernel_ridge)
    type is(dummy_regressor)
    class default
      ok=.false.
    end select
  end function
  subroutine fit_base(model,x,y,weights)
    class(*), intent(inout) :: model
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), optional, intent(in) :: weights(:)
    real(dp) :: w(size(y))
    w=1;if(present(weights)) w=weights
    select type(model)
    type is(linear_regression)
      call weighted_linear(model,x,y,w,0._dp)
    type is(ridge)
      call weighted_linear(model,x,y,w,model%alpha)
    class default
      call train_regressor(model,x,y,weights)
    end select
  end subroutine
  subroutine fit_vector(self,x,y,sample_weight)
    class(transformed_target_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    call fit_matrix(self,x,reshape(y,[size(y),1]),sample_weight)
    self%training_dim_=1
  end subroutine
  subroutine fit_matrix(self,x,y,sample_weight)
    class(transformed_target_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    real(dp), allocatable :: transformed(:,:)
    class(*), allocatable :: prototype
    type(linear_regression) :: ordinary
    type(target_regressor_bundle) :: bundle
    integer :: j
    call check_xy(x,size(y,1));call check_x(y)
    if(present(sample_weight)) then
      call require(size(sample_weight)==size(y,1),'TransformedTargetRegressor weight length mismatch')
      call require(all(ieee_is_finite(sample_weight)),'TransformedTargetRegressor weights must be finite')
      call require(all(sample_weight>=0),'TransformedTargetRegressor weights must be nonnegative')
    end if
    call prepare_transformer(self,y)
    transformed=target_map(self%transformer_,y,.false.)
    call check_x(transformed)
    call require(size(transformed,1)==size(x,1),'TransformedTargetRegressor transformed sample count mismatch')
    if(allocated(self%regressor)) then
      allocate(prototype,source=self%regressor)
    else
      allocate(prototype,source=ordinary)
    end if
    call require(base_supported(prototype,.false.),'unsupported TransformedTargetRegressor regressor')
    if(allocated(self%regressor_)) deallocate(self%regressor_)
    if(size(transformed,2)==1) then
      allocate(self%regressor_,source=prototype)
      call fit_base(self%regressor_,x,transformed(:,1),sample_weight)
    else
      call require(independent_targets(prototype),'TransformedTargetRegressor base has no joint multi-target fit')
      allocate(bundle%estimators_(size(transformed,2)))
      do j=1,size(transformed,2)
        allocate(bundle%estimators_(j)%model_,source=prototype)
        call fit_base(bundle%estimators_(j)%model_,x,transformed(:,j),sample_weight)
      end do
      allocate(self%regressor_,source=bundle)
    end if
    self%training_dim_=2;self%n_outputs_=size(y,2);self%n_transformed_outputs_=size(transformed,2)
    self%n_features_in_=size(x,2)
  end subroutine
  function target_transformed_predict(self,x) result(values)
    class(transformed_target_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:,:)
    integer :: j
    call require(allocated(self%regressor_).and.allocated(self%transformer_),'TransformedTargetRegressor is not fitted')
    call check_x(x)
    call require(size(x,2)==self%n_features_in_,'TransformedTargetRegressor query feature width mismatch')
    allocate(values(size(x,1),self%n_transformed_outputs_))
    select type(model=>self%regressor_)
    type is(target_regressor_bundle)
      do j=1,size(values,2)
        values(:,j)=regression(model%estimators_(j)%model_,x)
      end do
    class default
      values(:,1)=regression(self%regressor_,x)
    end select
  end function
  function target_predict_matrix(self,x) result(values)
    class(transformed_target_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:,:),transformed(:,:)
    transformed=self%transformed_predict(x);values=target_map(self%transformer_,transformed,.true.)
  end function
  function target_predict(self,x) result(values)
    class(transformed_target_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: values(:),matrix(:,:)
    matrix=self%predict_matrix(x)
    call require(size(matrix,2)==1,'TransformedTargetRegressor vector prediction requires one output')
    values=matrix(:,1)
  end function
  real(dp) function score_vector(self,x,y,sample_weight) result(value)
    class(transformed_target_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    value=regression_score(y,self%predict(x),sample_weight,'TransformedTargetRegressor')
  end function
  real(dp) function score_matrix(self,x,y,sample_weight) result(value)
    class(transformed_target_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    real(dp), allocatable :: predicted(:,:)
    real(dp) :: scores(size(y,2))
    integer :: j
    predicted=self%predict_matrix(x)
    call require(all(shape(predicted)==shape(y)),'TransformedTargetRegressor score output shape mismatch')
    call require(size(y,2)>0,'TransformedTargetRegressor score needs an output')
    do j=1,size(y,2)
      scores(j)=regression_score(y(:,j),predicted(:,j),sample_weight,'TransformedTargetRegressor')
    end do
    value=ieee_mean(scores)
  end function
end module
