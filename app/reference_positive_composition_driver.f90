program reference_positive_composition_driver
  use sklearn_common, only: dp
  use sklearn_linear_model, only: linear_regression,ridge
  use sklearn_pipeline, only: pipeline
  use sklearn_compose, only: transformed_target_regressor
  use sklearn_preprocessing, only: standard_scaler
  use sklearn_bagging, only: bagging_regressor
  use sklearn_multioutput, only: multi_output_regressor,regressor_chain
  implicit none
  class(linear_regression), allocatable :: base
  type(pipeline) :: pipe
  type(transformed_target_regressor) :: target
  type(standard_scaler) :: scale
  type(bagging_regressor) :: bag
  type(multi_output_regressor) :: multi
  type(regressor_chain) :: chain
  real(dp), allocatable :: x(:,:),y(:,:),q(:,:),w(:)
  real(dp) :: alpha
  integer :: n,p,m,kind,base_kind,intercept,seed,weighted,j
  read(*,*) n,p,m,kind,base_kind,intercept,seed,weighted,alpha
  allocate(x(n,p),y(n,2),q(m,p),w(n));read(*,*) x;read(*,*) y;read(*,*) q;read(*,*) w
  if(base_kind==0) then
    allocate(linear_regression::base)
  else
    allocate(ridge::base)
    select type(base)
    type is(ridge)
      base%alpha=alpha
    end select
  end if
  base%positive=.true.;base%fit_intercept=intercept/=0
  select case(kind)
  case(0)
    allocate(pipe%steps(2));pipe%steps(1)%name='scale';allocate(pipe%steps(1)%model,source=scale)
    pipe%steps(2)%name='model';allocate(pipe%steps(2)%model,source=base)
    if(weighted/=0) then
      call pipe%fit(x,y(:,1),w)
    else
      call pipe%fit(x,y(:,1))
    end if
    call vector(pipe%predict(q));call estimator(pipe%steps(2)%model)
  case(1)
    allocate(target%regressor,source=base);allocate(target%transformer,source=scale)
    if(weighted/=0) then
      call target%fit(x,y(:,1),w)
    else
      call target%fit(x,y(:,1))
    end if
    call vector(target%predict(q));call estimator(target%regressor_)
  case(2)
    bag%n_estimators=5;bag%random_state=seed;bag%max_samples_fraction=.75_dp;bag%max_features_fraction=.75_dp
    if(weighted/=0) then
      call bag%fit(x,y(:,1),base,w)
    else
      call bag%fit(x,y(:,1),base)
    end if
    call vector(bag%predict(q));call vector(real(bag%seeds_,dp))
    do j=1,5
      call vector(real(bag%estimators_features_(j)%values,dp));call vector(real(bag%estimators_samples_(j)%values,dp))
      call estimator(bag%estimators_(j)%estimator)
    end do
  case(3)
    call multi%fit(x,y,base);call matrix(multi%predict(q))
    do j=1,2
      call estimator(multi%estimators_(j)%estimator_)
    end do
  case(4)
    call chain%fit(x,y,base);call matrix(chain%predict(q));call vector(real(chain%order_,dp))
    do j=1,2
      call estimator(chain%estimators_(j)%estimator_)
    end do
  case default;error stop 'unknown positive composition'
  end select
contains
  subroutine estimator(model)
    class(*), intent(in) :: model
    select type(model)
    class is(linear_regression)
      call vector(model%coef_);call vector([model%intercept_,merge(1._dp,0._dp,model%positive)])
    class default;error stop 'positive composition lost linear estimator'
    end select
  end subroutine
  subroutine vector(a)
    real(dp), intent(in) :: a(:)
    call matrix(reshape(a,[size(a),1]))
  end subroutine
  subroutine matrix(a)
    real(dp), intent(in) :: a(:,:)
    print *,shape(a)
    print '(*(es25.16e3,1x))',a
  end subroutine
end program
