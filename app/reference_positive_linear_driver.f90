program reference_positive_linear_driver
  use sklearn_linear_model
  use sklearn_weighted_linear, only: weighted_linear
  use sklearn_model_selection, only: ridge_cv
  use sklearn_nnls, only: nnls_solve
  use sklearn_common, only: dp
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  class(linear_regression), allocatable :: model
  real(dp), allocatable :: x(:,:),y(:),q(:,:),weights(:),alphas(:),ratios(:),coef(:)
  real(dp) :: alpha,ratio,tol,eps,residual
  integer :: n,p,m,kind,intercept,positive,weighted,steps,cv,grid_code,na,nr,iterations
  logical :: converged
  character(40) :: argument
  call get_command_argument(1,argument)
  if(len_trim(argument)>0.and.argument/='kernel') then
    call reject(trim(argument));error stop 'invalid positive-linear input accepted'
  end if
  if(argument=='kernel') then
    read(*,*) n,p,steps
    allocate(x(n,p),y(n));read(*,*) x;read(*,*) y
    call nnls_solve(x,y,coef,residual,iterations,converged,steps)
    call vector(coef);call vector([residual,real(iterations,dp),merge(1._dp,0._dp,converged)])
    stop
  end if
  read(*,*) n,p,m,kind,intercept,positive,weighted,steps,tol,alpha,ratio,cv,grid_code,na,nr,eps
  allocate(x(n,p),y(n),q(m,p),weights(n),alphas(na),ratios(nr))
  read(*,*) x;read(*,*) y;read(*,*) q;read(*,*) weights;read(*,*) alphas;read(*,*) ratios
  select case(kind)
  case(0);allocate(linear_regression::model)
  case(1);allocate(ridge::model)
  case(2);allocate(elastic_net::model)
  case(3);allocate(lasso::model)
  case(4);allocate(elastic_net_cv::model)
  case(5);allocate(lasso_cv::model)
  case default;error stop 'unknown positive linear kind'
  end select
  model%fit_intercept=intercept/=0;model%positive=positive/=0
  select type(model)
  type is(ridge)
    model%alpha=alpha
  class is(elastic_net)
    model%alpha=alpha;model%l1_ratio=ratio;model%tol=tol;model%max_iter=steps
  end select
  select type(model)
  class is(elastic_net_cv)
    model%cv=cv;model%n_alphas=na;model%eps=eps;model%l1_ratios=ratios
    if(grid_code/=0) model%alphas=alphas
  end select
  if(weighted/=0) then
    if(kind==0) alpha=0
    call weighted_linear(model,x,y,weights,alpha)
  else
    call model%fit(x,y)
  end if
  call vector(model%coef_);call vector([model%intercept_]);call vector(model%predict(x));call vector(model%predict(q))
  call vector([model%score(x,y),real(model%rank_,dp)])
  select type(model)
  class is(elastic_net)
    call vector(real([model%n_iter_,merge(1,0,model%converged_)],dp))
  class default
    call vector([0._dp,1._dp])
  end select
  select type(model)
  class is(elastic_net_cv)
    call vector([model%alpha_,model%l1_ratio_,model%best_mse_])
    call matrix(model%alphas_)
    call matrix(reshape(model%mse_path_,[size(model%mse_path_,1)*size(model%mse_path_,2),size(model%mse_path_,3)]))
  end select
contains
  subroutine vector(a)
    real(dp), intent(in) :: a(:)
    call matrix(reshape(a,[size(a),1]))
  end subroutine
  subroutine matrix(a)
    real(dp), intent(in) :: a(:,:)
    print *,shape(a)
    print '(*(es25.16e3,1x))',a
  end subroutine
  subroutine reject(name)
    character(*), intent(in) :: name
    class(linear_regression), allocatable :: bad_model
    type(linear_regression) :: ols
    type(ridge) :: rr
    type(elastic_net) :: en
    real(dp) :: a(4,2),b(4),w(4),nan
    real(dp), allocatable :: answer(:)
    real(dp) :: res
    integer :: it
    logical :: ok
    a(:,1)=[1._dp,2._dp,3._dp,4._dp];a(:,2)=[2._dp,1._dp,4._dp,3._dp];b=[1._dp,2._dp,0._dp,3._dp];w=1
    ols%positive=.true.;rr%positive=.true.;en%positive=.true.;nan=ieee_value(0._dp,ieee_quiet_nan)
    select case(name)
    case('bayesian');allocate(bayesian_ridge::bad_model)
    case('ard');allocate(ard_regression::bad_model)
    case('omp');allocate(orthogonal_matching_pursuit::bad_model)
    case('huber');allocate(huber_regressor::bad_model)
    case('ransac');allocate(ransac_regressor::bad_model)
    case('theil_sen');allocate(theil_sen_regressor::bad_model)
    case('pa');allocate(passive_aggressive_regressor::bad_model)
    case('tweedie');allocate(tweedie_regressor::bad_model)
    case('poisson');allocate(poisson_regressor::bad_model)
    case('gamma');allocate(gamma_regressor::bad_model)
    case('ridge_cv');allocate(ridge_cv::bad_model)
    case('empty');call ols%fit(a(:0,:),b(:0));return
    case('width');call ols%fit(a,b);answer=ols%predict(a(:,:1));return
    case('unfitted');answer=ols%predict(a);return
    case('x_nan');a(1,1)=nan
    case('y_inf');b(1)=ieee_value(0._dp,ieee_positive_inf)
    case('length');call ols%fit(a,b(:3));return
    case('data_range');a=huge(1._dp)/4
    case('coefficient_range');a=1.e-160_dp;b=1.e150_dp;ols%fit_intercept=.false.
    case('ridge_alpha_nan');rr%alpha=nan;call rr%fit(a,b);return
    case('ridge_alpha_negative');rr%alpha=-1;call rr%fit(a,b);return
    case('elastic_alpha_nan');en%alpha=nan;call en%fit(a,b);return
    case('elastic_ratio_nan');en%l1_ratio=nan;call en%fit(a,b);return
    case('elastic_tol_nan');en%tol=nan;call en%fit(a,b);return
    case('weights_length');call weighted_linear(ols,a,b,w(:3),0._dp);return
    case('weights_nan');w(1)=nan;call weighted_linear(ols,a,b,w,0._dp);return
    case('weights_negative');w(1)=-1;call weighted_linear(ols,a,b,w,0._dp);return
    case('weights_zero');w=0;call weighted_linear(ols,a,b,w,0._dp);return
    case('nnls_iterations');call nnls_solve(a,b,answer,res,it,ok,0);return
    case('prediction_range')
      ols%fit_intercept=.false.;a=1.e-100_dp;b=1.e100_dp;call ols%fit(a,b)
      a=1.e150_dp;answer=ols%predict(a);return
    case default;error stop 'unknown positive-linear rejection'
    end select
    if(allocated(bad_model)) then
      bad_model%positive=.true.;call bad_model%fit(a,b)
    else
      call ols%fit(a,b)
    end if
  end subroutine
end program
