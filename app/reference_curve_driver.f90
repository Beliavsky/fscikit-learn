program reference_curve_driver
  use sklearn_validation
  use sklearn_common, only: dp,require
  use sklearn_parameter_space, only: param,parameter_value,parameter_set
  use sklearn_search_adapters, only: fit_search_native,set_search_params,search_regression,search_labels
  use sklearn_feature_composition, only: fit_model
  use sklearn_linear_model, only: ridge,elastic_net,logistic_regression
  use sklearn_sgd, only: sgd_regressor,sgd_classifier
  use sklearn_naive_bayes, only: gaussian_nb
  use sklearn_neighbors, only: kneighbors_regressor,kneighbors_classifier
  use sklearn_preprocessing, only: standard_scaler
  use sklearn_pipeline, only: pipeline
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type :: counter_model
    integer :: fits=0
    real(dp) :: offset=0
  end type
  type(counter_model) :: counter
  class(*), allocatable :: estimator
  type(ridge) :: reg
  type(elastic_net) :: elastic
  type(logistic_regression) :: logistic
  type(gaussian_nb) :: bayes
  type(kneighbors_regressor) :: knr
  type(kneighbors_classifier) :: knc
  type(sgd_regressor) :: sgdreg
  type(sgd_classifier) :: sgdclf
  type(standard_scaler) :: scale
  type(pipeline) :: chain
  type(cross_validation_options) :: cv
  type(learning_curve_options) :: options
  type(learning_curve_result) :: learning
  type(validation_curve_result) :: validation
  type(permutation_score_result) :: permutation
  type(parameter_value), allocatable :: param_range(:)
  real(dp), allocatable :: x(:,:),yr(:),weights(:),w(:),trace(:,:)
  integer, allocatable :: yi(:),groups(:),given_groups(:)
  integer :: mode,family,config,n,p,ncv,weighted,integer_target,incremental,seed,nperm,grouped,i,j,k
  integer :: fit_calls=0,partial_calls=0,score_calls=0,init_calls=0,set_calls=0
  logical :: classifier,no_target
  character(64) :: parameter_name
  read(*,*) mode,family,config,n,p,ncv,weighted,integer_target,incremental,seed,nperm,grouped
  allocate(x(n,p),yr(n),yi(n),weights(n),groups(n),trace(0,4))
  read(*,*) x;read(*,*) yr;read(*,*) yi;read(*,*) weights;read(*,*) groups
  classifier=any(family==[2,3,5,8]);no_target=family==9
  if(weighted==1) w=weights
  if(grouped==1) given_groups=groups
  reg%alpha=.4_dp;elastic%alpha=.08_dp;elastic%l1_ratio=.4_dp;elastic%tol=1.e-11_dp
  logistic%c=.8_dp;logistic%tol=1.e-10_dp;bayes%var_smoothing=.05_dp
  knr%n_neighbors=1;knc%n_neighbors=1
  sgdreg%learning_rate='constant';sgdreg%eta0=.02_dp;sgdreg%max_iter=6;sgdreg%tol=-1
  sgdclf%loss='log_loss';sgdclf%learning_rate='constant';sgdclf%eta0=.02_dp;sgdclf%max_iter=6;sgdclf%tol=-1
  if(config==6.and.incremental==1) then
    call sgdreg%fit(x,yr);call sgdclf%fit(x,yi);call bayes%fit(x,yi)
  end if
  select case(family)
  case(0);allocate(estimator,source=reg)
  case(1);allocate(estimator,source=elastic)
  case(2);allocate(estimator,source=logistic)
  case(3);allocate(estimator,source=bayes)
  case(4);allocate(estimator,source=knr)
  case(5);allocate(estimator,source=knc)
  case(6)
    allocate(chain%steps(2));chain%steps(1)%name='scale';chain%steps(2)%name='model'
    allocate(chain%steps(1)%model,source=scale);allocate(chain%steps(2)%model,source=reg)
    allocate(estimator,source=chain)
  case(7);allocate(estimator,source=sgdreg)
  case(8);allocate(estimator,source=sgdclf)
  case(9);allocate(estimator,source=scale)
  case(10);allocate(estimator,source=counter)
  end select
  cv%cv=ncv;cv%fit_func=>traced_fit
  if(family==10) cv%task='regression'
  if(classifier) then
    cv%scoring='accuracy'
  else
    cv%scoring='neg_mean_squared_error'
    if(mod(config,3)==0) cv%scoring='r2'
  end if
  if(config==3) cv%scorer_func=>mean_error
  if(family==9) cv%scorer_func=>scaled_moment
  if(family==10) cv%scorer_func=>counter_score
  if(config==7.and.family==0) cv%scorer_func=>nonfinite_score
  if(config==4) then
    allocate(cv%cv_splits(2));cv%cv_splits(1)%train=[(i,i=1,n-6)];cv%cv_splits(1)%test=[(i,i=n-5,n)]
    cv%cv_splits(2)%train=[(i,i=1,12)];cv%cv_splits(2)%test=[(i,i=13,n)]
  else if(config==5) then
    allocate(cv%cv_splits(3))
    do i=1,3
      cv%cv_splits(i)%test=[(j,j=i,n,3)]
      cv%cv_splits(i)%train=pack([(j,j=1,n)],.not.[(any(cv%cv_splits(i)%test==j),j=1,n)])
    end do
  end if
  select case(mode)
  case(0)
    options%cross_validation_options=cv;options%exploit_incremental_learning=incremental==1
    options%shuffle=mod(config,2)==1;options%random_state=seed;options%return_times=mod(config,2)==0
    select case(config)
    case(1);options%train_sizes=[1._dp,.5_dp,.5_dp,.05_dp]
    case(2);options%train_counts=[12,3,12,18]
    case(3);options%train_sizes=[.25_dp,.6_dp,1._dp]
    case(4)
      options%train_counts=[5,15,n-6]
    case(5);options%train_sizes=[.3_dp,.7_dp,1._dp]
    case(6);options%train_counts=[4,12,18]
    case(7);options%train_sizes=[1.e-8_dp,.5_dp,1._dp]
    end select
    if(incremental==1.and.mod(config,2)==1) options%partial_fit_func=>traced_partial
    if(family==10.and.incremental==1) then
      options%partial_fit_func=>traced_partial;options%incremental_init_func=>initialize_counter
    end if
    call learn()
    write(*,*) size(learning%train_sizes),size(learning%folds),merge(1,0,allocated(learning%fit_time)), &
      fit_calls,partial_calls,score_calls,init_calls,set_calls
    write(*,*) learning%train_sizes
    call emit(learning%train_score);call emit(learning%test_score);call emit(real(learning%actual_train_sizes,dp))
    if(allocated(learning%fit_time)) then
      call emit(learning%fit_time);call emit(learning%score_time)
    end if
    do i=1,size(learning%folds)
      call emit_fold(learning%folds(i)%train,learning%folds(i)%test)
    end do
  case(1)
    parameter_name='alpha';param_range=[param(.4_dp),param(.01_dp),param(3._dp),param(.4_dp)]
    select case(family)
    case(2);parameter_name='C';param_range=[param(.1_dp),param(1._dp),param(10._dp),param(.1_dp)]
    case(3);parameter_name='var_smoothing';param_range=[param(.01_dp),param(.05_dp),param(.3_dp),param(.01_dp)]
    case(4,5);parameter_name='n_neighbors';param_range=[param(1),param(3),param(5),param(1)]
    case(6);parameter_name='model__alpha'
    case(9);parameter_name='with_mean';param_range=[param(.true.),param(.false.),param(.true.)]
    case(10);parameter_name='offset'
    end select
    call validate_parameters()
    write(*,*) size(param_range),size(validation%folds),fit_calls,partial_calls,score_calls,init_calls,set_calls
    call emit(validation%train_score);call emit(validation%test_score)
    do i=1,size(validation%folds)
      call emit_fold(validation%folds(i)%train,validation%folds(i)%test)
    end do
  case(2)
    call permute()
    write(*,*) size(permutation%permutation_scores),fit_calls,partial_calls,score_calls,init_calls,set_calls
    write(*,'(*(es26.17e3,1x))') permutation%score,permutation%pvalue
    write(*,'(*(es26.17e3,1x))') permutation%permutation_scores
  end select
  call emit(trace)
contains
  subroutine learn()
    if(no_target) then
      learning=learning_curve(estimator,x,options,w)
    else if(integer_target==1) then
      learning=learning_curve(estimator,x,yi,options,w)
    else if(classifier) then
      learning=learning_curve(estimator,x,real(yi,dp),options,w)
    else
      learning=learning_curve(estimator,x,yr,options,w)
    end if
  end subroutine
  subroutine validate_parameters()
    if(no_target) then
      validation=validation_curve(estimator,x,trim(parameter_name),param_range,cv,w,traced_set)
    else if(integer_target==1) then
      validation=validation_curve(estimator,x,yi,trim(parameter_name),param_range,cv,w,traced_set)
    else if(classifier) then
      validation=validation_curve(estimator,x,real(yi,dp),trim(parameter_name),param_range,cv,w,traced_set)
    else
      validation=validation_curve(estimator,x,yr,trim(parameter_name),param_range,cv,w,traced_set)
    end if
  end subroutine
  subroutine permute()
    if(integer_target==1) then
      permutation=permutation_test_score(estimator,x,yi,cv,nperm,seed,given_groups,w)
    else if(classifier) then
      permutation=permutation_test_score(estimator,x,real(yi,dp),cv,nperm,seed,given_groups,w)
    else
      permutation=permutation_test_score(estimator,x,yr,cv,nperm,seed,given_groups,w)
    end if
  end subroutine
  subroutine record_trace(x,yr,yi,weights)
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: yr(:),weights(:)
    integer, optional, intent(in) :: yi(:)
    real(dp) :: row(4)
    real(dp), allocatable :: extended(:,:)
    row=[real(size(x,1),dp),0._dp,0._dp,0._dp]
    if(present(yr)) then
      row(2)=sum(yr);row(3)=dot_product(x(:,1),yr)
    else if(present(yi)) then
      row(2)=sum(real(yi,dp));row(3)=dot_product(x(:,1),real(yi,dp))
    end if
    if(present(weights)) row(4)=size(weights)
    allocate(extended(size(trace,1)+1,4));extended(:size(trace,1),:)=trace
    extended(size(trace,1)+1,:)=row;call move_alloc(extended,trace)
  end subroutine
  subroutine traced_fit(model,x,yr,yi,sample_weight)
    class(*), intent(inout) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: yr(:),sample_weight(:)
    integer, optional, intent(in) :: yi(:)
    fit_calls=fit_calls+1;call record_trace(x,yr,yi,sample_weight)
    select type(model)
    type is(counter_model);model%fits=model%fits+1
    class default
      if(present(yr).or.present(yi)) then
        call fit_search_native(model,x,yr,yi,sample_weight)
      else
        call fit_model(model,x)
      end if
    end select
  end subroutine
  subroutine traced_partial(model,x,yr,yi,classes,sample_weight)
    class(*), intent(inout) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: yr(:),sample_weight(:)
    integer, optional, intent(in) :: yi(:),classes(:)
    partial_calls=partial_calls+1;call record_trace(x,yr,yi,sample_weight)
    select type(model)
    type is(sgd_regressor);call model%partial_fit(x,yr,sample_weight)
    type is(sgd_classifier);call model%partial_fit(x,yi,classes,sample_weight)
    type is(gaussian_nb);call model%partial_fit(x,yi,classes,sample_weight)
    type is(counter_model);model%fits=model%fits+1
    end select
  end subroutine
  subroutine initialize_counter(model)
    class(*), intent(inout) :: model
    init_calls=init_calls+1
    select type(model)
    type is(counter_model);model%fits=0
    end select
  end subroutine
  subroutine traced_set(model,parameters)
    class(*), intent(inout) :: model
    type(parameter_set), intent(in) :: parameters
    type(parameter_value) :: value
    set_calls=set_calls+1
    select type(model)
    type is(counter_model);value=parameters%get('offset');model%offset=value%as_real()
    class default;call set_search_params(model,parameters)
    end select
  end subroutine
  real(dp) function mean_error(model,x,yr,yi) result(value)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: yr(:)
    integer, optional, intent(in) :: yi(:)
    real(dp), allocatable :: predicted(:)
    integer, allocatable :: labels(:)
    score_calls=score_calls+1
    if(present(yi)) then
      labels=search_labels(model,x);value=sum(real(labels-yi,dp))/size(yi)
    else
      predicted=search_regression(model,x);value=sum(predicted-yr)/size(yr)
    end if
  end function
  real(dp) function scaled_moment(model,x,yr,yi) result(value)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: yr(:)
    integer, optional, intent(in) :: yi(:)
    real(dp), allocatable :: z(:,:)
    score_calls=score_calls+1;value=0
    select type(model)
    type is(standard_scaler);z=model%transform(x);value=sum(z*z)/size(z)
    end select
  end function
  real(dp) function counter_score(model,x,yr,yi) result(value)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: yr(:)
    integer, optional, intent(in) :: yi(:)
    score_calls=score_calls+1;value=0
    select type(model)
    type is(counter_model);value=model%fits+.001_dp*model%offset
    end select
  end function
  real(dp) function nonfinite_score(model,x,yr,yi) result(value)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: yr(:)
    integer, optional, intent(in) :: yi(:)
    score_calls=score_calls+1;value=ieee_value(0._dp,ieee_quiet_nan)
    if(mod(size(x,1),2)==0) value=ieee_value(0._dp,ieee_positive_inf)
  end function
  subroutine emit(values)
    real(dp), intent(in) :: values(:,:)
    integer :: row
    write(*,*) shape(values)
    do row=1,size(values,1)
      write(*,'(*(es26.17e3,1x))') values(row,:)
    end do
  end subroutine
  subroutine emit_fold(train,test)
    integer, intent(in) :: train(:),test(:)
    write(*,*) size(train),size(test);write(*,*) train;write(*,*) test
  end subroutine
end program
