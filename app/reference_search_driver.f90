program reference_search_driver
  use sklearn_search
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_common, only: dp
  use sklearn_linear_model, only: ridge,elastic_net,logistic_regression
  use sklearn_naive_bayes, only: gaussian_nb
  use sklearn_neighbors, only: kneighbors_regressor,kneighbors_classifier
  use sklearn_pipeline, only: pipeline
  use sklearn_preprocessing, only: standard_scaler
  use sklearn_feature_selection, only: select_from_model
  use sklearn_search_adapters, only: fit_search_native,set_search_params
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf,ieee_negative_inf
  implicit none
  class(grid_search_cv), allocatable :: model
  type(ridge) :: regularized
  type(elastic_net) :: elastic
  type(logistic_regression) :: logistic
  type(gaussian_nb) :: bayes
  type(kneighbors_regressor) :: neighbor_reg
  type(kneighbors_classifier) :: neighbor_clf
  type(pipeline) :: chain
  type(standard_scaler) :: scale
  type(select_from_model) :: selector
  type(parameter_value) :: value
  real(dp), allocatable :: x(:,:),q(:,:),yr(:),qr(:),weights(:),prediction(:,:),prob(:,:),decision(:,:)
  integer, allocatable :: yi(:),qi(:),labels(:)
  integer :: family,config,seed,random,n_iter,n,p,m,cv,i,j,k,weighted,classification,integer_target,fit_calls,set_calls
  real(dp) :: score
  read(*,*) family,config,seed,random,n_iter,n,p,m,cv,weighted,classification,integer_target
  allocate(x(n,p),q(m,p),yr(n),qr(m),yi(n),qi(m),weights(n))
  read(*,*) x;read(*,*) q;read(*,*) yr;read(*,*) qr;read(*,*) yi;read(*,*) qi;read(*,*) weights
  if(random==1) then
    allocate(randomized_search_cv::model)
    select type(model)
    type is(randomized_search_cv);model%n_iter=n_iter;model%random_state=seed
    end select
  else
    allocate(grid_search_cv::model)
  end if
  select case(family)
  case(0,8,9);allocate(model%estimator,source=regularized)
  case(1);elastic%tol=1.e-11_dp;allocate(model%estimator,source=elastic)
  case(2);logistic%tol=1.e-10_dp;allocate(model%estimator,source=logistic)
  case(3);allocate(model%estimator,source=bayes)
  case(4);allocate(model%estimator,source=neighbor_reg)
  case(5);allocate(model%estimator,source=neighbor_clf)
  case(6,7)
    allocate(chain%steps(merge(3,2,family==7)))
    chain%steps(1)%name='scale';allocate(chain%steps(1)%model,source=scale)
    if(family==7) then
      allocate(selector%estimator,source=regularized)
      chain%steps(2)%name='select';allocate(chain%steps(2)%model,source=selector)
    end if
    chain%steps(size(chain%steps))%name='model';allocate(chain%steps(size(chain%steps))%model,source=regularized)
    allocate(model%estimator,source=chain)
  end select
  allocate(model%param_grid(1));allocate(model%param_grid(1)%axes(0))
  select case(family)
  case(0,9)
    call axis('fit_intercept',[param(.true.),param(.false.)]);call axis('alpha',[param(.01_dp),param(.3_dp),param(3._dp)])
  case(1)
    call axis('l1_ratio',[param(.25_dp),param(.8_dp)]);call axis('alpha',[param(.02_dp),param(.15_dp)])
    call axis('fit_intercept',[param(.true.),param(.false.)])
  case(2)
    call axis('fit_intercept',[param(.true.),param(.false.)]);call axis('C',[param(.1_dp),param(1._dp),param(10._dp)])
  case(3);call axis('var_smoothing',[param(1.e-9_dp),param(.05_dp),param(1._dp)])
  case(4,5)
    call axis('weights',[param('uniform'),param('distance')]);call axis('n_neighbors',[param(1),param(3),param(5)])
  case(6)
    call axis('scale__with_mean',[param(.true.),param(.false.)])
    call axis('model__alpha',[param(.03_dp),param(1._dp),param(15._dp)])
  case(7)
    call axis('select__threshold',[param('mean'),param('median')]);call axis('model__alpha',[param(.2_dp),param(2._dp)])
    call axis('select__estimator__alpha',[param(.1_dp),param(1._dp)])
  case(8);call axis('alpha',[param(0._dp),param(1._dp),param(2._dp),param(3._dp)])
  end select
  if(family==0.and.(config==4.or.config==6)) then
    model%param_grid=[model%param_grid,parameter_grid()]
    allocate(model%param_grid(2)%axes(1));model%param_grid(2)%axes(1)%name='alpha'
    model%param_grid(2)%axes(1)%values=[param(30._dp)]
  end if
  if(family==9.and.random==1.and.mod(config,2)==0) then
    deallocate(model%param_grid(1)%axes(2)%values);model%param_grid(1)%axes(2)%rvs=>random_alpha
  end if
  model%cv=cv;model%return_train_score=mod(config,2)==0
  if(classification==1) then
    model%scoring='accuracy'
    if(config==1) model%scoring='neg_log_loss'
    if(config==7) model%scoring='balanced_accuracy'
    if(config==2) model%scoring='roc_auc'
  else
    model%scoring='r2'
    if(config==1) model%scoring='neg_mean_squared_error'
    if(config==2.or.config==7) model%scoring='neg_mean_absolute_error'
  end if
  if(config==0) model%scoring='auto'
  if(config==3) model%refit=.false.
  if(config==4.or.config==6) then
    allocate(model%scorers(2));model%scorers(1)%name='primary';model%scorers(2)%name='secondary'
    if(classification==1) then
      model%scorers(1)%scoring='accuracy';model%scorers(2)%scoring='neg_log_loss'
    else
      model%scorers(1)%scoring='neg_mean_squared_error';model%scorers(2)%scoring='r2'
    end if
    model%refit_metric='secondary'
    if(config==6) then
      model%refit=.false.;model%refit_metric=''
    end if
  end if
  if(config==5) model%refit_func=>last_candidate
  if(config==7.or.config==8) then
    allocate(model%cv_splits(merge(4,3,config==8)))
    do i=1,size(model%cv_splits)
      if(config==8) then
        model%cv_splits(i)%test=[i]
      else
        model%cv_splits(i)%test=[(j,j=i,n,4+i)]
      end if
      model%cv_splits(i)%train=pack([(j,j=1,n)],.not.[(any(model%cv_splits(i)%test==j),j=1,n)])
    end do
  end if
  if(family==8) then
    if(allocated(model%scorers)) deallocate(model%scorers)
    model%scorer_func=>special_score;model%refit_metric=''
  end if
  fit_calls=0;set_calls=0
  if(family==9) then
    model%fit_func=>counted_fit;model%set_params_func=>counted_set
  end if
  call train()
  allocate(prediction(0,0),prob(0,0),decision(0,0));score=ieee_value(0._dp,ieee_quiet_nan)
  if(model%refit) then
    if(classification==1) then
      labels=model%predict_class(q);prediction=reshape(real(labels,dp),[m,1]);prob=model%predict_proba(q)
      if(family==2) decision=model%decision_function(q)
      score=model%score(q,qi)
    else
      prediction=reshape(model%predict(q),[m,1])
      if(integer_target==1) then
        score=model%score(q,qi)
      else
        score=model%score(q,qr)
      end if
    end if
  end if
  write(*,*) model%best_index_,merge(1,0,model%best_score_available_),model%n_splits_,model%n_features_in_, &
    size(model%cv_results_%params),size(model%scorers_),merge(1,0,model%multimetric_),fit_calls,set_calls
  write(*,'(*(es26.17e3,1x))') model%best_score_,score
  do i=1,size(model%scorers_)
    write(*,'(a)') trim(model%scorers_(i)%name)
  end do
  call emit(model%cv_results_%mean_test_score);call emit(model%cv_results_%std_test_score)
  call emit(real(model%cv_results_%rank_test_score,dp))
  do i=1,size(model%scorers_)
    call emit(model%cv_results_%split_test_score(:,:,i))
  end do
  if(model%return_train_score) then
    call emit(model%cv_results_%mean_train_score);call emit(model%cv_results_%std_train_score)
    do i=1,size(model%scorers_)
      call emit(model%cv_results_%split_train_score(:,:,i))
    end do
  end if
  call emit(prediction);call emit(prob);call emit(decision)
  call emit(model%cv_results_%fit_time);call emit(model%cv_results_%score_time)
  do i=1,size(model%folds_)
    write(*,*) size(model%folds_(i)%train),size(model%folds_(i)%test)
    write(*,*) model%folds_(i)%train;write(*,*) model%folds_(i)%test
  end do
  do i=1,size(model%cv_results_%params)
    write(*,*) size(model%cv_results_%params(i)%items)
    do j=1,size(model%cv_results_%params(i)%items)
      associate(item=>model%cv_results_%params(i)%items(j))
        write(*,'(a)') item%name;value=item%value;write(*,*) value%kind
        select case(value%kind)
        case(1);write(*,'(es26.17e3)') value%real
        case(2);write(*,*) value%integer
        case(3);write(*,*) merge(1,0,value%logical)
        case(4);write(*,'(a)') value%text
        end select
      end associate
    end do
  end do
contains
  function random_alpha(rng) result(value)
    type(legacy_random_state), intent(inout) :: rng
    type(parameter_value) :: value
    value=param(.01_dp+10*rng%uniform())
  end function
  subroutine axis(name,values)
    character(*), intent(in) :: name
    type(parameter_value), intent(in) :: values(:)
    type(parameter_axis) :: entry
    entry%name=name;entry%values=values;model%param_grid(1)%axes=[model%param_grid(1)%axes,entry]
  end subroutine
  subroutine train()
    real(dp), allocatable :: w(:)
    if(weighted==1) w=weights
    if(classification==1.and.integer_target==0) then
      call model%fit(x,real(yi,dp),w)
    else if(integer_target==1.or.classification==1) then
      call model%fit(x,yi,w)
    else
      call model%fit(x,yr,w)
    end if
  end subroutine
  subroutine emit(values)
    real(dp), intent(in) :: values(:,:)
    integer :: row
    write(*,*) shape(values)
    do row=1,size(values,1)
      write(*,'(*(es26.17e3,1x))') values(row,:)
    end do
  end subroutine
  integer function last_candidate(results) result(index)
    type(search_cv_results), intent(in) :: results
    index=size(results%params)
  end function
  real(dp) function special_score(estimator,x,yr,yi) result(value)
    class(*), intent(in) :: estimator
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: yr(:)
    integer, optional, intent(in) :: yi(:)
    select type(estimator)
    type is(ridge)
      select case(nint(estimator%alpha))
      case(0);value=ieee_value(0._dp,ieee_quiet_nan)
      case(1);value=ieee_value(0._dp,ieee_positive_inf)
      case(2);value=ieee_value(0._dp,ieee_negative_inf)
      case(3);value=1
      end select
    end select
  end function
  subroutine counted_fit(estimator,x,yr,yi,w)
    class(*), intent(inout) :: estimator
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: yr(:),w(:)
    integer, optional, intent(in) :: yi(:)
    fit_calls=fit_calls+1;call fit_search_native(estimator,x,yr,yi,w)
  end subroutine
  subroutine counted_set(estimator,parameters)
    class(*), intent(inout) :: estimator
    type(parameter_set), intent(in) :: parameters
    set_calls=set_calls+1;call set_search_params(estimator,parameters)
  end subroutine
end program
