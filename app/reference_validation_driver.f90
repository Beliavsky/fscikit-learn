program reference_validation_driver
  use sklearn_validation
  use sklearn_common, only: dp
  use sklearn_linear_model, only: ridge,elastic_net,logistic_regression
  use sklearn_naive_bayes, only: gaussian_nb
  use sklearn_neighbors, only: kneighbors_regressor,kneighbors_classifier
  use sklearn_pipeline, only: pipeline
  use sklearn_preprocessing, only: standard_scaler
  use sklearn_feature_selection, only: select_k_best
  use sklearn_search_adapters, only: fit_search_native,search_regression,search_labels,search_probabilities,search_classes
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  class(*), allocatable :: estimator
  type(ridge) :: reg
  type(elastic_net) :: elastic
  type(logistic_regression) :: classifier
  type(gaussian_nb) :: bayes
  type(kneighbors_regressor) :: neighbor_reg
  type(kneighbors_classifier) :: neighbor_clf
  type(pipeline) :: chain
  type(standard_scaler) :: scale
  type(select_k_best) :: selector
  type(cross_validation_options) :: options,score_options
  type(cross_validation_result) :: results
  real(dp), allocatable :: x(:,:),q(:,:),yr(:),weights(:),w(:),prediction(:,:),probability(:,:),score_vector(:)
  integer, allocatable :: yi(:),labels(:),classes(:)
  integer :: mode,family,config,n,p,m,cv,weighted,integer_target,method,i,j,k,classification
  integer :: fit_calls=0,score_calls=0,response_calls=0,encoded_min=huge(1),encoded_max=-huge(1)
  logical :: no_target
  read(*,*) mode,family,config,n,p,m,cv,weighted,integer_target,method
  allocate(x(n,p),q(m,p),yr(n),weights(n),yi(n))
  read(*,*) x;read(*,*) q;read(*,*) yr;read(*,*) yi;read(*,*) weights
  classification=merge(1,0,any(family==[2,3,5,7]));no_target=family==9
  reg%alpha=.3_dp;elastic%alpha=.08_dp;elastic%l1_ratio=.4_dp;elastic%tol=1.e-11_dp
  classifier%c=.8_dp;classifier%tol=1.e-10_dp
  neighbor_reg%n_neighbors=3;neighbor_clf%n_neighbors=3
  select case(family)
  case(0,10,11);allocate(estimator,source=reg)
  case(1);allocate(estimator,source=elastic)
  case(2);allocate(estimator,source=classifier)
  case(3);allocate(estimator,source=bayes)
  case(4);allocate(estimator,source=neighbor_reg)
  case(5);allocate(estimator,source=neighbor_clf)
  case(6,7,8)
    allocate(chain%steps(merge(3,2,family==8)))
    chain%steps(1)%name='scale';allocate(chain%steps(1)%model,source=scale)
    if(family==8) then
      selector%k=2;chain%steps(2)%name='select';allocate(chain%steps(2)%model,source=selector)
    end if
    k=size(chain%steps);chain%steps(k)%name='model'
    if(family==7) then
      allocate(chain%steps(k)%model,source=classifier)
    else
      allocate(chain%steps(k)%model,source=reg)
    end if
    allocate(estimator,source=chain)
  case(9);allocate(estimator,source=scale)
  end select
  options%cv=cv
  if(weighted==1) w=weights
  if(family==10) options%fit_func=>counted_fit
  if(mode==0) then
    options%return_train_score=mod(config,2)==1.or.config==2.or.config==6
    options%return_estimator=config/=0.and.config/=3
    options%return_indices=config/=0.and.config/=2
    if(classification==1) then
      options%scoring='accuracy'
      if(config==4) options%scoring='balanced_accuracy'
      if(config==5) options%scoring='f1_macro'
    else
      if(config==1.or.config==4) options%scoring='neg_mean_squared_error'
      if(config==5) options%scoring='neg_root_mean_squared_error'
    end if
    if(config==2) then
      allocate(options%scorers(2));options%scorers(1)%name='primary';options%scorers(2)%name='secondary'
      if(classification==1) then
        options%scorers(1)%scoring='accuracy';options%scorers(2)%scoring='neg_log_loss'
      else
        options%scorers(1)%scoring='neg_mean_squared_error';options%scorers(2)%scoring='r2'
      end if
    end if
    if(config==3) options%scorer_func=>mean_error
    if(family==9.or.family==11) then
      if(allocated(options%scorers)) deallocate(options%scorers)
      if(family==9) options%scorer_func=>scaled_moment
      if(family==11) options%scorer_func=>nonfinite_score
    end if
    if(config==4.or.config==6) then
      allocate(options%cv_splits(merge(4,3,config==6)))
      do i=1,size(options%cv_splits)
        if(config==6) then
          options%cv_splits(i)%test=[i]
        else
          options%cv_splits(i)%test=[i,(j,j=i,n,5)]
        end if
        options%cv_splits(i)%train=pack([(j,j=1,n)],.not.[(any(options%cv_splits(i)%test==j),j=1,n)])
      end do
    end if
    if(no_target) then
      results=cross_validate(estimator,x,options,w)
    else if(integer_target==1) then
      results=cross_validate(estimator,x,yi,options,w)
    else if(classification==1) then
      results=cross_validate(estimator,x,real(yi,dp),options,w)
    else
      results=cross_validate(estimator,x,yr,options,w)
    end if
    allocate(score_vector(0))
    if(config==3) then
      score_options=options
      if(no_target) then
        score_vector=cross_val_score(estimator,x,score_options,w)
      else if(integer_target==1) then
        score_vector=cross_val_score(estimator,x,yi,score_options,w)
      else if(classification==1) then
        score_vector=cross_val_score(estimator,x,real(yi,dp),score_options,w)
      else
        score_vector=cross_val_score(estimator,x,yr,score_options,w)
      end if
    end if
    write(*,*) results%n_splits,results%n_features_in,size(results%metric_names), &
      merge(1,0,allocated(results%estimators)),merge(1,0,allocated(results%indices)),fit_calls,score_calls
    do i=1,size(results%metric_names)
      write(*,'(a)') trim(results%metric_names(i))
    end do
    call emit(results%test_score)
    if(allocated(results%train_score)) then
      call emit(results%train_score)
    else
      call emit(reshape([real(dp)::],[0,0]))
    end if
    write(*,*) size(score_vector);write(*,'(*(es26.17e3,1x))') score_vector
    write(*,'(*(es26.17e3,1x))') results%fit_time;write(*,'(*(es26.17e3,1x))') results%score_time
    if(allocated(results%indices)) then
      do i=1,size(results%indices)
        write(*,*) size(results%indices(i)%train),size(results%indices(i)%test)
        write(*,*) results%indices(i)%train;write(*,*) results%indices(i)%test
      end do
    end if
    if(allocated(results%estimators)) then
      do i=1,size(results%estimators)
        associate(model=>results%estimators(i)%model)
          if(classification==1) then
            labels=search_labels(model,q);prediction=reshape(real(labels,dp),[m,1])
            probability=search_probabilities(model,q);classes=search_classes(model)
          else if(no_target) then
            select type(model)
            type is(standard_scaler);prediction=model%transform(q)
            end select
            allocate(probability(0,0),classes(0))
          else
            prediction=reshape(search_regression(model,q),[m,1]);allocate(probability(0,0),classes(0))
          end if
          call emit(prediction);call emit(probability);write(*,*) size(classes);write(*,*) classes
          deallocate(prediction,probability,classes)
        end associate
      end do
    end if
  else
    select case(method)
    case(0);options%method='predict'
    case(1);options%method='predict_proba'
    case(2);options%method='decision_function'
    case(3);options%method='predict_log_proba'
    end select
    if(config==4) then
      allocate(options%cv_splits(3))
      do i=1,3
        options%cv_splits(i)%test=[(j,j=n-i+1,1,-3)]
        options%cv_splits(i)%train=pack([(j,j=1,n)],.not.[(any(options%cv_splits(i)%test==j),j=1,n)])
      end do
    end if
    if(classification==1) options%fit_func=>counted_fit
    if(family==10) options%response_func=>counted_response
    if(classification==1.and.integer_target==0) then
      prediction=cross_val_predict(estimator,x,real(yi,dp),options,w)
    else if(classification==1.or.integer_target==1) then
      prediction=cross_val_predict(estimator,x,yi,options,w)
    else
      prediction=cross_val_predict(estimator,x,yr,options,w)
    end if
    write(*,*) fit_calls,response_calls,encoded_min,encoded_max
    call emit(prediction)
  end if
contains
  subroutine emit(values)
    real(dp), intent(in) :: values(:,:)
    integer :: row
    write(*,*) shape(values)
    do row=1,size(values,1)
      write(*,'(*(es26.17e3,1x))') values(row,:)
    end do
  end subroutine
  subroutine counted_fit(model,x,yr,yi,sample_weight)
    class(*), intent(inout) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: yr(:),sample_weight(:)
    integer, optional, intent(in) :: yi(:)
    fit_calls=fit_calls+1
    if(present(yi)) then
      encoded_min=min(encoded_min,minval(yi));encoded_max=max(encoded_max,maxval(yi))
    end if
    call fit_search_native(model,x,yr,yi,sample_weight)
  end subroutine
  function counted_response(model,x,method) result(values)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    character(*), intent(in) :: method
    real(dp), allocatable :: values(:,:)
    response_calls=response_calls+1;values=reshape(search_regression(model,x),[size(x,1),1])
  end function
  real(dp) function mean_error(model,x,yr,yi) result(value)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: yr(:)
    integer, optional, intent(in) :: yi(:)
    real(dp), allocatable :: values(:)
    integer, allocatable :: labels(:)
    score_calls=score_calls+1
    if(present(yi)) then
      labels=search_labels(model,x);value=sum(real(labels-yi,dp))/size(yi)
    else
      values=search_regression(model,x);value=sum(values-yr)/size(yr)
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
  real(dp) function nonfinite_score(model,x,yr,yi) result(value)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: yr(:)
    integer, optional, intent(in) :: yi(:)
    score_calls=score_calls+1;value=ieee_value(0._dp,ieee_quiet_nan)
    if(mod(size(x,1),2)==0) value=ieee_value(0._dp,ieee_positive_inf)
  end function
end program
