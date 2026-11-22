program test_validation
  use sklearn_validation
  use sklearn_common, only: dp,require
  use sklearn_linear_model, only: ridge
  use sklearn_pipeline, only: pipeline
  use sklearn_preprocessing, only: standard_scaler
  use sklearn_feature_selection, only: select_k_best,f_classif
  use sklearn_search_adapters, only: fit_search_native
  implicit none
  type(ridge) :: base
  type(cross_validation_options) :: options
  type(cross_validation_result) :: results,copy
  type(pipeline) :: chain
  type(standard_scaler) :: scale
  type(select_k_best) :: selector
  real(dp) :: x(36,3),y(36),weights(36)
  real(dp), allocatable :: scores(:),prediction(:),expected(:),selected_scores(:)
  integer :: i,fit_calls
  integer :: labels(36)
  do i=1,36
    x(i,:)=[real(i,dp)/20,sin(real(i,dp)),cos(real(2*i,dp))]
  end do
  y=2*x(:,1)-x(:,2)+.2_dp*x(:,3);weights=1;weights(:3)=0;base%alpha=.3_dp
  options%cv=3;options%return_train_score=.true.;options%return_indices=.true.;options%return_estimator=.true.
  options%scoring='neg_mean_squared_error';options%fit_func=>counted_fit;fit_calls=0
  results=cross_validate(base,x,y,options,weights)
  call require(fit_calls==3,'cross_validate fits each fold once without full refit')
  call require(all(shape(results%test_score)==[3,1]),'CV score shape')
  call require(all(results%fit_time>=0).and.all(results%score_time>=0),'CV nonnegative timings')
  call require(.not.allocated(base%coef_),'CV leaves caller estimator unfitted')
  do i=1,3
    select type(model=>results%estimators(i)%model)
    type is(ridge)
      prediction=model%predict(x(results%indices(i)%test,:))
      expected=y(results%indices(i)%test)
      call require(abs(results%test_score(i,1)+sum((prediction-expected)**2)/size(expected))<1.e-12_dp, &
        'stored CV estimator reproduces held-out score')
    end select
  end do
  copy=results
  select type(model=>results%estimators(1)%model)
  type is(ridge);model%coef_=0
  end select
  select type(model=>copy%estimators(1)%model)
  type is(ridge);call require(any(model%coef_/=0),'returned model copies are independent')
  end select
  options%fit_func=>null();allocate(options%scorers(2))
  options%scorers(1)%name='mse';options%scorers(1)%scoring='neg_mean_squared_error'
  options%scorers(2)%name='r2';options%scorers(2)%scoring='r2'
  results=cross_validate(base,x,y,options)
  call require(size(results%metric_names)==2.and.all(results%train_score(:,2)>.99_dp),'multiple CV scorers')
  deallocate(options%scorers);scores=cross_val_score(base,x,y,options)
  call require(maxval(abs(scores-results%test_score(:,1)))<1.e-12_dp,'general scalar CV score wrapper')
  options=cross_validation_options();options%scorer_func=>scaled_variance
  options%return_estimator=.true.;results=cross_validate(scale,x,options)
  call require(all(results%test_score>0),'no-target transformer fitting and custom scoring')
  allocate(chain%steps(2));chain%steps(1)%name='select';chain%steps(2)%name='regress'
  selector%k=2;allocate(chain%steps(1)%model,source=selector);allocate(chain%steps(2)%model,source=base)
  labels=mod([(i,i=1,36)],3);options=cross_validation_options();options%return_estimator=.true.
  results=cross_validate(chain,x,labels,options)
  select type(model=>results%estimators(1)%model)
  type is(pipeline)
    select type(child=>model%steps(1)%model)
    type is(select_k_best)
      selected_scores=child%scores_
    end select
  end select
  ! Default KFold puts rows 1:8 in the first held-out fold.
  expected=f_classif(x(9:,:),labels(9:))
  call require(maxval(abs(selected_scores-expected))<1.e-12_dp,'integer regression targets reach Pipeline selectors')
  print *, 'Cross-validation results and scorers: PASS'
contains
  subroutine counted_fit(model,x,yr,yi,sample_weight)
    class(*), intent(inout) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: yr(:),sample_weight(:)
    integer, optional, intent(in) :: yi(:)
    fit_calls=fit_calls+1;call fit_search_native(model,x,yr,yi,sample_weight)
  end subroutine
  real(dp) function scaled_variance(model,x,yr,yi) result(score)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: yr(:)
    integer, optional, intent(in) :: yi(:)
    real(dp), allocatable :: z(:,:)
    score=0
    select type(model)
    type is(standard_scaler)
      z=model%transform(x);score=sum(z*z)/size(z)
    end select
  end function
end program
