program model_selector_invalid_driver
  use sklearn_common, only: dp
  use sklearn_model_selection_features, only: select_from_model
  use sklearn_linear_model, only: linear_regression,logistic_regression
  use sklearn_naive_bayes, only: gaussian_nb
  use sklearn_feature_extraction, only: feature_string
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(select_from_model) :: model
  type(linear_regression) :: base
  type(logistic_regression) :: clf
  type(gaussian_nb) :: nb
  type(feature_string) :: names(3),bad_names(2)
  real(dp) :: x(6,3),y(6),nan,inf
  integer :: yi(6),i
  real(dp), allocatable :: z(:,:),scores(:)
  type(feature_string), allocatable :: output_names(:)
  logical, allocatable :: mask(:)
  character(64) :: test_case
  call get_command_argument(1,test_case)
  nan=ieee_value(0._dp,ieee_quiet_nan);inf=ieee_value(0._dp,ieee_positive_inf)
  x=reshape([(real(i,dp)/10,i=1,18)],[6,3]);y=[1._dp,2._dp,3._dp,4._dp,5._dp,6._dp]
  yi=[-4,9,-4,9,-4,9];base%coef_=[1._dp,2._dp,3._dp]
  do i=1,3
    names(i)%text=achar(96+i)
  end do
  model%prefit=.true.;allocate(model%estimator,source=base)
  select case(trim(test_case))
  case('missing_estimator')
    deallocate(model%estimator);call model%fit(x)
  case('not_fitted')
    model%prefit=.false.;mask=model%get_support()
  case('prefit_unfitted')
    deallocate(base%coef_);deallocate(model%estimator);allocate(model%estimator,source=base);call model%fit(x)
  case('missing_target')
    model%prefit=.false.;call model%fit(x)
  case('norm_zero')
    model%norm_order=0;call model%fit(x)
  case('norm_fraction')
    model%norm_order=.5_dp;call model%fit(x)
  case('norm_nan')
    model%norm_order=nan;call model%fit(x)
  case('negative_cap')
    model%max_features=-2;call model%fit(x)
  case('large_cap')
    model%max_features=4;call model%fit(x)
  case('negative_cap_callback','large_cap_callback')
    model%max_features_func=>cap;call model%fit(x)
  case('direct_cap_callback')
    model%max_features_func=>cap;mask=model%get_support()
  case('unknown_getter')
    model%importance_getter='unknown';call model%fit(x)
  case('missing_callback')
    model%importance_getter='callback';call model%fit(x)
  case('unchosen_callback')
    model%importance_func=>importance;call model%fit(x)
  case('empty_importance','nonfinite_importance','wrong_importance_width','vector_importance_rows')
    model%importance_getter='callback';model%importance_func=>importance;call model%fit(x)
  case('missing_feature_importance')
    model%importance_getter='feature_importances_'
  case('nonfinite_coef')
    select type(est=>model%estimator)
    type is(linear_regression)
      est%coef_(1)=nan
    end select
  case('bad_threshold')
    model%threshold='mode'
  case('bad_scale')
    model%threshold='hello*mean'
  case('missing_scale')
    model%threshold='*mean'
  case('missing_reference')
    model%threshold='2*'
  case('bad_reference')
    model%threshold='2*average'
  case('infinite_scale')
    model%threshold='Infinity*mean'
  case('nan_threshold')
    model%threshold='value';model%numeric_threshold=nan
  case('transform_width')
    z=model%transform(x(:,:2))
  case('inverse_width')
    z=model%inverse_transform(x)
  case('inverse_empty')
    z=model%inverse_transform(x(:,:0))
  case('empty_x')
    call model%fit(x(:0,:))
  case('nonfinite_x')
    x(1,1)=nan;call model%fit(x)
  case('direct_names')
    output_names=model%get_feature_names_out()
  case('name_count')
    call model%fit(x,feature_names=bad_names)
  case('name_unallocated')
    deallocate(names(2)%text);call model%fit(x,feature_names=names)
  case('name_mismatch')
    call model%fit(x,feature_names=names);names(1)%text='other';output_names=model%get_feature_names_out(names)
  case('target_count')
    model%prefit=.false.;call model%fit(x,y(:5))
  case('nonfinite_target')
    model%prefit=.false.;y(1)=nan;call model%fit(x,y)
  case('weight_count')
    model%prefit=.false.;call model%fit(x,y,sample_weight=y(:5))
  case('weight_nan')
    model%prefit=.false.;y(1)=nan;call model%fit(x,real(yi,dp),sample_weight=y)
  case('weight_negative')
    model%prefit=.false.;y(1)=-1;call model%fit(x,real(yi,dp),sample_weight=y)
  case('matrix_base')
    model%prefit=.false.;call model%fit(x,reshape(y,[6,1]))
  case('partial_base')
    model%prefit=.false.;call model%partial_fit(x,y)
  case('classifier_continuous','classifier_large')
    model%prefit=.false.;deallocate(model%estimator);allocate(model%estimator,source=clf)
    y=real(yi,dp)
    if(test_case=='classifier_continuous') y(1)=.5_dp
    if(test_case=='classifier_large') y(1)=real(huge(1),dp)+1
    call model%fit(x,y)
  case('no_auto_importance')
    model%prefit=.false.;deallocate(model%estimator);allocate(model%estimator,source=nb)
    call model%fit(x,yi)
  case default
    error stop 'unknown model selector invalid case'
  end select
  z=model%transform(x)
  error stop 'model selector invalid case unexpectedly accepted'
contains
  integer function cap(data) result(count)
    real(dp), intent(in) :: data(:,:)
    count=-2
    if(test_case=='large_cap_callback') count=4
  end function
  subroutine importance(estimator,values,matrix)
    class(*), intent(in) :: estimator
    real(dp), allocatable, intent(out) :: values(:,:)
    logical, intent(out) :: matrix
    matrix=.false.;values=reshape([1._dp,2._dp,3._dp],[1,3])
    select case(trim(test_case))
    case('empty_importance');values=reshape([real(dp)::],[1,0])
    case('nonfinite_importance');values(1,1)=nan
    case('wrong_importance_width');values=reshape([1._dp,2._dp],[1,2])
    case('vector_importance_rows');values=reshape([1._dp,2._dp,3._dp,4._dp,5._dp,6._dp],[2,3])
    end select
  end subroutine
end program
