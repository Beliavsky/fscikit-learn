program reference_centroid_priors_driver
  use sklearn_neighbors, only: nearest_centroid
  use sklearn_common, only: dp
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(nearest_centroid) :: model
  real(dp), allocatable :: x(:,:),q(:,:),original(:,:),values(:,:)
  integer, allocatable :: y(:)
  real(dp) :: small(4,2),query(1,2)
  integer :: n,p,m,metric,mode,k,actions
  character(40) :: argument
  call get_command_argument(1,argument)
  if(len_trim(argument)>0) then
    small(1,:)=[-2._dp,-1._dp];small(2,:)=[-1._dp,0._dp]
    small(3,:)=[1._dp,0._dp];small(4,:)=[2._dp,1._dp];query=0
    select case(trim(argument))
    case('unfitted_predict');values=reshape(real(model%predict(query),dp),[1,1])
    case('unfitted_decision');values=model%decision_function(query)
    case('unfitted_proba');values=model%predict_proba(query)
    case('one_class');call model%fit(small,[1,1,1,1])
    case('metric');model%metric='cosine';call model%fit(small,[0,0,1,1])
    case('prior_mode');model%prior_mode='invalid';call model%fit(small,[0,0,1,1])
    case('negative_shrink');model%shrink_threshold=-1;call model%fit(small,[0,0,1,1])
    case('nan_shrink');model%shrink_threshold=ieee_value(0._dp,ieee_quiet_nan);call model%fit(small,[0,0,1,1])
    case('inf_shrink');model%shrink_threshold=ieee_value(0._dp,ieee_positive_inf);call model%fit(small,[0,0,1,1])
    case('prior_size');model%priors=[1._dp];call model%fit(small,[0,0,1,1])
    case('prior_negative');model%priors=[-1._dp,2._dp];call model%fit(small,[0,0,1,1])
    case('prior_nan');model%priors=[ieee_value(0._dp,ieee_quiet_nan),1._dp];call model%fit(small,[0,0,1,1])
    case('prior_inf');model%priors=[ieee_value(0._dp,ieee_positive_inf),1._dp];call model%fit(small,[0,0,1,1])
    case('prior_range');model%priors=[huge(1._dp),1._dp];call model%fit(small,[0,0,1,1])
    case('data_range');small=huge(1._dp);call model%fit(small,[0,0,1,1])
    case('data_nan');small(1,1)=ieee_value(0._dp,ieee_quiet_nan);call model%fit(small,[0,0,1,1])
    case('constant');small=1;call model%fit(small,[0,0,1,1])
    case('empty');call model%fit(small(:0,:),[integer::])
    case('target_size');call model%fit(small,[0,1])
    case('width');call model%fit(small,[0,0,1,1]);values=model%predict_proba(query(:,:1))
    case('query_nan');call model%fit(small,[0,0,1,1]);query=ieee_value(0._dp,ieee_quiet_nan);values=model%predict_proba(query)
    case('query_empty');call model%fit(small,[0,0,1,1]);values=model%predict_proba(query(:0,:))
    case('query_range');call model%fit(small,[0,0,1,1]);query=huge(1._dp);values=model%predict_proba(query)
    case('manhattan_decision')
      model%metric='manhattan';call model%fit(small,[0,0,1,1]);values=model%decision_function(query)
    case('manhattan_proba')
      model%metric='manhattan';call model%fit(small,[0,0,1,1]);values=model%predict_proba(query)
    case('manhattan_log_proba')
      model%metric='manhattan';call model%fit(small,[0,0,1,1]);values=model%predict_log_proba(query)
    case('singleton_decision');call model%fit(small([1,4],:),[0,1]);values=model%decision_function(query)
    case('singleton_nonuniform')
      model%priors=[1._dp,2._dp];call model%fit(small([1,4],:),[0,1]);values=reshape(real(model%predict(query),dp),[1,1])
    case('undefined_shrink')
      small(:2,:)=-1;small(3:,:)=2;model%shrink_threshold=1
      call model%fit(small,[0,0,1,1]);values=model%predict_proba(query)
    case('standardized_range')
      small=small*1.e-100_dp;call model%fit(small,[0,0,1,1]);query=1.e100_dp;values=model%predict_proba(query)
    case default;error stop 'unknown centroid rejection case'
    end select
    error stop 'centroid invalid input accepted'
  end if
  read(*,*) n,p,m,metric,mode,k,actions,model%shrink_threshold
  allocate(x(n,p),q(m,p),y(n));read(*,*) x;read(*,*) y;read(*,*) q
  if(metric==1) model%metric='manhattan'
  if(mode==1) model%prior_mode='empirical'
  if(mode==2) then
    allocate(model%priors(k));read(*,*) model%priors
  end if
  original=x;call model%fit(x,y)
  call vector(real(model%classes_,dp));call matrix(model%centroids_)
  call vector(model%within_class_std_dev_);call matrix(model%deviations_)
  call vector(model%class_prior_);call vector([real(model%n_features_in_,dp)])
  if(btest(actions,0)) then
    call vector(real(model%predict(q),dp));call vector(real(model%predict(x),dp));call vector([model%score(x,y)])
  end if
  if(btest(actions,1)) then
    call matrix(model%decision_function(q));call matrix(model%predict_log_proba(q));call matrix(model%predict_proba(q))
  end if
  if(any(original/=x)) error stop 'centroid fit modified original data'
contains
  subroutine vector(value)
    real(dp), intent(in) :: value(:)
    call matrix(reshape(value,[size(value),1]))
  end subroutine
  subroutine matrix(value)
    real(dp), intent(in) :: value(:,:)
    print *,shape(value)
    print '(*(es25.16e3,1x))',value
  end subroutine
end program
