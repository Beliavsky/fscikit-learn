program reference_frozen_driver
  use sklearn, only: dp,frozen_estimator,frozen_parameters,pipeline,standard_scaler,minmax_scaler,robust_scaler, &
    polynomial_features,pca,truncated_svd,normalizer,binarizer,function_transformer,variance_threshold, &
    linear_regression,ridge,gaussian_nb,kneighbors_classifier,kneighbors_regressor,dummy_classifier,dummy_regressor, &
    logistic_regression,mlp_regressor,mlp_classifier,feature_string
  use sklearn_common, only: require
  use sklearn_transformer_protocol, only: dense_transformer
  use sklearn_feature_composition, only: transform_model,model_names,input_names
  use sklearn_base_dispatch, only: labels,regression,probabilities,decisions
  use sklearn_metrics, only: accuracy_score,r2_score
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_negative_inf
  implicit none
  class(*), allocatable, target :: base,other
  type(standard_scaler), target :: fixed_scales(2)
  type(gaussian_nb), target :: fixed_nb(2)
  type(frozen_estimator), target :: frozen
  class(frozen_estimator), pointer :: alias
  type(frozen_parameters) :: params,shallow
  real(dp), allocatable :: x(:,:),q(:,:),yr(:,:),qr(:,:),changed(:,:),yr_changed(:,:)
  integer, allocatable :: yc(:,:),qc(:,:),yc_changed(:,:)
  integer :: cases,index,code,n,p,nq,seed,variant,i,phase,kind
  real(dp) :: nan
  character(64) :: mode
  nan=ieee_value(0._dp,ieee_quiet_nan)
  call get_command_argument(1,mode)
  if(len_trim(mode)>0) then
    call rejection(trim(mode));error stop 'FrozenEstimator rejection unexpectedly returned'
  end if
  read(*,*) cases
  do index=1,cases
    read(*,*) code,n,p,nq,seed,variant
    allocate(x(n,p),q(nq,p),yr(n,3),qr(nq,3),yc(n,2),qc(nq,2))
    do i=1,n
      read(*,*) x(i,:)
    end do
    do i=1,n
      read(*,*) yr(i,:)
    end do
    do i=1,n
      read(*,*) yc(i,:)
    end do
    do i=1,nq
      read(*,*) q(i,:)
    end do
    do i=1,nq
      read(*,*) qr(i,:)
    end do
    do i=1,nq
      read(*,*) qc(i,:)
    end do
    call make_model(base,code,1,x,yr,yc,seed,variant)
    call make_model(other,code,2,x*.7_dp+2,yr*3+2,shift_labels(yc,code,202),seed+1,variant)
    frozen%estimator=>base;alias=>frozen%clone()
    changed=x+1;yr_changed=yr*2+1;yc_changed=shift_labels(yc,code,101)
    do phase=0,3
      if(phase==1) then
        call frozen%fit(nan,'ignored',sample_weight=nan)
        call frozen%fit(x(:0,:1),[integer::])
      else if(phase==2) then
        call fit_model(base,changed,yr_changed,yc_changed,code,.false.)
      else if(phase==3) then
        call alias%set_params(other)
      end if
      write(*,'(a,1x,i0,1x,i0)') 'BEGIN',index,phase
      params=frozen%get_params(.true.);shallow=frozen%get_params(.false.)
      call vector('identity',[real(merge(1,0,associated(alias,frozen)),dp), &
        real(merge(1,0,associated(params%estimator,shallow%estimator)),dp)])
      if(phase==3) then
        call vector('target_identity',[real(merge(1,0,associated(params%estimator,other)),dp)])
        call dump(frozen,other,code)
      else
        call vector('target_identity',[real(merge(1,0,associated(params%estimator,base)),dp)])
        call dump(frozen,base,code)
      end if
      write(*,'(a)') 'END';flush(6)
    end do
    nullify(frozen%estimator,alias);deallocate(base,other,x,q,yr,qr,yc,qc)
  end do
contains
  function shift_labels(y,code,offset) result(z)
    integer, intent(in) :: y(:,:),code,offset
    integer :: z(size(y,1),size(y,2))
    z=y
    if(code==21) then
      z=1-y
    else
      z(:,1)=y(:,1)+offset
    end if
  end function
  subroutine make_model(model,code,which,data,reg_y,class_y,seed,variant)
    class(*), allocatable, target, intent(out) :: model
    integer, intent(in) :: code,which,seed,variant,class_y(:,:)
    real(dp), intent(in) :: data(:,:),reg_y(:,:)
    type(frozen_estimator) :: view
    select case(code)
    case(0);allocate(standard_scaler::model)
    case(1);allocate(minmax_scaler::model)
    case(2);allocate(robust_scaler::model)
    case(3);allocate(polynomial_features::model)
    case(4);allocate(pca::model)
    case(5);allocate(truncated_svd::model)
    case(6);allocate(normalizer::model)
    case(7);allocate(binarizer::model)
    case(8);allocate(function_transformer::model)
    case(9);allocate(variance_threshold::model)
    case(10);allocate(linear_regression::model)
    case(11);allocate(ridge::model)
    case(12);allocate(gaussian_nb::model)
    case(13);allocate(kneighbors_classifier::model)
    case(14);allocate(kneighbors_regressor::model)
    case(15);allocate(dummy_classifier::model)
    case(16);allocate(dummy_regressor::model)
    case(17);allocate(logistic_regression::model)
    case(18,20);allocate(mlp_regressor::model)
    case(19,21);allocate(mlp_classifier::model)
    case(22:25)
      allocate(pipeline::model)
      select type(model)
      type is(pipeline)
        if(code==25) then
          call fixed_nb(which)%fit(data,class_y(:,1));view%estimator=>fixed_nb(which)
          allocate(model%steps(1));model%steps(1)%name='frozen'
          allocate(model%steps(1)%model,source=view)
        else
          allocate(model%steps(2));model%steps(1)%name='scale';model%steps(2)%name='model'
          if(code==24) then
            call fixed_scales(which)%fit(data);view%estimator=>fixed_scales(which)
            allocate(model%steps(1)%model,source=view)
          else
            allocate(model%steps(1)%model,source=standard_scaler())
          end if
          if(code==23) then
            allocate(model%steps(2)%model,source=gaussian_nb())
          else
            allocate(model%steps(2)%model,source=ridge())
          end if
        end if
      end select
    case(26)
      call fixed_nb(which)%fit(data,class_y(:,1));allocate(frozen_estimator::model)
      select type(model)
      type is(frozen_estimator);model%estimator=>fixed_nb(which)
      end select
    end select
    select type(model)
    type is(pca);model%n_components=2
    type is(truncated_svd);model%n_components=2;model%algorithm='full'
    type is(function_transformer);model%feature_names_out='one-to-one'
    type is(kneighbors_classifier);model%n_neighbors=3
    type is(kneighbors_regressor);model%n_neighbors=3
    type is(logistic_regression);model%c=.8_dp;model%max_iter=500;model%tol=1.e-12_dp
    type is(mlp_regressor)
      model%hidden_layer_sizes=[4];model%activation='tanh';model%max_iter=3;model%batch_size=5;model%random_state=seed
    type is(mlp_classifier)
      model%hidden_layer_sizes=[4];model%activation='tanh';model%max_iter=3;model%batch_size=5;model%random_state=seed
    end select
    call fit_model(model,data,reg_y,class_y,code,mod(variant,2)==0)
  end subroutine
  subroutine fit_model(model,data,reg_y,class_y,code,skip_stateless)
    class(*), intent(inout) :: model
    real(dp), intent(in) :: data(:,:),reg_y(:,:)
    integer, intent(in) :: class_y(:,:),code
    logical, intent(in) :: skip_stateless
    select type(model)
    type is(pipeline)
      if(code==23.or.code==25) then
        call model%fit(data,class_y(:,1))
      else
        call model%fit(data,reg_y(:,1))
      end if
    type is(frozen_estimator);call model%fit(data,class_y)
    class is(dense_transformer)
      if(.not.(skip_stateless.and.(code==6.or.code==7))) call model%fit_data(data)
    type is(function_transformer)
      if(.not.skip_stateless) call model%fit(data)
    type is(standard_scaler);call model%fit(data)
    type is(minmax_scaler);call model%fit(data)
    type is(robust_scaler);call model%fit(data)
    type is(pca);call model%fit(data)
    type is(truncated_svd);call model%fit(data)
    type is(variance_threshold);call model%fit(data)
    type is(linear_regression);call model%fit(data,reg_y(:,1))
    type is(ridge);call model%fit(data,reg_y(:,1))
    type is(gaussian_nb);call model%fit(data,class_y(:,1))
    type is(kneighbors_classifier);call model%fit(data,class_y(:,1))
    type is(kneighbors_regressor);call model%fit(data,reg_y(:,1))
    type is(dummy_classifier);call model%fit(data,class_y(:,1))
    type is(dummy_regressor);call model%fit(data,reg_y(:,1))
    type is(logistic_regression);call model%fit(data,class_y(:,1))
    type is(mlp_regressor)
      if(code==20) then
        call model%fit(data,reg_y)
      else
        call model%fit(data,reg_y(:,1))
      end if
    type is(mlp_classifier)
      if(code==21) then
        call model%fit(data,class_y)
      else
        call model%fit(data,class_y(:,1))
      end if
    end select
  end subroutine
  function bare_prediction(model,code) result(z)
    class(*), intent(in) :: model
    integer, intent(in) :: code
    real(dp), allocatable :: z(:,:),v(:)
    integer, allocatable :: ints(:)
    select type(model)
    type is(mlp_classifier)
      if(code==21) then
        z=real(model%predict_indicator(q),dp)
      else
        ints=model%predict(q);z=reshape(real(ints,dp),[nq,1])
      end if
    type is(mlp_regressor);z=model%predict_multioutput(q)
    class is(dense_transformer)
      if(model%bridge_kind()==1) then
        ints=model%bridge_labels(q);z=reshape(real(ints,dp),[nq,1])
      else
        z=model%bridge_matrix(q)
      end if
    class default
      if(kind==1) then
        ints=labels(model,q);z=reshape(real(ints,dp),[nq,1])
      else
        v=regression(model,q);z=reshape(v,[nq,1])
      end if
    end select
  end function
  function bare_proba(model,logarithm) result(z)
    class(*), intent(in) :: model
    logical, intent(in) :: logarithm
    real(dp), allocatable :: z(:,:)
    integer :: row,col
    if(logarithm) then
      select type(model)
      type is(gaussian_nb);z=model%predict_log_proba(q*1000);return
      type is(mlp_classifier);z=model%predict_log_proba(q*1000);return
      class is(dense_transformer);z=model%bridge_log_proba(q*1000);return
      end select
    end if
    select type(model)
    type is(mlp_classifier);z=model%predict_proba(q)
    class is(dense_transformer);z=model%bridge_proba(q)
    class default;z=probabilities(model,q)
    end select
    if(logarithm) then
      select type(model)
      class default;z=probabilities(model,q*1000)
      end select
      do col=1,size(z,2)
        do row=1,size(z,1)
          if(z(row,col)==0) then
            z(row,col)=ieee_value(0._dp,ieee_negative_inf)
          else
            z(row,col)=log(z(row,col))
          end if
        end do
      end do
    end if
  end function
  real(dp) function bare_score(model,code) result(value)
    class(*), intent(in) :: model
    integer, intent(in) :: code
    real(dp), allocatable :: z(:,:)
    integer :: j
    select type(model)
    type is(pca);value=model%score(q)
    type is(mlp_regressor)
      if(code==20) then
        value=model%score(q,qr)
      else
        value=model%score(q,qr(:,1))
      end if
    type is(mlp_classifier)
      if(code==21) then
        value=model%score(q,qc)
      else
        value=model%score(q,qc(:,1))
      end if
    class is(dense_transformer)
      if(kind==1) then
        value=model%bridge_score_integer(q,qc(:,1))
      else
        value=model%bridge_score_real(q,qr(:,1))
      end if
    class default
      z=bare_prediction(model,code)
      if(kind==1) then
        value=accuracy_score(qc(:,1),int(z(:,1)))
      else
        value=r2_score(qr(:,1),z(:,1))
      end if
    end select
  end function
  subroutine dump(view,model,code)
    type(frozen_estimator), intent(in) :: view
    class(*), intent(in) :: model
    integer, intent(in) :: code
    character(24), parameter :: methods(11)=[character(24)::'transform','inverse_transform','predict', &
      'predict_proba','predict_log_proba','decision_function','score','get_feature_names_out', &
      'fit_transform','fit_predict','score_samples']
    real(dp) :: caps(13),score
    real(dp), allocatable :: z(:,:),b(:,:),v(:)
    type(feature_string), allocatable :: features(:),names(:)
    integer :: j
    kind=view%bridge_kind();caps(1)=real(merge(1,0,view%is_fitted()),dp)
    do j=1,11
      caps(j+1)=real(merge(1,0,view%has_method(trim(methods(j)))),dp)
    end do
    caps(13)=real(kind,dp);call vector('capabilities',caps)
    call vector('features',[real(view%bridge_features(),dp)])
    if(kind==1) then
      call vector('classes',real(view%get_classes(),dp))
    else
      call vector('classes',[real(dp)::])
    end if
    z=reshape([real(dp)::],[0,0]);b=z
    if(view%has_method('predict')) then
      if(kind==1) then
        if(code==21) then
          z=real(view%predict_indicator(q),dp)
        else
          z=reshape(real(view%predict_class(q),dp),[nq,1])
        end if
      else
        z=view%predict_matrix(q)
      end if
      b=bare_prediction(model,code)
    end if
    call pair('prediction',z,b)
    z=reshape([real(dp)::],[0,0]);b=z
    if(view%has_method('predict_proba')) then
      z=view%predict_proba(q);b=bare_proba(model,.false.)
    end if
    call pair('probability',z,b)
    z=reshape([real(dp)::],[0,0]);b=z
    if(view%has_method('predict_log_proba')) then
      z=view%predict_log_proba(q*1000);b=bare_proba(model,.true.)
    end if
    call pair('log_probability',z,b)
    z=reshape([real(dp)::],[0,0]);b=z
    if(view%has_method('decision_function')) then
      z=view%decision_function(q);b=decisions(model,q)
    end if
    call pair('decision',z,b)
    z=reshape([real(dp)::],[0,0]);b=z
    if(view%can_transform()) then
      z=view%transform(q);b=transform_model(model,q)
    end if
    call pair('transform',z,b)
    if(view%can_transform().and.view%can_inverse()) then
      z=view%inverse_transform(z)
      select type(model)
      class is(dense_transformer);b=model%inverse_data(b)
      type is(standard_scaler);b=model%inverse_transform(b)
      type is(minmax_scaler);b=model%inverse_transform(b)
      type is(robust_scaler);b=model%inverse_transform(b)
      type is(pca);b=model%inverse_transform(b)
      type is(truncated_svd);b=model%inverse_transform(b)
      type is(function_transformer);b=model%inverse_transform(b)
      type is(variance_threshold);b=model%inverse_transform(b)
      end select
    else
      z=reshape([real(dp)::],[0,0]);b=z
    end if
    call pair('inverse',z,b)
    if(view%has_method('score')) then
      if(code==4) then
        score=view%score(q)
      else if(kind==1) then
        if(code==21) then
          score=view%score(q,qc)
        else
          score=view%score(q,qc(:,1))
        end if
      else if(code==20) then
        score=view%score(q,qr)
      else
        score=view%score(q,qr(:,1))
      end if
      call vector('score',[score]);call vector('base_score',[bare_score(model,code)])
    else
      call vector('score',[real(dp)::]);call vector('base_score',[real(dp)::])
    end if
    if(view%has_method('score_samples')) then
      call vector('score_samples',view%score_samples(q))
      select type(model)
      type is(pca);call vector('base_score_samples',model%score_samples(q))
      end select
    else
      call vector('score_samples',[real(dp)::]);call vector('base_score_samples',[real(dp)::])
    end if
    features=input_names(p)
    if(view%can_transform().and.view%has_method('get_feature_names_out').and. &
       .not.((code==6.or.code==7).and.view%bridge_features()==0)) then
      names=view%get_feature_names_out(features)
      write(*,'(a,1x,i0)') 'NAMES',size(names)
      do j=1,size(names)
        write(*,'(a)') names(j)%text
      end do
    else
      write(*,'(a)') 'NAMES 0'
    end if
  end subroutine
  subroutine vector(name,a)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:)
    write(*,'(a,1x,*(es25.16e3,1x))') name,a
  end subroutine
  subroutine pair(name,a,b)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:,:),b(:,:)
    call vector(name,reshape(a,[size(a)]));call vector('base_'//name,reshape(b,[size(b)]))
  end subroutine
  subroutine rejection(mode)
    character(*), intent(in) :: mode
    type(standard_scaler), target :: scale
    type(linear_regression), target :: reg
    type(gaussian_nb), target :: nb
    type(frozen_estimator) :: view
    real(dp) :: data(4,2),target(4),score
    real(dp), allocatable :: z(:,:),v(:)
    integer, allocatable :: ints(:)
    integer, target :: not_model=1
    type(feature_string), allocatable :: names(:)
    data=reshape([0._dp,1._dp,2._dp,3._dp,1._dp,0._dp,1._dp,0._dp],[4,2]);target=data(:,1)
    select case(mode)
    case('unbound');call view%fit()
    case('unfitted');view%estimator=>scale;call view%fit(data,target)
    case('missing_predict');call scale%fit(data);view%estimator=>scale;v=view%predict(data)
    case('missing_transform');call reg%fit(data,target);view%estimator=>reg;z=view%transform(data)
    case('wrong_regression_kind');call nb%fit(data,[0,1,0,1]);view%estimator=>nb;v=view%predict(data)
    case('wrong_classifier_kind');call reg%fit(data,target);view%estimator=>reg;ints=view%predict_class(data)
    case('query_width');call scale%fit(data);view%estimator=>scale;z=view%transform(data(:,:1))
    case('query_nan');call reg%fit(data,target);view%estimator=>reg;data(1,1)=nan;v=view%predict(data)
    case('names_width');call scale%fit(data);view%estimator=>scale;names=view%get_feature_names_out(input_names(1))
    case('score_count');call nb%fit(data,[0,1,0,1]);view%estimator=>nb;score=view%score(data,[0])
    case('wrong_classes');call reg%fit(data,target);view%estimator=>reg;ints=view%get_classes()
    case('unsupported');view%estimator=>not_model;call view%fit()
    case default;error stop 'unknown frozen rejection'
    end select
  end subroutine
end program
