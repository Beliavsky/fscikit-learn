program reference_ransac_controls_driver
  use sklearn_common,only:dp
  use sklearn_linear_model,only:ransac_regressor,linear_regression
  use sklearn_sparse,only:csr_matrix,csr_from_dense,csr_from_arrays
  use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan
  implicit none
  type(ransac_regressor)::model
  type(csr_matrix)::sparse,queries
  real(dp),allocatable::x(:,:),y(:,:),w(:),query(:,:),prediction(:,:),train(:,:)
  integer,allocatable::trace(:,:)
  integer::cases,c,n,p,t,q,style,target_kind,weighted,intercept,seed,min_kind,min_int
  integer::max_trials,max_skips,stop_n,loss_mode,data_mode,model_mode,data_calls,model_calls,loss_calls,i,j
  real(dp)::min_real,threshold,probability,stop_score,scores(2)
  read(*,*)cases
  do c=1,cases
    read(*,*)n,p,t,q,style,target_kind,weighted,intercept,seed,min_kind,min_int,min_real, &
      max_trials,max_skips,stop_n,threshold,probability,stop_score,loss_mode,data_mode,model_mode
    allocate(x(n,p),y(n,t),w(n),query(q,p),trace(n,max_trials));trace=0
    read(*,*)x;read(*,*)y;read(*,*)w;read(*,*)query
    model=ransac_regressor();model%fit_intercept=intercept/=0;model%random_state=seed
    model%max_trials=max_trials;model%max_skips=max_skips;model%stop_n_inliers=stop_n
    model%residual_threshold=threshold;model%stop_probability=probability;model%stop_score=stop_score
    if(min_kind==1)call model%set_min_samples(min_int)
    if(min_kind==2)call model%set_min_samples(min_real)
    if(loss_mode==1)model%loss='squared_error'
    if(loss_mode>=2)model%loss_callback=>custom_loss
    model%is_data_valid=>valid_data;model%is_model_valid=>valid_model
    data_calls=0;model_calls=0;loss_calls=0
    sparse=matrix(x,style);queries=matrix(query,style)
    if(style==0)then
      if(target_kind==0)then
        if(weighted==0)then
          call model%fit(x,y(:,1))
        else if(weighted==1)then
          call model%fit(x,y(:,1),w)
        else
          call model%fit(x,y(:,1),w(1))
        end if
      else
        if(weighted==0)then
          call model%fit(x,y)
        else if(weighted==1)then
          call model%fit(x,y,w)
        else
          call model%fit(x,y,w(1))
        end if
      end if
      prediction=model%predict_matrix(query);train=model%predict_matrix(x)
      if(target_kind==0)then
        scores=[model%score(x,y(:,1)),model%score(x,y(:,1),abs(w)+.5_dp)]
      else
        scores=[model%score(x,y),model%score(x,y,abs(w)+.5_dp)]
      end if
    else
      if(target_kind==0)then
        if(weighted==0)then
          call model%fit(sparse,y(:,1))
        else if(weighted==1)then
          call model%fit(sparse,y(:,1),w)
        else
          call model%fit(sparse,y(:,1),w(1))
        end if
      else
        if(weighted==0)then
          call model%fit(sparse,y)
        else if(weighted==1)then
          call model%fit(sparse,y,w)
        else
          call model%fit(sparse,y,w(1))
        end if
      end if
      prediction=model%predict_matrix(queries);train=model%predict_matrix(sparse)
      if(target_kind==0)then
        scores=[model%score(sparse,y(:,1)),model%score(sparse,y(:,1),abs(w)+.5_dp)]
      else
        scores=[model%score(sparse,y),model%score(sparse,y,abs(w)+.5_dp)]
      end if
    end if
    print '(*(i0,1x))',model%n_features_in_,model%n_targets_,model%n_trials_,model%n_skips_no_inliers_, &
      model%n_skips_invalid_data_,model%n_skips_invalid_model_,model%min_samples_,merge(1,0,model%skips_exceeded_)
    print '(*(es25.16e3,1x))',transpose(model%coef_matrix_)
    print '(*(es25.16e3,1x))',model%intercepts_
    print '(*(es25.16e3,1x))',transpose(prediction)
    print '(*(es25.16e3,1x))',transpose(train)
    print '(*(i0,1x))',merge(1,0,model%inlier_mask_)
    print '(*(es25.16e3,1x))',model%residual_threshold_
    print '(*(es25.16e3,1x))',scores
    print '(*(i0,1x))',((trace(i,j),i=1,model%min_samples_),j=1,data_calls)
    print '(*(i0,1x))',data_calls,model_calls,loss_calls
    deallocate(x,y,w,query,trace)
  end do
contains
  function matrix(x,style)result(csr)
    real(dp),intent(in)::x(:,:)
    integer,intent(in)::style
    type(csr_matrix)::csr
    integer,allocatable::ptr(:),indices(:)
    real(dp),allocatable::data(:)
    integer::i,j,k,n,p
    if(style<2)then
      csr=csr_from_dense(x);return
    end if
    n=size(x,1);p=size(x,2);allocate(ptr(n+1),indices(2*n*p+n),data(2*n*p+n));k=0;ptr(1)=1
    do i=1,n
      do j=p,1,-1
        if(x(i,j)==0)cycle
        k=k+1;indices(k)=j;data(k)=.25_dp*x(i,j)
        k=k+1;indices(k)=j;data(k)=.75_dp*x(i,j)
      end do
      k=k+1;indices(k)=1;data(k)=0;ptr(i+1)=k+1
    end do
    csr=csr_from_arrays(n,p,ptr,indices(:k),data(:k))
  end function
  logical function valid_data(a,b)result(ok)
    real(dp),intent(in)::a(:,:),b(:,:)
    data_calls=data_calls+1
    trace(:size(a,1),data_calls)=nint(a(:,1))+1
    ok=.true.
    if(data_mode==1)ok=mod(nint(a(1,1)),3)/=0
    if(data_mode==2)ok=.false.
  end function
  logical function valid_model(candidate,a,b)result(ok)
    type(linear_regression),intent(in)::candidate
    real(dp),intent(in)::a(:,:),b(:,:)
    model_calls=model_calls+1;ok=.true.
    if(model_mode==1)ok=maxval(abs(candidate%coef_matrix_))<3
    if(model_mode==2)ok=.false.
  end function
  function custom_loss(actual,predicted)result(residual)
    real(dp),intent(in)::actual(:,:),predicted(:,:)
    real(dp)::residual(size(actual,1))
    loss_calls=loss_calls+1;residual=.75_dp*sum(abs(actual-predicted),dim=2)
    if(loss_mode==3)residual(1)=ieee_value(0._dp,ieee_quiet_nan)
  end function
end program
