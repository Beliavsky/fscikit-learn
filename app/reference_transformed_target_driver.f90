program reference_transformed_target_driver
  use sklearn_compose
  use sklearn_function_transformer
  use sklearn_preprocessing, only: standard_scaler,minmax_scaler,maxabs_scaler,robust_scaler
  use sklearn_power_transform, only: power_transformer
  use sklearn_quantile_transform, only: quantile_transformer
  use sklearn_linear_model, only: linear_regression,ridge
  use sklearn_neighbors, only: kneighbors_regressor
  use sklearn_tree, only: decision_tree_regressor
  use sklearn_dummy, only: dummy_regressor
  use sklearn_common, only: dp
  use reference_transform_functions
  implicit none
  type(transformed_target_regressor) :: model
  type(standard_scaler) :: standard
  type(minmax_scaler) :: minmax
  type(maxabs_scaler) :: maxabs
  type(robust_scaler) :: robust
  type(power_transformer) :: power
  type(quantile_transformer) :: quantile
  type(function_transformer) :: functions
  type(ridge) :: ridge_model
  type(kneighbors_regressor) :: knn
  type(decision_tree_regressor) :: tree
  type(dummy_regressor) :: dummy
  real(dp), allocatable :: x(:,:),y(:,:),q(:,:),w(:),transformed(:,:),predicted(:,:)
  integer :: n,p,k,m,base,trans,rank,weighted,check,roundtrip,j,ft_width
  real(dp) :: alpha
  read(*,*) n,p,k,m,base,trans,callback_mode,rank,weighted,check,alpha
  allocate(x(n,p),y(n,k),q(m,p),w(n));read(*,*) x;read(*,*) y;read(*,*) q;read(*,*) w
  model%check_inverse=check==1
  select case(base)
  case(1);ridge_model%alpha=alpha;allocate(model%regressor,source=ridge_model)
  case(2);knn%n_neighbors=min(3,n);knn%weights='distance';allocate(model%regressor,source=knn)
  case(3);tree%max_depth=3;tree%random_state=0;allocate(model%regressor,source=tree)
  case(4);dummy%strategy='mean';allocate(model%regressor,source=dummy)
  end select
  select case(trans)
  case(0)
    if(callback_mode/=0) then
      model%func=>forward;model%inverse_func=>backward
    end if
  case(1);allocate(model%transformer,source=standard)
  case(2);minmax%feature_min=-2;minmax%feature_max=3;allocate(model%transformer,source=minmax)
  case(3);allocate(model%transformer,source=maxabs)
  case(4);allocate(model%transformer,source=robust)
  case(5);power%method='yeo-johnson';allocate(model%transformer,source=power)
  case(6);quantile%n_quantiles=min(17,n);quantile%subsample=0;allocate(model%transformer,source=quantile)
  case(7)
    functions%func=>forward;functions%inverse_func=>backward;functions%validate=.true.
    call functions%fit(reshape([(1._dp,j=1,2*(k+1))],[2,k+1]));functions%validate=.false.
    allocate(model%transformer,source=functions)
  end select
  if(rank==1) then
    if(weighted==1) then
      call model%fit(x,y(:,1),w)
    else
      call model%fit(x,y(:,1))
    end if
  else
    if(weighted==1) then
      call model%fit(x,y,w)
    else
      call model%fit(x,y)
    end if
  end if
  roundtrip=-1;ft_width=-1
  select type(fitted=>model%transformer_)
  type is(function_transformer)
    roundtrip=merge(1,0,fitted%inverse_consistent_);ft_width=fitted%n_features_in_
  end select
  print *,model%training_dim_,model%n_outputs_,model%n_transformed_outputs_,model%n_features_in_, &
    merge(1,0,model%inverse_checked_),merge(1,0,model%inverse_consistent_),roundtrip,ft_width
  select type(fitted=>model%transformer_)
  type is(function_transformer);transformed=fitted%transform(y)
  type is(standard_scaler);transformed=fitted%transform(y)
  type is(minmax_scaler);transformed=fitted%transform(y)
  type is(maxabs_scaler);transformed=fitted%transform(y)
  type is(robust_scaler);transformed=fitted%transform(y)
  type is(power_transformer);transformed=fitted%transform(y)
  type is(quantile_transformer);transformed=fitted%transform(y)
  end select
  print *,shape(transformed);print '(*(es25.16e3,1x))',transformed
  select type(fitted=>model%regressor_)
  type is(decision_tree_regressor)
    print *,fitted%node_count_
    do j=1,fitted%node_count_
      associate(a=>fitted%nodes_(j))
        print *,a%left,a%right,a%feature,a%n_samples,a%depth,merge(1,0,a%missing_go_left)
        print '(*(es25.16e3,1x))',a%threshold,a%weighted_n_samples,a%impurity
        print '(*(es25.16e3,1x))',a%value
      end associate
    end do
  class default
    print *,0
  end select
  transformed=model%transformed_predict(q);print *,shape(transformed)
  print '(*(es25.16e3,1x))',transformed
  predicted=model%predict_matrix(q);print *,shape(predicted);print '(*(es25.16e3,1x))',predicted
  predicted=model%predict_matrix(x);print *,shape(predicted);print '(*(es25.16e3,1x))',predicted
  if(rank==1) then
    print '(*(es25.16e3,1x))',model%score(x,y(:,1)),model%score(x,y(:,1),w)
  else
    print '(*(es25.16e3,1x))',model%score(x,y),model%score(x,y,w)
  end if
  select type(fitted=>model%transformer_)
  type is(standard_scaler)
    print '(*(es25.16e3,1x))',fitted%mean_,fitted%var_,fitted%scale_
  type is(minmax_scaler)
    print '(*(es25.16e3,1x))',fitted%data_min_,fitted%data_max_,fitted%scale_,fitted%min_
  type is(maxabs_scaler)
    print '(*(es25.16e3,1x))',fitted%scale_
  type is(robust_scaler)
    print '(*(es25.16e3,1x))',fitted%center_,fitted%scale_
  type is(power_transformer)
    print '(*(es25.16e3,1x))',fitted%lambdas_,fitted%mean_,fitted%scale_
  type is(quantile_transformer)
    print '(*(es25.16e3,1x))',fitted%quantiles_,fitted%references_
  class default
    print *
  end select
end program
