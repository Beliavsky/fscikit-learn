module reference_pipeline_models
  use sklearn_pipeline
  use sklearn_common, only: dp
  use sklearn_feature_extraction, only: feature_string
  use sklearn_function_transformer, only: function_transformer
  use sklearn_preprocessing
  use sklearn_decomposition, only: pca
  use sklearn_feature_selection, only: select_k_best,select_percentile,variance_threshold
  use sklearn_compose, only: column_transformer,transformed_target_regressor
  use sklearn_linear_model, only: linear_regression,ridge,logistic_regression,poisson_regressor
  use sklearn_naive_bayes, only: gaussian_nb
  use sklearn_neighbors, only: kneighbors_regressor,kneighbors_classifier
  implicit none
  integer :: forward_calls=0,inverse_calls=0
contains
  function affine(x) result(z)
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    forward_calls=forward_calls+1;z=2*x+3
  end function
  function undo(x) result(z)
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    inverse_calls=inverse_calls+1;z=(x-3)/2
  end function
  function expand(x) result(z)
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    forward_calls=forward_calls+1;z=reshape([x,x*x],[size(x,1),2*size(x,2)])
  end function
  function expand_names(model,input) result(names)
    class(*), intent(in) :: model
    type(feature_string), intent(in) :: input(:)
    type(feature_string), allocatable :: names(:)
    integer :: i,p
    p=size(input);allocate(names(2*p))
    do i=1,p
      names(i)%text=input(i)%text//'_original';names(p+i)%text=input(i)%text//'_square'
    end do
  end function
  subroutine add(pipe,name,model,pass)
    type(pipeline), intent(inout) :: pipe
    character(*), intent(in) :: name
    class(*), optional, intent(in) :: model
    logical, optional, intent(in) :: pass
    type(pipeline_step) :: step
    if(.not.allocated(pipe%steps)) allocate(pipe%steps(0))
    step%name=name
    if(present(pass)) then
      if(pass) step%action='passthrough'
    end if
    if(present(model)) allocate(step%model,source=model)
    pipe%steps=[pipe%steps,step]
  end subroutine
  subroutine configure(task,kind,config,pipe)
    integer, intent(in) :: task,kind,config
    type(pipeline), intent(out) :: pipe
    type(pipeline) :: child,final
    type(standard_scaler) :: standard
    type(minmax_scaler) :: minmax
    type(robust_scaler) :: robust
    type(simple_imputer) :: imputer
    type(quantile_transformer) :: quantile
    type(ordinal_encoder) :: ordinal
    type(one_hot_encoder) :: onehot
    type(pca) :: principal
    type(select_k_best) :: best
    type(select_percentile) :: percentile
    type(target_encoder) :: encoder
    type(variance_threshold) :: variance
    type(function_transformer) :: function,identity
    type(column_transformer) :: columns
    type(feature_union) :: union
    type(linear_regression) :: linear
    type(ridge) :: ridge_model
    type(logistic_regression) :: logistic
    type(gaussian_nb) :: nb
    type(kneighbors_regressor) :: knnr
    type(kneighbors_classifier) :: knnc
    type(poisson_regressor) :: poisson
    type(transformed_target_regressor) :: target
    class(*), allocatable :: tail
    imputer%strategy='median';minmax%feature_min=-1;minmax%feature_max=2
    principal%n_components=2;principal%whiten=mod(config,2)==0
    best%k=2;percentile%percentile=50
    encoder%cv=3;encoder%random_state=42;encoder%auto_smooth=.false.;encoder%smooth=1
    encoder%target_type='continuous';if(task==2) encoder%target_type='auto'
    function%func=>affine;function%inverse_func=>undo;function%feature_names_out='one-to-one'
    identity%feature_names_out='one-to-one'
    select case(kind)
    case(0);call add(pipe,'scale',standard)
    case(1);call add(pipe,'impute',imputer);call add(pipe,'scale',standard)
    case(2);call add(pipe,'pca',principal);call add(pipe,'range',minmax)
    case(3);call add(pipe,'select',best);call add(pipe,'scale',standard)
    case(4);call add(pipe,'select',percentile);call add(pipe,'scale',standard)
    case(5);call add(pipe,'encode',encoder);call add(pipe,'scale',standard)
    case(6)
      call add(child,'impute',imputer);call add(child,'scale',standard)
      allocate(columns%transformers(1));columns%transformers(1)%name='processed'
      columns%transformers(1)%columns=[3,1];allocate(columns%transformers(1)%model,source=child)
      columns%remainder%action='passthrough';call add(pipe,'columns',columns);call add(pipe,'scale',standard)
    case(7)
      principal%n_components=1;call add(child,'scale',standard);call add(child,'pca',principal)
      allocate(union%transformer_list(2));union%transformer_list(1)%name='latent'
      allocate(union%transformer_list(1)%model,source=child)
      union%transformer_list(2)%name='raw';union%transformer_list(2)%action='passthrough'
      union%transformer_list(2)%weight=.5_dp;call add(pipe,'union',union);call add(pipe,'scale',standard)
    case(8)
      call add(child,'scale',standard);call add(child,'range',minmax)
      call add(pipe,'nested',child);call add(pipe,'affine',function)
    case(9)
      nullify(function%inverse_func);function%func=>expand;function%feature_names_out='none'
      function%names_func=>expand_names
      call add(pipe,'expand',function);call add(pipe,'vary',variance);call add(pipe,'scale',standard)
    case(10)
      quantile%n_quantiles=7;quantile%subsample=0
      call add(pipe,'off',pass=.true.);call add(pipe,'robust',robust);call add(pipe,'quantile',quantile)
    case(11);call add(pipe,'affine',function);call add(pipe,'range',minmax)
    case(12)
      ordinal%handle_unknown='use_encoded_value';ordinal%unknown_value=-1
      onehot%handle_unknown_ignore=.true.
      call add(pipe,'ordinal',ordinal);call add(pipe,'onehot',onehot)
    case(13,14);call add(pipe,'off',pass=.true.)
    end select
    select case(task)
    case(0)
      call add(pipe,'final',pass=.true.);return
    case(1)
      select case(config)
      case(0,3);allocate(tail,source=linear)
      case(1);ridge_model%alpha=.3_dp;allocate(tail,source=ridge_model)
      case(2);knnr%n_neighbors=3;knnr%weights='distance';allocate(tail,source=knnr)
      end select
    case(2)
      select case(config)
      case(0,3);allocate(tail,source=nb)
      case(1);logistic%c=.7_dp;logistic%tol=1.e-10_dp;logistic%max_iter=300;allocate(tail,source=logistic)
      case(2);knnc%n_neighbors=3;knnc%weights='distance';allocate(tail,source=knnc)
      end select
    case(3)
      ridge_model%alpha=.3_dp;allocate(target%regressor,source=ridge_model);allocate(tail,source=target)
    case(4)
      poisson%alpha=.3_dp;poisson%tol=1.e-10_dp;poisson%max_iter=300;allocate(tail,source=poisson)
    end select
    if(config==3) then
      if(kind==14) then
        call add(final,'select',best)
      else
        call add(final,'identity',identity)
      end if
      call add(final,'estimator',tail);call add(pipe,'final',final)
    else
      call add(pipe,'final',tail)
    end if
  end subroutine
end module
