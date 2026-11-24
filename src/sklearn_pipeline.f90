! Sequential Pipeline contracts: sklearn 1.6.1 pipeline.py (BSD-3-Clause).
! Native typed adapters are original MIT; see THIRD_PARTY_NOTICES.md.
module sklearn_pipeline
  use sklearn_common, only: dp,require,check_xy
  use sklearn_naive_bayes, only: gaussian_nb,multinomial_nb
  use sklearn_sgd, only: sgd_classifier
  use sklearn_neighbors, only: kneighbors_classifier
  use sklearn_frozen, only: frozen_estimator,frozen_model_features
  use sklearn_feature_extraction, only: feature_string
  use sklearn_transformer_protocol, only: dense_transformer
  use sklearn_feature_composition, only: feature_union,transformer_branch,column_transformer, &
    composition_supported=>supported,composition_fit=>fit_model,composition_fit_transform=>fit_transform_model, &
    composition_transform=>transform_model,composition_names=>model_names,input_names,integer_text,real_text
  use sklearn_function_transformer, only: function_transformer
  use sklearn_preprocessing, only: standard_scaler,minmax_scaler,maxabs_scaler,robust_scaler,power_transformer, &
    quantile_transformer,kbins_discretizer,ordinal_encoder,simple_imputer
  use sklearn_decomposition, only: pca,truncated_svd
  use sklearn_feature_selection, only: select_k_best,variance_threshold,generic_univariate_select,select_from_model
  use sklearn_target_encoding, only: target_encoder
  use sklearn_base_dispatch, only: base_supported,train_classifier,train_regressor,regression,labels,probabilities, &
    decisions,base_classes,base_proba,base_decision
  use sklearn_compose, only: transformed_target_regressor
  use sklearn_linear_model, only: linear_regression,ridge,tweedie_regressor
  use sklearn_weighted_linear, only: weighted_linear
  use sklearn_composite_scores, only: regression_score,score_weights
  use sklearn_ieee_math, only: ieee_mean
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: pipeline,pipeline_step,feature_union,transformer_branch
  type :: pipeline_step
    character(:), allocatable :: name
    character(16) :: action='model'
    class(*), allocatable :: model
    integer :: input_width_=0,output_width_=-1
    logical :: integer_target_=.false.
    type(feature_string), allocatable :: input_names_(:)
  end type
  type, extends(dense_transformer) :: pipeline
    type(pipeline_step), allocatable :: steps(:)
    integer :: n_features_in_=0,n_features_out_=-1,estimator_kind_=0,target_rank_=0
    integer, allocatable :: classes_(:)
    type(feature_string), allocatable :: feature_names_in_(:)
    logical, private :: fitted_=.false.
  contains
    procedure :: is_fitted=>pipeline_is_fitted
    procedure :: can_transform=>pipeline_can_transform
    procedure :: bridge_kind=>pipeline_kind
    procedure :: bridge_features=>pipeline_features
    procedure :: bridge_supports=>pipeline_supports
    procedure :: bridge_classes=>pipeline_classes
    procedure :: bridge_predict=>pipeline_predict
    procedure :: bridge_labels=>pipeline_labels
    procedure :: bridge_matrix=>pipeline_predict_matrix
    procedure :: bridge_proba=>pipeline_proba
    procedure :: bridge_log_proba=>pipeline_log_proba
    procedure :: bridge_decision=>pipeline_decisions
    procedure :: bridge_score_real=>score_real
    procedure :: bridge_score_integer=>score_integer
    procedure :: bridge_score_matrix=>score_matrix
    procedure :: fit_data=>fit_unsupervised
    procedure, private :: fit_regression,fit_classification,fit_matrix
    generic :: fit=>fit_data,fit_regression,fit_classification,fit_matrix
    procedure :: fit_map_data=>fit_transform_unsupervised
    procedure, private :: fit_transform_regression,fit_transform_classification
    generic :: fit_transform=>fit_map_data,fit_transform_regression,fit_transform_classification
    procedure :: map_data=>pipeline_transform
    procedure :: transform=>pipeline_transform
    procedure :: inverse_data=>pipeline_inverse
    procedure :: inverse_transform=>pipeline_inverse
    procedure :: can_inverse=>pipeline_can_inverse
    procedure :: feature_names=>pipeline_names
    procedure :: get_feature_names_out=>pipeline_names
    procedure :: predict=>pipeline_predict
    procedure :: predict_matrix=>pipeline_predict_matrix
    procedure :: predict_class=>pipeline_labels
    procedure :: predict_proba=>pipeline_proba
    procedure :: predict_log_proba=>pipeline_log_proba
    procedure :: decision_function=>pipeline_decisions
    procedure, private :: score_real,score_integer,score_matrix
    generic :: score=>score_real,score_integer,score_matrix
    procedure :: named_step_index
    procedure :: slice=>pipeline_slice
  end type
contains
  logical function same_name(a,b) result(equal)
    character(*), intent(in) :: a,b
    equal=len(a)==len(b)
    if(equal) equal=a==b
  end function
  recursive logical function transform_supported(model) result(ok)
    class(*), intent(in) :: model
    ok=.false.
    select type(model)
    type is(pipeline)
      ok=.false.
      if(allocated(model%steps)) then
        if(size(model%steps)>0) then
          associate(last=>model%steps(size(model%steps)))
            ok=last%action=='passthrough'
            if(.not.ok.and.allocated(last%model)) ok=transform_supported(last%model)
          end associate
        end if
      end if
    type is(select_from_model)
      ok=.true.
    type is(select_k_best)
      ok=.true.
    class is(generic_univariate_select)
      ok=.true.
    type is(target_encoder)
      ok=.true.
    class default
      ok=composition_supported(model)
    end select
  end function
  recursive integer function estimator_kind(model) result(kind)
    class(*), intent(in) :: model
    kind=0
    select type(model)
    type is(pipeline)
      if(allocated(model%steps)) then
        if(size(model%steps)>0) then
          associate(last=>model%steps(size(model%steps)))
            if(last%action=='model'.and.allocated(last%model)) kind=estimator_kind(last%model)
          end associate
        end if
      end if
    type is(frozen_estimator)
      kind=model%bridge_kind()
    type is(transformed_target_regressor)
      kind=2
    class default
      if(base_supported(model,.false.)) kind=2
      if(base_supported(model,.true.)) kind=1
    end select
  end function
  subroutine validate(self,transform_final)
    class(pipeline), intent(in) :: self
    logical, intent(in) :: transform_final
    character(15), parameter :: reserved(4)=[character(15)::'steps','memory','verbose','transform_input']
    integer :: i,j,n
    call require(allocated(self%steps),'Pipeline steps are unallocated')
    n=size(self%steps);call require(n>0,'Pipeline requires at least one step')
    do i=1,n
      associate(step=>self%steps(i))
        call require(allocated(step%name),'Pipeline step name is unallocated')
        call require(index(step%name,'__')==0,'Pipeline step name contains __')
        do j=1,size(reserved)
          call require(.not.same_name(step%name,trim(reserved(j))),'Pipeline step name conflicts with a control')
        end do
        do j=1,i-1
          call require(.not.same_name(step%name,self%steps(j)%name),'Pipeline step names must be unique')
        end do
        call require(any(step%action==['model           ','passthrough     ']),'Pipeline invalid step action')
        if(step%action=='passthrough') cycle
        call require(allocated(step%model),'Pipeline step model is unallocated')
        if(i<n.or.transform_final) then
          call require(transform_supported(step%model),'Pipeline step must support transform')
        else
          call require(transform_supported(step%model).or.estimator_kind(step%model)>0,'Pipeline unsupported final model')
        end if
      end associate
    end do
  end subroutine
  recursive logical function preserves_names(model) result(preserved)
    class(*), intent(in) :: model
    integer :: i
    preserved=.false.
    select type(model)
    type is(function_transformer)
      preserved=.not.model%validate.and..not.associated(model%func)
    type is(pipeline)
      preserved=.true.
      do i=1,size(model%steps)
        if(model%steps(i)%action=='passthrough') cycle
        if(.not.preserves_names(model%steps(i)%model)) preserved=.false.
      end do
    end select
  end function
  function integer_labels(y) result(values)
    real(dp), intent(in) :: y(:)
    integer, allocatable :: values(:)
    call require(all(ieee_is_finite(y)),'Pipeline classifier targets must be finite')
    call require(all(y>=-real(huge(1),dp)-1).and.all(y<=real(huge(1),dp)), &
      'Pipeline classifier target exceeds native integer range')
    call require(all(modulo(y,1._dp)==0),'Pipeline classifier requires integral targets')
    values=int(y)
  end function
  recursive function train_transform(model,x,names,do_transform,yr,yi) result(z)
    class(*), intent(inout) :: model
    real(dp), intent(in) :: x(:,:)
    type(feature_string), optional, intent(in) :: names(:)
    logical, intent(in) :: do_transform
    real(dp), optional, intent(in) :: yr(:)
    integer, optional, intent(in) :: yi(:)
    real(dp), allocatable :: z(:,:)
    select type(model)
    type is(pipeline)
      if(do_transform) then
        if(present(yi)) then
          z=model%fit_transform(x,yi,feature_names=names)
        else if(present(yr)) then
          z=model%fit_transform(x,yr,feature_names=names)
        else
          z=model%fit_transform(x,names)
        end if
      else
        if(present(yi)) then
          call model%fit(x,yi,feature_names=names)
        else if(present(yr)) then
          call model%fit(x,yr,feature_names=names)
        else
          call model%fit(x,names)
        end if
      end if
    type is(target_encoder)
      call require(present(yr).or.present(yi),'Pipeline supervised transformer requires a vector target')
      if(do_transform) then
        if(present(yi)) then
          z=model%fit_transform(x,yi)
        else
          z=model%fit_transform(x,yr)
        end if
      else
        if(present(yi)) then
          call model%fit(x,yi)
        else
          call model%fit(x,yr)
        end if
      end if
    type is(select_from_model)
      if(present(yi)) then
        call model%fit(x,yi,feature_names=names)
      else if(present(yr)) then
        call model%fit(x,yr,feature_names=names)
      else
        call model%fit(x,feature_names=names)
      end if
      if(do_transform) z=model%transform(x)
    type is(select_k_best)
      call require(present(yr).or.present(yi),'Pipeline supervised transformer requires a vector target')
      if(present(yi)) then
        call model%fit(x,yi)
      else
        call model%fit(x,yr)
      end if
      if(do_transform) z=model%transform(x)
    class is(generic_univariate_select)
      call require(present(yr).or.present(yi),'Pipeline supervised transformer requires a vector target')
      if(present(yi)) then
        call model%fit(x,yi)
      else
        call model%fit(x,yr)
      end if
      if(do_transform) z=model%transform(x)
    class default
      if(do_transform) then
        z=composition_fit_transform(model,x,names)
      else
        call composition_fit(model,x,names)
      end if
    end select
    if(.not.allocated(z)) allocate(z(0,0))
  end function
  recursive function map_step(model,x) result(z)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    select type(model)
    type is(select_from_model)
      z=model%transform(x)
    type is(select_k_best)
      z=model%transform(x)
    class is(generic_univariate_select)
      z=model%transform(x)
    type is(target_encoder)
      z=model%transform(x)
    class default
      z=composition_transform(model,x)
    end select
  end function
  recursive subroutine fit_regressor(model,x,y,weights,names)
    class(*), intent(inout) :: model
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), optional, intent(in) :: weights(:)
    type(feature_string), optional, intent(in) :: names(:)
    real(dp) :: w(size(y)),alpha
    select type(model)
    type is(pipeline)
      call model%fit(x,y,weights,feature_names=names)
    type is(frozen_estimator)
      call model%fit(x,y,weights)
    type is(transformed_target_regressor)
      call model%fit(x,y,weights)
    type is(linear_regression)
      call linear_fit(0._dp)
    type is(ridge)
      call linear_fit(model%alpha)
    class default
      call train_regressor(model,x,y,weights)
    end select
  contains
    subroutine linear_fit(penalty)
      real(dp), intent(in) :: penalty
      call check_xy(x,size(y));call require(all(ieee_is_finite(y)),'Pipeline regression targets must be finite')
      w=1
      if(present(weights)) then
        call require(size(weights)==size(y),'Pipeline fit weight length mismatch')
        call require(all(ieee_is_finite(weights)),'Pipeline fit weights must be finite')
        w=weights
      end if
      select type(model)
      class is(linear_regression)
        call weighted_linear(model,x,y,w,penalty)
      end select
    end subroutine
  end subroutine
  recursive subroutine fit_core(self,x,feature_names,transform_final,z,yr,yi,ym,weights)
    class(pipeline), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    type(feature_string), optional, intent(in) :: feature_names(:)
    logical, intent(in) :: transform_final
    real(dp), allocatable, intent(out) :: z(:,:)
    real(dp), optional, intent(in) :: yr(:),ym(:,:),weights(:)
    integer, optional, intent(in) :: yi(:)
    type(feature_string), allocatable :: names(:),checked(:)
    real(dp), allocatable :: work(:,:),discard(:,:)
    integer, allocatable :: target(:)
    integer :: i,n,kind
    call validate(self,transform_final);self%fitted_=.false.
    if(allocated(self%classes_)) deallocate(self%classes_)
    if(allocated(self%feature_names_in_)) deallocate(self%feature_names_in_)
    if(present(feature_names)) then
      checked=input_names(size(x,2),feature_names);names=checked;self%feature_names_in_=checked
    end if
    self%n_features_in_=size(x,2);self%n_features_out_=-1;self%target_rank_=0
    if(present(yr).or.present(yi)) self%target_rank_=1
    if(present(ym)) self%target_rank_=2
    work=x;n=size(self%steps)
    do i=1,n
      associate(step=>self%steps(i))
        step%input_width_=size(work,2);step%output_width_=-1;step%integer_target_=present(yi)
        if(allocated(step%input_names_)) deallocate(step%input_names_)
        if(allocated(names)) step%input_names_=names
        if(step%action=='passthrough') then
          step%output_width_=size(work,2);cycle
        end if
        if(i<n.or.transform_final) then
          discard=train_transform(step%model,work,names,.true.,yr,yi)
          if(.not.preserves_names(step%model)) then
            if(allocated(names)) deallocate(names)
          end if
          call move_alloc(discard,work);step%output_width_=size(work,2)
        else
          kind=estimator_kind(step%model)
          select type(model=>step%model)
          type is(frozen_estimator)
            call model%fit(work)
            if(kind==1) self%classes_=model%get_classes()
            if(i==1) then
              if(model%bridge_features()>0) self%n_features_in_=model%bridge_features()
            end if
            cycle
          end select
          select case(kind)
          case(1)
            call require(present(yi).or.present(yr),'Pipeline classifier requires a vector target')
            if(present(yi)) then
              target=yi
            else
              target=integer_labels(yr)
            end if
            select type(model=>step%model)
            type is(pipeline)
              if(present(yi)) then
                call model%fit(work,yi,weights,feature_names=names)
              else
                call model%fit(work,yr,weights,feature_names=names)
              end if
              self%classes_=model%classes_
            class default
              call train_classifier(model,work,target,weights);self%classes_=base_classes(model)
            end select
          case(2)
            if(present(ym)) then
              select type(model=>step%model)
              type is(transformed_target_regressor)
                call model%fit(work,ym,weights)
              type is(pipeline)
                call model%fit(work,ym,weights,feature_names=names)
              class default
                call require(.false.,'Pipeline matrix targets require a native matrix-target final estimator')
              end select
            else
              call require(present(yr).or.present(yi),'Pipeline regressor requires a target')
              if(present(yi)) then
                select type(model=>step%model)
                type is(pipeline)
                  call model%fit(work,yi,weights,feature_names=names)
                class default
                  call fit_regressor(model,work,real(yi,dp),weights,names)
                end select
              else
                call fit_regressor(step%model,work,yr,weights,names)
              end if
            end if
          case default
            call require(.not.present(weights),'Pipeline final transformer has no native fit weights')
            discard=train_transform(step%model,work,names,.false.,yr,yi)
          end select
        end if
      end associate
    end do
    self%estimator_kind_=0
    associate(last=>self%steps(n))
      if(last%action=='model') self%estimator_kind_=estimator_kind(last%model)
      if(transform_final.or.last%action=='passthrough') self%n_features_out_=size(work,2)
    end associate
    z=work;self%fitted_=.true.
  end subroutine
  recursive subroutine fit_unsupervised(self,x,feature_names)
    class(pipeline), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    type(feature_string), optional, intent(in) :: feature_names(:)
    real(dp), allocatable :: discarded(:,:)
    call fit_core(self,x,feature_names,.false.,discarded)
  end subroutine
  recursive subroutine fit_regression(self,x,y,sample_weight,feature_names)
    class(pipeline), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(in) :: y(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    type(feature_string), optional, intent(in) :: feature_names(:)
    real(dp), allocatable :: discarded(:,:)
    call fit_core(self,x,feature_names,.false.,discarded,yr=y,weights=sample_weight)
  end subroutine
  recursive subroutine fit_classification(self,x,y,sample_weight,feature_names)
    class(pipeline), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    type(feature_string), optional, intent(in) :: feature_names(:)
    real(dp), allocatable :: discarded(:,:)
    call fit_core(self,x,feature_names,.false.,discarded,yi=y,weights=sample_weight)
  end subroutine
  recursive subroutine fit_matrix(self,x,y,sample_weight,feature_names)
    class(pipeline), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(in) :: y(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    type(feature_string), optional, intent(in) :: feature_names(:)
    real(dp), allocatable :: discarded(:,:)
    call fit_core(self,x,feature_names,.false.,discarded,ym=y,weights=sample_weight)
  end subroutine
  recursive function fit_transform_unsupervised(self,x,feature_names) result(z)
    class(pipeline), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    type(feature_string), optional, intent(in) :: feature_names(:)
    real(dp), allocatable :: z(:,:)
    call fit_core(self,x,feature_names,.true.,z)
  end function
  recursive function fit_transform_regression(self,x,y,feature_names) result(z)
    class(pipeline), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(in) :: y(:)
    type(feature_string), optional, intent(in) :: feature_names(:)
    real(dp), allocatable :: z(:,:)
    call fit_core(self,x,feature_names,.true.,z,yr=y)
  end function
  recursive function fit_transform_classification(self,x,y,feature_names) result(z)
    class(pipeline), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    type(feature_string), optional, intent(in) :: feature_names(:)
    real(dp), allocatable :: z(:,:)
    call fit_core(self,x,feature_names,.true.,z,yi=y)
  end function
  recursive function through(self,x,include_final) result(z)
    class(pipeline), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    logical, intent(in) :: include_final
    real(dp), allocatable :: z(:,:),next(:,:)
    integer :: i,last
    call require(self%fitted_,'Pipeline is not fitted')
    call require(allocated(self%steps),'Pipeline steps are unallocated')
    call require(size(self%steps)>0,'Pipeline requires at least one step')
    last=size(self%steps);if(.not.include_final) last=last-1
    z=x
    do i=1,last
      if(self%steps(i)%action=='passthrough') cycle
      next=map_step(self%steps(i)%model,z);call move_alloc(next,z)
    end do
  end function
  recursive function pipeline_transform(self,x) result(z)
    class(pipeline), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call require(transform_supported(self),'Pipeline final estimator has no transform')
    z=through(self,x,.true.)
  end function
  recursive function regression_values(model,x) result(y)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:)
    select type(model)
    type is(frozen_estimator)
      y=model%predict(x)
    type is(pipeline)
      y=model%predict(x)
    type is(transformed_target_regressor)
      y=model%predict(x)
    class default
      y=regression(model,x)
    end select
  end function
  recursive function pipeline_predict(self,x) result(y)
    class(pipeline), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:),z(:,:)
    integer :: n
    z=through(self,x,.false.);n=size(self%steps)
    call require(self%steps(n)%action=='model','Pipeline final passthrough has no predict')
    call require(estimator_kind(self%steps(n)%model)==2,'Pipeline vector predict requires a regressor')
    y=regression_values(self%steps(n)%model,z)
  end function
  recursive function pipeline_predict_matrix(self,x) result(y)
    class(pipeline), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:,:),z(:,:),vector(:)
    integer :: n
    z=through(self,x,.false.);n=size(self%steps)
    call require(self%steps(n)%action=='model','Pipeline final passthrough has no predict')
    select type(model=>self%steps(n)%model)
    type is(frozen_estimator)
      y=model%predict_matrix(z)
    type is(pipeline)
      y=model%predict_matrix(z)
    type is(transformed_target_regressor)
      y=model%predict_matrix(z)
    class default
      call require(estimator_kind(model)==2,'Pipeline matrix predict requires a regressor')
      vector=regression(model,z);y=reshape(vector,[size(vector),1])
    end select
  end function
  recursive function classifier_values(model,x) result(y)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: y(:)
    select type(model)
    type is(frozen_estimator)
      y=model%predict_class(x)
    type is(pipeline)
      y=model%predict_class(x)
    class default
      y=labels(model,x)
    end select
  end function
  recursive function pipeline_labels(self,x) result(y)
    class(pipeline), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: y(:)
    real(dp), allocatable :: z(:,:)
    integer :: n
    z=through(self,x,.false.);n=size(self%steps)
    call require(self%steps(n)%action=='model','Pipeline final passthrough has no predict')
    call require(estimator_kind(self%steps(n)%model)==1,'Pipeline predict_class requires a classifier')
    y=classifier_values(self%steps(n)%model,z)
  end function
  recursive function pipeline_proba(self,x) result(y)
    class(pipeline), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:,:),z(:,:)
    integer :: n
    z=through(self,x,.false.);n=size(self%steps)
    call require(self%steps(n)%action=='model','Pipeline final passthrough has no predict_proba')
    select type(model=>self%steps(n)%model)
    type is(frozen_estimator)
      y=model%predict_proba(z)
    type is(pipeline)
      y=model%predict_proba(z)
    class default
      call require(base_proba(model),'Pipeline final estimator has no predict_proba')
      y=probabilities(model,z)
    end select
  end function
  recursive function pipeline_decisions(self,x) result(y)
    class(pipeline), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:,:),z(:,:)
    integer :: n
    z=through(self,x,.false.);n=size(self%steps)
    call require(self%steps(n)%action=='model','Pipeline final passthrough has no decision_function')
    select type(model=>self%steps(n)%model)
    type is(frozen_estimator)
      y=model%decision_function(z)
    type is(pipeline)
      y=model%decision_function(z)
    class default
      call require(base_decision(model),'Pipeline final estimator has no decision_function')
      y=decisions(model,z)
    end select
  end function
  recursive real(dp) function regression_model_score(model,x,y,w) result(value)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), optional, intent(in) :: w(:)
    select type(model)
    type is(frozen_estimator)
      value=model%score(x,y,w)
    type is(pipeline)
      value=model%score(x,y,w)
    type is(transformed_target_regressor)
      value=model%score(x,y,w)
    class is(tweedie_regressor)
      call require(.not.present(w),'Pipeline native GLM score does not support weights')
      value=model%score(x,y)
    class default
      value=regression_score(y,regression_values(model,x),w,'Pipeline')
    end select
  end function
  recursive real(dp) function score_real(self,x,y,sample_weight) result(value)
    class(pipeline), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    real(dp), allocatable :: z(:,:)
    integer, allocatable :: target(:),predicted(:)
    integer :: n
    z=through(self,x,.false.);n=size(self%steps)
    call require(self%steps(n)%action=='model','Pipeline final passthrough has no score')
    select case(estimator_kind(self%steps(n)%model))
    case(1)
      target=integer_labels(y);predicted=classifier_values(self%steps(n)%model,z)
      call require(size(predicted)==size(target),'Pipeline score target length mismatch')
      call score_weights(sample_weight,size(target),'Pipeline')
      value=ieee_mean(merge(1._dp,0._dp,predicted==target),sample_weight)
    case(2)
      value=regression_model_score(self%steps(n)%model,z,y,sample_weight)
    case default
      call require(.false.,'Pipeline final transformer has no score')
    end select
  end function
  recursive real(dp) function score_integer(self,x,y,sample_weight) result(value)
    class(pipeline), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    value=score_real(self,x,real(y,dp),sample_weight)
  end function
  recursive real(dp) function score_matrix(self,x,y,sample_weight) result(value)
    class(pipeline), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    real(dp), allocatable :: z(:,:)
    integer :: n
    z=through(self,x,.false.);n=size(self%steps)
    call require(self%steps(n)%action=='model','Pipeline final passthrough has no score')
    select type(model=>self%steps(n)%model)
    type is(frozen_estimator)
      value=model%score(z,y,sample_weight)
    type is(pipeline)
      value=model%score(z,y,sample_weight)
    type is(transformed_target_regressor)
      value=model%score(z,y,sample_weight)
    class default
      call require(.false.,'Pipeline matrix score requires a native matrix-target final estimator')
    end select
  end function
  recursive logical function inverse_supported(model) result(available)
    class(*), intent(in) :: model
    available=.true.
    select type(model)
    type is(function_transformer)
    type is(simple_imputer)
    type is(standard_scaler)
    type is(minmax_scaler)
    type is(maxabs_scaler)
    type is(robust_scaler)
    type is(power_transformer)
    type is(quantile_transformer)
    type is(kbins_discretizer)
    type is(ordinal_encoder)
    type is(pca)
    type is(truncated_svd)
    type is(variance_threshold)
    type is(select_from_model)
    type is(select_k_best)
    class is(generic_univariate_select)
    class is(dense_transformer)
      available=model%can_inverse()
    class default
      available=.false.
    end select
  end function
  recursive logical function pipeline_can_inverse(self) result(available)
    class(pipeline), intent(in) :: self
    integer :: i
    available=.false.
    if(.not.allocated(self%steps)) return
    if(size(self%steps)==0) return
    do i=1,size(self%steps)
      if(self%steps(i)%action=='passthrough') cycle
      if(.not.allocated(self%steps(i)%model)) return
      if(.not.inverse_supported(self%steps(i)%model)) return
    end do
    available=.true.
  end function
  recursive function inverse_model(model,x) result(z)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    select type(model)
    type is(function_transformer)
      z=model%inverse_transform(x)
    type is(simple_imputer)
      z=model%inverse_transform(x)
    type is(standard_scaler)
      z=model%inverse_transform(x)
    type is(minmax_scaler)
      z=model%inverse_transform(x)
    type is(maxabs_scaler)
      z=model%inverse_transform(x)
    type is(robust_scaler)
      z=model%inverse_transform(x)
    type is(power_transformer)
      z=model%inverse_transform(x)
    type is(quantile_transformer)
      z=model%inverse_transform(x)
    type is(kbins_discretizer)
      z=model%inverse_transform(x)
    type is(ordinal_encoder)
      z=model%inverse_transform(x)
    type is(pca)
      z=model%inverse_transform(x)
    type is(truncated_svd)
      z=model%inverse_transform(x)
    type is(variance_threshold)
      z=model%inverse_transform(x)
    type is(select_from_model)
      z=model%inverse_transform(x)
    type is(select_k_best)
      z=model%inverse_transform(x)
    class is(generic_univariate_select)
      z=model%inverse_transform(x)
    class is(dense_transformer)
      z=model%inverse_data(x)
    class default
      call require(.false.,'Pipeline step has no inverse_transform')
    end select
  end function
  recursive function pipeline_inverse(self,x) result(z)
    class(pipeline), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:),next(:,:)
    integer :: i
    call require(self%fitted_,'Pipeline is not fitted')
    call require(self%can_inverse(),'Pipeline step has no inverse_transform')
    z=x
    do i=size(self%steps),1,-1
      if(self%steps(i)%action=='passthrough') cycle
      next=inverse_model(self%steps(i)%model,z);call move_alloc(next,z)
    end do
  end function
  recursive function step_names(step,input) result(names)
    type(pipeline_step), intent(in) :: step
    type(feature_string), intent(in) :: input(:)
    type(feature_string), allocatable :: names(:)
    integer :: i,j,k
    character(:), allocatable :: suffix
    if(step%action=='passthrough') then
      names=input;return
    end if
    select type(model=>step%model)
    type is(select_from_model)
      names=model%get_feature_names_out(input)
    type is(select_k_best)
      names=pack(input,model%get_support())
    class is(generic_univariate_select)
      names=pack(input,model%get_support())
    type is(target_encoder)
      if(model%target_type_/='multiclass') then
        names=input
      else
        allocate(names(size(input)*size(model%classes_)));k=0
        do i=1,size(input)
          do j=1,size(model%classes_)
            if(step%integer_target_) then
              suffix=integer_text(int(model%classes_(j)))
            else
              suffix=real_text(model%classes_(j))
            end if
            k=k+1;names(k)%text=input(i)%text//'_'//suffix
          end do
        end do
      end if
    class default
      names=composition_names(model,input)
    end select
  end function
  recursive function pipeline_names(self,input_features) result(names)
    class(pipeline), intent(in) :: self
    type(feature_string), optional, intent(in) :: input_features(:)
    type(feature_string), allocatable :: names(:),next(:)
    integer :: i
    call require(self%fitted_,'Pipeline is not fitted')
    call require(transform_supported(self),'Pipeline final estimator has no output feature names')
    names=input_names(self%n_features_in_,input_features,self%feature_names_in_)
    do i=1,size(self%steps)
      next=step_names(self%steps(i),names);call move_alloc(next,names)
    end do
  end function
  integer function named_step_index(self,name) result(found)
    class(pipeline), intent(in) :: self
    character(*), intent(in) :: name
    integer :: i
    found=0
    if(.not.allocated(self%steps)) return
    do i=1,size(self%steps)
      if(.not.allocated(self%steps(i)%name)) cycle
      if(same_name(self%steps(i)%name,name)) found=i
    end do
  end function
  function pipeline_slice(self,first,last) result(selected)
    class(pipeline), intent(in) :: self
    integer, intent(in) :: first,last
    type(pipeline) :: selected
    call require(allocated(self%steps),'Pipeline steps are unallocated')
    call require(first>=1.and.last>=first.and.last<=size(self%steps),'Pipeline slice indices out of bounds')
    selected%steps=self%steps(first:last);selected%fitted_=self%fitted_
    selected%n_features_in_=self%steps(first)%input_width_
    if(allocated(self%steps(first)%input_names_)) selected%feature_names_in_=self%steps(first)%input_names_
    selected%n_features_out_=self%steps(last)%output_width_
    selected%target_rank_=self%target_rank_
    if(selected%steps(size(selected%steps))%action=='model') then
      selected%estimator_kind_=estimator_kind(selected%steps(size(selected%steps))%model)
    end if
    if(selected%estimator_kind_==1.and.allocated(self%classes_)) selected%classes_=self%classes_
  end function
  logical function pipeline_is_fitted(self) result(ok)
    class(pipeline), intent(in) :: self
    ok=self%fitted_
  end function
  recursive logical function pipeline_can_transform(self) result(ok)
    class(pipeline), intent(in) :: self
    ok=transform_supported(self)
  end function
  recursive integer function pipeline_kind(self) result(kind)
    class(pipeline), intent(in) :: self
    kind=estimator_kind(self)
  end function
  integer function pipeline_features(self) result(n)
    class(pipeline), intent(in) :: self
    n=self%n_features_in_
  end function
  recursive function pipeline_classes(self) result(values)
    class(pipeline), intent(in) :: self
    integer, allocatable :: values(:)
    call require(self%fitted_,'Pipeline is not fitted')
    call require(allocated(self%classes_),'Pipeline is not a classifier')
    select type(model=>self%steps(size(self%steps))%model)
    type is(frozen_estimator);values=model%get_classes()
    type is(pipeline);values=model%bridge_classes()
    class default;values=base_classes(model)
    end select
  end function
  recursive logical function pipeline_supports(self,name) result(ok)
    class(pipeline), intent(in) :: self
    character(*), intent(in) :: name
    integer :: n,kind
    ok=.false.;if(.not.allocated(self%steps)) return
    n=size(self%steps);if(n==0) return
    select case(name)
    case('fit');ok=.true.
    case('get_feature_names_out');ok=.true.
    case('transform','fit_transform');ok=transform_supported(self)
    case('inverse_transform');ok=self%can_inverse()
    case('predict','score');ok=estimator_kind(self)>0
    case('predict_proba','predict_log_proba','decision_function')
      if(self%steps(n)%action/='model') return
      if(.not.allocated(self%steps(n)%model)) return
      select type(model=>self%steps(n)%model)
      type is(pipeline);ok=model%bridge_supports(name)
      type is(frozen_estimator);ok=model%has_method(name)
      type is(kneighbors_classifier);ok=name=='predict_proba'
      class default
        if(name=='decision_function') then
          ok=base_decision(model)
        else
          ok=base_proba(model)
        end if
      end select
    end select
  end function
  recursive function pipeline_log_proba(self,x) result(y)
    class(pipeline), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:,:),z(:,:)
    integer :: n
    call require(self%bridge_supports('predict_log_proba'),'Pipeline final estimator has no predict_log_proba')
    z=through(self,x,.false.);n=size(self%steps)
    call require(self%steps(n)%action=='model','Pipeline final passthrough has no predict_log_proba')
    select type(model=>self%steps(n)%model)
    type is(pipeline);y=model%bridge_log_proba(z)
    type is(frozen_estimator);y=model%predict_log_proba(z)
    type is(gaussian_nb);y=model%predict_log_proba(z)
    class is(multinomial_nb);y=model%predict_log_proba(z)
    type is(sgd_classifier);y=model%predict_log_proba(z)
    class default
      y=probabilities(model,z)
      block
        use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_negative_inf
        integer :: i,j
        do j=1,size(y,2)
          do i=1,size(y,1)
            if(y(i,j)==0) then
              y(i,j)=ieee_value(0._dp,ieee_negative_inf)
            else
              y(i,j)=log(y(i,j))
            end if
          end do
        end do
      end block
    end select
  end function
end module
