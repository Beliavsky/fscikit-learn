program test_pa_warm_clone
  use sklearn, only: dp,passive_aggressive_regressor,passive_aggressive_classifier,multi_output_regressor,multi_output_classifier, &
    regressor_chain,classifier_chain,one_vs_rest_classifier,one_vs_one_classifier,label_set, &
    voting_regressor,voting_classifier,voting_estimator,bagging_regressor,bagging_classifier, &
    stacking_regressor,stacking_classifier,select_from_model,fixed_threshold_classifier,pipeline
  use sklearn_common, only: require
  use sklearn_sgd_clone, only: clone_pa_parameters
  use sklearn_validation, only: cross_validate,cross_validation_options,cross_validation_result, &
    learning_curve,learning_curve_options,learning_curve_result
  use sklearn_search, only: grid_search_cv,param
  use sklearn_compose,only:transformed_target_regressor
  implicit none
  type(passive_aggressive_regressor) :: r,reg
  type(passive_aggressive_classifier) :: c,clf
  type(multi_output_regressor) :: mor,mor_stream
  type(multi_output_classifier) :: moc,moc_stream
  type(regressor_chain) :: rc
  type(classifier_chain) :: cc
  type(one_vs_rest_classifier) :: ovr,ovr_stream
  type(one_vs_one_classifier) :: ovo
  type(voting_estimator) :: bases(1)
  type(voting_regressor) :: vr
  type(voting_classifier) :: vc
  type(bagging_regressor) :: br
  type(bagging_classifier) :: bc
  type(stacking_regressor) :: sr
  type(stacking_classifier) :: sc
  type(select_from_model) :: selector
  type(fixed_threshold_classifier) :: threshold
  type(transformed_target_regressor)::target
  type(pipeline) :: inner,outer
  type(cross_validation_options) :: cv
  type(cross_validation_result) :: result
  type(learning_curve_options) :: curve_options
  type(learning_curve_result) :: curve
  type(grid_search_cv) :: search
  type(label_set) :: classes(2)
  real(dp) :: x(24,3),augmented(24,4),y(24),targets(24,2)
  real(dp), allocatable :: outputs(:),baseline(:),saved_r(:),saved_c(:,:),prediction(:,:)
  integer :: i,trial,labels(24),binary(24),multiple(24,2)
  do i=1,24
    x(i,:)=[sin(real(i,dp)),cos(.3_dp*i),sin(.7_dp*i)]*.2_dp
    y(i)=2*x(i,1)-x(i,2)+.3_dp
    labels(i)=mod(i,3);binary(i)=mod(i,2)
  end do
  augmented(:,:3)=x;augmented(:,4)=.1_dp
  targets(:,1)=y;targets(:,2)=.5_dp*y+x(:,3)
  multiple(:,1)=binary;multiple(:,2)=labels
  classes(1)%values=[0,1];classes(2)%values=[0,1,2]
  r%c=.25_dp;r%epsilon=.47_dp;r%max_iter=3;r%tol=-1
  r%average=1;r%fit_intercept=.false.;r%shuffle=.true.;r%random_state=42;r%warm_start=.true.
  c%c=.25_dp;c%loss='squared_hinge';c%max_iter=3;c%tol=-1
  c%average=1;c%fit_intercept=.false.;c%shuffle=.true.;c%random_state=42;c%warm_start=.true.
  do trial=1,2
    reg=r;clf=c
    if(trial==2) then
      call reg%fit(augmented,y);call clf%fit(augmented,labels)
      saved_r=reg%coef_;saved_c=clf%coef_
    end if
    outputs=[real(dp)::]
    call mor%fit(x,targets,reg);call collect(reshape(mor%predict(x),[48]))
    call controls(mor%estimators_(1)%estimator_)
    target=transformed_target_regressor();allocate(target%regressor,source=reg)
    call target%fit(x,y);call collect(target%predict(x));call controls(target%regressor_)
    call rc%fit(x,targets,reg);call collect(reshape(rc%predict(x),[48]))
    call ovr%fit(x,labels,clf);prediction=ovr%decision_function(x);call collect(reshape(prediction,[size(prediction)]))
    call controls(ovr%estimators_(1)%estimator_)
    call ovo%fit(x,labels,clf);prediction=ovo%decision_function(x);call collect(reshape(prediction,[size(prediction)]))
    bases(1)%name='sgd'
    if(allocated(bases(1)%estimator)) deallocate(bases(1)%estimator)
    allocate(bases(1)%estimator,source=reg)
    call vr%fit(x,y,bases);call collect(vr%predict(x));call controls(vr%estimators_(1)%estimator)
    sr%cv=3;call sr%fit(x,y,bases,final_estimator=reg);call collect(sr%predict(x))
    call controls(sr%final_estimator_)
    deallocate(bases(1)%estimator);allocate(bases(1)%estimator,source=clf)
    call vc%fit(x,labels,bases);call collect(real(vc%predict(x),dp));call controls(vc%estimators_(1)%estimator)
    sc%cv=3;call sc%fit(x,labels,bases,final_estimator=clf);call collect(real(sc%predict(x),dp))
    call controls(sc%final_estimator_)
    br%n_estimators=3;br%random_state=42;call br%fit(x,y,reg)
    call collect(br%predict(x));call controls(br%estimators_(1)%estimator)
    bc%n_estimators=3;bc%random_state=42;call bc%fit(x,labels,clf)
    prediction=bc%predict_proba(x);call collect(reshape(prediction,[size(prediction)]))
    call controls(bc%estimators_(1)%estimator)
    call selector%fit(x,y,reg);call collect(real(merge(1,0,selector%get_support()),dp));call controls(selector%estimator_)
    threshold%response_method='decision_function';threshold%threshold=0
    call threshold%fit(x,binary,clf);call collect(real(threshold%predict(x),dp));call controls(threshold%estimator_)
    cv%cv=3;result=cross_validate(reg,x,y,cv);call collect(reshape(result%test_score,[size(result%test_score)]))
    result=cross_validate(clf,x,labels,cv);call collect(reshape(result%test_score,[size(result%test_score)]))
    inner=pipeline();outer=pipeline();allocate(inner%steps(1),outer%steps(1))
    inner%steps(1)%name='sgd';allocate(inner%steps(1)%model,source=reg)
    outer%steps(1)%name='nested';allocate(outer%steps(1)%model,source=inner)
    result=cross_validate(outer,x,y,cv);call collect(reshape(result%test_score,[size(result%test_score)]))
    search=grid_search_cv();search%cv=3;allocate(search%estimator,source=outer)
    allocate(search%param_grid(1));allocate(search%param_grid(1)%axes(6))
    search%param_grid(1)%axes(1)%name='nested__sgd__epsilon'
    search%param_grid(1)%axes(1)%values=[param(.47_dp)]
    search%param_grid(1)%axes(2)%name='nested__sgd__C'
    search%param_grid(1)%axes(2)%values=[param(.25_dp)]
    search%param_grid(1)%axes(3)%name='nested__sgd__warm_start'
    search%param_grid(1)%axes(3)%values=[param(.true.)]
    search%param_grid(1)%axes(4)%name='nested__sgd__average'
    search%param_grid(1)%axes(4)%values=[param(1)]
    search%param_grid(1)%axes(5)%name='nested__sgd__random_state'
    search%param_grid(1)%axes(5)%values=[param(42)]
    search%param_grid(1)%axes(6)%name='nested__sgd__n_iter_no_change'
    search%param_grid(1)%axes(6)%values=[param(2)]
    call search%fit(x,y);call collect(search%predict(x))
    search=grid_search_cv();search%cv=3;allocate(search%estimator,source=clf)
    allocate(search%param_grid(1));allocate(search%param_grid(1)%axes(2))
    search%param_grid(1)%axes(1)%name='C';search%param_grid(1)%axes(1)%values=[param(.25_dp)]
    search%param_grid(1)%axes(2)%name='warm_start';search%param_grid(1)%axes(2)%values=[param(.true.)]
    call search%fit(x,labels);call collect(real(search%predict_class(x),dp))
    if(trial==1) then
      baseline=outputs
    else
      call require(size(outputs)==size(baseline),'PA clone property output inventory')
      call require(all(outputs==baseline),'fitted warm prototypes must not alter cold clone predictions')
      call require(all(reg%coef_==saved_r).and.all(clf%coef_==saved_c),'wrappers preserve fitted caller prototypes')
    end if
  end do
  reg=r;call reg%fit(x,y)
  inner=pipeline();allocate(inner%steps(1));inner%steps(1)%name='pa';allocate(inner%steps(1)%model,source=reg)
  call inner%fit(x,y);call reg%fit(x,y)
  call require(all(inner%predict(x)==reg%predict(x)),'direct pipeline fits retain warm state')
  reg=clone_pa_parameters(reg);clf=clone_pa_parameters(clf)
  call require(.not.allocated(reg%coef_).and..not.allocated(clf%coef_),'parameter copies omit fitted coefficients')
  call controls(reg);call controls(clf)
  print *, 'PA parameter-only clones and wrapper isolation: PASS',size(outputs)
contains
  subroutine collect(values)
    real(dp), intent(in) :: values(:)
    outputs=[outputs,values]
  end subroutine
  subroutine controls(model)
    class(*), intent(in) :: model
    select type(model)
    type is(passive_aggressive_regressor)
      call require(model%c==.25_dp.and.model%warm_start, &
        'regressor clones preserve cost and warm_start')
      call require(model%average==1.and..not.model%fit_intercept.and.model%shuffle,'regressor clone flags')
    type is(passive_aggressive_classifier)
      call require(model%c==.25_dp.and.model%warm_start, &
        'classifier clones preserve cost and warm_start')
      call require(model%average==1.and..not.model%fit_intercept.and.model%shuffle,'classifier clone flags')
    class default
      call require(.false.,'unexpected PA clone type')
    end select
  end subroutine
end program
