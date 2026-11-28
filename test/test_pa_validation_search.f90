program test_pa_validation_search
  use sklearn_common,only:dp,require
  use sklearn_linear_model,only:passive_aggressive_regressor,passive_aggressive_classifier
  use sklearn_pipeline,only:pipeline
  use sklearn_search,only:grid_search_cv,param
  use sklearn_validation,only:cross_validate,cross_validation_options,cross_validation_result
  implicit none
  type(passive_aggressive_regressor)::r
  type(passive_aggressive_classifier)::c
  type(pipeline)::pipe
  type(grid_search_cv)::search
  type(cross_validation_options)::cv
  type(cross_validation_result)::result
  real(dp)::x(48,2),wide(48,3),y(48)
  real(dp),allocatable::outputs(:),baseline(:)
  integer::labels(48),i,trial
  do i=1,48
    x(i,:)=[sin(real(i,dp)),cos(.3_dp*i)];y(i)=x(i,1)-x(i,2);labels(i)=mod(i,3)
  end do
  wide(:,:2)=x;wide(:,3)=.1_dp
  cv%cv=3
  do trial=1,2
    r=passive_aggressive_regressor();c=passive_aggressive_classifier()
    r%early_stopping=.true.;c%early_stopping=.true.;r%validation_fraction=.25_dp;c%validation_fraction=.25_dp
    r%warm_start=.true.;c%warm_start=.true.;r%random_state=42;c%random_state=42;r%max_iter=5;c%max_iter=5
    if(trial==2)then
      call r%fit(wide,y);call c%fit(wide,labels)
    end if
    result=cross_validate(r,x,y,cv);outputs=reshape(result%test_score,[size(result%test_score)])
    result=cross_validate(c,x,labels,cv);outputs=[outputs,reshape(result%test_score,[size(result%test_score)])]
    pipe=pipeline();allocate(pipe%steps(1));pipe%steps(1)%name='pa';allocate(pipe%steps(1)%model,source=r)
    search=grid_search_cv();search%cv=3;allocate(search%estimator,source=pipe)
    allocate(search%param_grid(1));allocate(search%param_grid(1)%axes(2))
    search%param_grid(1)%axes(1)%name='pa__early_stopping';search%param_grid(1)%axes(1)%values=[param(.true.)]
    search%param_grid(1)%axes(2)%name='pa__validation_fraction';search%param_grid(1)%axes(2)%values=[param(.375_dp)]
    call search%fit(x,y);outputs=[outputs,search%predict(x)]
    select type(best=>search%best_estimator_)
    type is(pipeline)
      select type(model=>best%steps(1)%model)
      type is(passive_aggressive_regressor)
        call require(model%early_stopping.and.model%validation_fraction==.375_dp,'nested PA search controls')
        call require(count(model%validation_mask_)==18,'nested PA search held-out mask')
      class default
        call require(.false.,'unexpected nested search model')
      end select
    class default
      call require(.false.,'unexpected search pipeline')
    end select
    search=grid_search_cv();search%cv=3;allocate(search%estimator,source=c)
    allocate(search%param_grid(1));allocate(search%param_grid(1)%axes(2))
    search%param_grid(1)%axes(1)%name='early_stopping';search%param_grid(1)%axes(1)%values=[param(.true.)]
    search%param_grid(1)%axes(2)%name='validation_fraction';search%param_grid(1)%axes(2)%values=[param(.375_dp)]
    call search%fit(x,labels);outputs=[outputs,real(search%predict_class(x),dp)]
    select type(model=>search%best_estimator_)
    type is(passive_aggressive_classifier)
      call require(model%early_stopping.and.model%validation_fraction==.375_dp,'classifier PA search controls')
      call require(count(model%validation_mask_)==18,'classifier PA search held-out mask')
    class default
      call require(.false.,'unexpected search classifier')
    end select
    if(trial==1)then
      baseline=outputs
    else
      call require(all(outputs==baseline),'validation CV/search discard learned prototype state')
      call require(size(r%coef_)==3.and.size(c%coef_,2)==3,'validation wrappers preserve caller prototypes')
    end if
  end do
  print *,'PA validation CV, search controls and clone isolation: PASS'
end program
