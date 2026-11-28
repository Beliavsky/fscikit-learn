program test_perceptron_search
  use sklearn_common,only:dp,require
  use sklearn_perceptron,only:perceptron
  use sklearn_pipeline,only:pipeline
  use sklearn_multiclass,only:one_vs_rest_classifier,one_vs_one_classifier
  use sklearn_search,only:grid_search_cv,param
  use sklearn_validation,only:cross_validate,cross_validation_options,cross_validation_result
  implicit none
  type(perceptron)::c
  type(pipeline)::pipe
  type(one_vs_rest_classifier)::ovr
  type(one_vs_one_classifier)::ovo
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
    c=perceptron()
    c%early_stopping=.true.;c%validation_fraction=.25_dp
    c%warm_start=.true.;c%random_state=42;c%max_iter=5
    if(trial==2)then
      call c%fit(wide,labels)
    end if
    result=cross_validate(c,x,labels,cv);outputs=reshape(result%test_score,[size(result%test_score)])
    pipe=pipeline();allocate(pipe%steps(1));pipe%steps(1)%name='pa';allocate(pipe%steps(1)%model,source=c)
    search=grid_search_cv();search%cv=3;allocate(search%estimator,source=pipe)
    allocate(search%param_grid(1));allocate(search%param_grid(1)%axes(2))
    search%param_grid(1)%axes(1)%name='pa__early_stopping';search%param_grid(1)%axes(1)%values=[param(.true.)]
    search%param_grid(1)%axes(2)%name='pa__validation_fraction';search%param_grid(1)%axes(2)%values=[param(.375_dp)]
    call search%fit(x,labels);outputs=[outputs,real(search%predict_class(x),dp)]
    select type(best=>search%best_estimator_)
    type is(pipeline)
      select type(model=>best%steps(1)%model)
      type is(perceptron)
        call require(model%early_stopping.and.model%validation_fraction==.375_dp,'nested Perceptron search controls')
        call require(count(model%validation_mask_)==18,'nested Perceptron search held-out mask')
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
    type is(perceptron)
      call require(model%early_stopping.and.model%validation_fraction==.375_dp,'classifier Perceptron search controls')
      call require(count(model%validation_mask_)==18,'classifier Perceptron search held-out mask')
    class default
      call require(.false.,'unexpected search classifier')
    end select
    c%early_stopping=.false.
    call ovr%fit(x,labels,c);outputs=[outputs,real(ovr%predict(x),dp)]
    call ovo%fit(x,labels,c);outputs=[outputs,real(ovo%predict(x),dp)]
    ovr=one_vs_rest_classifier()
    call ovr%partial_fit(x(:24,:),labels(:24),c,classes=[0,1,2])
    call ovr%partial_fit(x(25:,:),labels(25:));outputs=[outputs,real(ovr%predict(x),dp)]
    if(trial==1)then
      baseline=outputs
    else
      call require(all(outputs==baseline),'validation CV/search discard learned prototype state')
      call require(size(c%coef_,2)==3,'validation wrappers preserve caller prototypes')
    end if
  end do
  print *,'Perceptron validation CV, search controls and clone isolation: PerceptronSS'
end program
