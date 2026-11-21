program reference_multiclass_driver
  use sklearn
  use sklearn_common, only: require
  implicit none
  type(one_vs_rest_classifier) :: ovr
  type(one_vs_one_classifier) :: ovo
  type(output_code_classifier) :: code
  type(logistic_regression) :: lr
  type(gaussian_nb) :: nb
  type(sgd_classifier) :: sgd
  type(kneighbors_classifier) :: knn
  type(decision_tree_classifier) :: tree
  type(multinomial_nb) :: mnb
  type(bernoulli_nb) :: bnb
  type(complement_nb) :: cnb
  type(perceptron) :: percep
  type(passive_aggressive_classifier) :: pa
  type(ridge_classifier) :: ridge_cls
  type(linear_svc) :: linear_margin
  type(svc) :: kernel_margin
  type(linear_regression) :: ols
  type(ridge) :: ridge_reg
  type(sgd_regressor) :: sgd_reg
  type(kneighbors_regressor) :: knn_reg
  type(decision_tree_regressor) :: tree_reg
  type(ridge_classifier_cv) :: ridge_cv_base
  class(*), allocatable :: base
  real(dp), allocatable :: x(:,:),query(:,:),w(:),sw(:),before(:),after(:),values(:),proba(:,:)
  integer, allocatable :: y(:,:),truth(:,:),classes(:),cuts(:),pre_y(:),prediction(:),indicator(:,:)
  integer :: mode,choice,kind,n,p,m,t,batches,weighted,seed,average,loss,initial_fit,refit,input,output
  integer :: j,k,c,first,last,heads,nfeatures,nclasses,multi,ready,has_proba,has_margin,status,value,features
  real(dp) :: code_size
  character(1024) :: input_path,output_path
  call get_command_argument(1,input_path); call get_command_argument(2,output_path)
  open(newunit=input,file=trim(input_path),status='old')
  read(input,*) mode,choice,kind,n,p,m,t,batches,weighted,seed,average,loss,initial_fit,refit,code_size
  read(input,*) k; allocate(classes(k)); read(input,*) classes
  allocate(x(n,p),query(m,p),y(n,t),truth(m,t),w(n),sw(m),cuts(batches))
  read(input,*) w; read(input,*) sw
  if(batches>0) read(input,*) cuts
  do j=1,n
    read(input,*) x(j,:)
  end do
  do j=1,n
    read(input,*) y(j,:)
  end do
  do j=1,m
    read(input,*) query(j,:)
  end do
  do j=1,m
    read(input,*) truth(j,:)
  end do
  close(input)
  lr%c=.7_dp; lr%tol=1.e-10_dp; lr%max_iter=500
  sgd%loss='log_loss'; sgd%learning_rate='constant'; sgd%eta0=.02_dp; sgd%alpha=.001_dp
  if(loss==1) sgd%loss='hinge'
  if(loss==2) sgd%loss='modified_huber'
  sgd%average=average; sgd%max_iter=4; sgd%tol=-1; sgd%shuffle=.false.
  if(batches>0.and.average>0) sgd%fit_intercept=.false.
  knn%n_neighbors=3; knn%weights='distance'
  tree%max_depth=2; tree%min_samples_leaf=4
  mnb%alpha=.4_dp; bnb%alpha=.4_dp; cnb%alpha=.4_dp; cnb%norm=.true.
  percep%eta0=.1_dp; percep%max_iter=4
  pa%c=.7_dp; pa%max_iter=4; pa%tol=-1; pa%shuffle=.false.
  ridge_cls%alpha=.7_dp
  linear_margin%c=.7_dp; linear_margin%tol=1.e-10_dp; linear_margin%max_iter=10000
  kernel_margin%c=.7_dp; kernel_margin%tol=1.e-8_dp; kernel_margin%gamma=.4_dp; kernel_margin%max_iter=100000
  ridge_reg%alpha=.7_dp
  sgd_reg%learning_rate='constant'; sgd_reg%eta0=.02_dp; sgd_reg%alpha=.001_dp
  sgd_reg%max_iter=4; sgd_reg%tol=-1; sgd_reg%shuffle=.false.
  knn_reg%n_neighbors=3; knn_reg%weights='distance'
  tree_reg%max_depth=2; tree_reg%min_samples_leaf=4
  ridge_cv_base%alphas=[.1_dp,.7_dp,2._dp]
  select case(choice)
  case(0)
    allocate(base,source=lr)
  case(1)
    allocate(base,source=nb)
  case(2)
    allocate(base,source=sgd)
  case(3)
    allocate(base,source=knn)
  case(4)
    allocate(base,source=tree)
  case(5)
    allocate(base,source=mnb)
  case(6)
    allocate(base,source=bnb)
  case(7)
    allocate(base,source=cnb)
  case(8)
    allocate(base,source=percep)
  case(9)
    allocate(base,source=pa)
  case(10)
    allocate(base,source=ridge_cls)
  case(11)
    allocate(base,source=linear_margin)
  case(12)
    allocate(base,source=kernel_margin)
  case(13)
    allocate(base,source=ols)
  case(14)
    allocate(base,source=ridge_reg)
  case(15)
    allocate(base,source=sgd_reg)
  case(16)
    allocate(base,source=knn_reg)
  case(17)
    allocate(base,source=tree_reg)
  case(18)
    allocate(base,source=ridge_cv_base)
  end select
  pre_y=[(mod(j,2),j=1,n)]; call fit_template(base); before=predict_base(base,query)
  code%random_state=seed; code%code_size=code_size
  open(newunit=output,file=trim(output_path),status='replace')
  call model_state(base)
  if(batches==0) then
    call fit_all(1,n,.true.); call snapshot()
  else
    first=1
    do j=1,batches
      last=cuts(j)
      call fit_all(first,last,j==1.and.initial_fit==1); call snapshot(); first=last+1
    end do
  end if
  if(refit==1) then
    call fit_all(1,n,.true.); call snapshot()
  end if
  after=predict_base(base,query)
  call vector(before); call vector(after); call model_state(base); call matrix(x); call matrix(query)
  do j=1,n
    write(output,*) y(j,:)
  end do
  call vector(w); close(output)
contains
  subroutine vector(a)
    real(dp), intent(in) :: a(:)
    write(output,'(*(es25.16e3,1x))') a
  end subroutine
  subroutine matrix(a)
    real(dp), intent(in) :: a(:,:)
    integer :: row
    do row=1,size(a,1)
      call vector(a(row,:))
    end do
  end subroutine
  subroutine fit_all(first,last,fresh)
    integer, intent(in) :: first,last
    logical, intent(in) :: fresh
    real(dp), allocatable :: weights(:)
    if(weighted==1) weights=w(first:last)
    select case(mode)
    case(0)
      if(fresh) then
        if(kind==1) then
          call ovr%fit(x(first:last,:),y(first:last,:),base,weights)
        else
          call ovr%fit(x(first:last,:),y(first:last,1),base,weights)
        end if
      else
        if(kind==1) then
          call ovr%partial_fit(x(first:last,:),y(first:last,:),base,classes,weights)
        else
          call ovr%partial_fit(x(first:last,:),y(first:last,1),base,classes,weights)
        end if
      end if
    case(1)
      if(fresh) then
        call ovo%fit(x(first:last,:),y(first:last,1),base,weights)
      else
        call ovo%partial_fit(x(first:last,:),y(first:last,1),base,classes,weights)
      end if
    case(2)
      call code%fit(x(first:last,:),y(first:last,1),base,weights)
    end select
  end subroutine
  subroutine snapshot()
    integer :: h,row
    ready=1; multi=0
    select case(mode)
    case(0)
      heads=size(ovr%estimators_); nfeatures=ovr%n_features_in_; nclasses=ovr%n_classes_
      if(ovr%multilabel_) multi=1
      if(any(ovr%estimators_%n_features_in_==0)) ready=0
      has_proba=merge(1,0,all(ovr%estimators_%constant_).or.base_proba())
      has_margin=merge(1,0,all(ovr%estimators_%constant_).or.base_margin())
      write(output,*) nfeatures,nclasses,heads,multi,ready,has_proba,has_margin
      write(output,*) ovr%classes_
    case(1)
      heads=size(ovo%estimators_); nfeatures=ovo%n_features_in_; nclasses=ovo%n_classes_
      if(any(ovo%estimators_%n_features_in_==0)) ready=0
      has_proba=0; has_margin=1
      write(output,*) nfeatures,nclasses,heads,multi,ready,has_proba,has_margin
      write(output,*) ovo%classes_
    case(2)
      heads=size(code%estimators_); nfeatures=code%n_features_in_; nclasses=code%n_classes_
      has_proba=0; has_margin=0
      write(output,*) nfeatures,nclasses,heads,multi,ready,has_proba,has_margin
      write(output,*) code%classes_; call matrix(code%code_book_)
    end select
    if(ready==1) then
      select case(mode)
      case(0)
        if(multi==1) then
          indicator=ovr%predict_indicator(query)
        else
          prediction=ovr%predict(query); indicator=reshape(prediction,[m,1])
        end if
        if(multi==1) then
          write(output,'(*(es25.16e3,1x))') ovr%score(query,truth),ovr%score(query,truth,sw)
        else
          write(output,'(*(es25.16e3,1x))') ovr%score(query,truth(:,1)),ovr%score(query,truth(:,1),sw)
        end if
      case(1)
        prediction=ovo%predict(query); indicator=reshape(prediction,[m,1])
        write(output,'(*(es25.16e3,1x))') ovo%score(query,truth(:,1)),ovo%score(query,truth(:,1),sw)
      case(2)
        prediction=code%predict(query); indicator=reshape(prediction,[m,1])
        write(output,'(*(es25.16e3,1x))') code%score(query,truth(:,1)),code%score(query,truth(:,1),sw)
      end select
      do row=1,m
        write(output,*) indicator(row,:)
      end do
      if(has_proba==1) call matrix(ovr%predict_proba(query))
      if(has_margin==1) then
        if(mode==0) call matrix(ovr%decision_function(query))
        if(mode==1) call matrix(ovo%decision_function(query))
      end if
    end if
    do h=1,heads
      select case(mode)
      case(0)
        status=2; value=ovr%estimators_(h)%constant_value_; features=ovr%estimators_(h)%n_features_in_
        if(ovr%estimators_(h)%constant_) status=1
        if(features==0) status=0
        write(output,*) status,value,features
        if(status==2) call normal_head(ovr%estimators_(h)%estimator_)
        if(status==1) call constant_head(value)
      case(1)
        status=2; value=ovo%estimators_(h)%constant_value_; features=ovo%estimators_(h)%n_features_in_
        if(ovo%estimators_(h)%constant_) status=1
        if(features==0) status=0
        write(output,*) status,value,features
        if(status==2) call normal_head(ovo%estimators_(h)%estimator_)
        if(status==1) call constant_head(value)
      case(2)
        status=2; value=code%estimators_(h)%constant_value_; features=code%estimators_(h)%n_features_in_
        if(code%estimators_(h)%constant_) status=1
        write(output,*) status,value,features
        if(status==2) call normal_head(code%estimators_(h)%estimator_)
        if(status==1) call constant_head(value)
      end select
    end do
  end subroutine
  logical function base_proba()
    base_proba=any(choice==[0,1,2,3,4,5,6,7])
    if(choice==2.and.loss==1) base_proba=.false.
  end function
  logical function base_margin()
    base_margin=any(choice==[0,2,8,9,10,11,12,18])
  end function
  subroutine constant_head(value)
    integer, intent(in) :: value
    real(dp) :: values(m)
    values=real(value,dp)
    call vector(values); call vector(values); write(output,*) 1; call vector(values)
  end subroutine
  subroutine normal_head(model)
    class(*), intent(in) :: model
    call vector(predict_base(model,query)); call vector(binary_confidence(model,query))
    write(output,*) merge(1,0,base_proba())
    if(base_proba()) call vector(binary_probability(model,query))
    call model_state(model)
  end subroutine
  subroutine fit_template(model)
    class(*), intent(inout) :: model
    select type(model)
    type is(logistic_regression)
      call model%fit(x,pre_y)
    type is(gaussian_nb)
      call model%fit(x,pre_y)
    type is(sgd_classifier)
      call model%fit(x,pre_y)
    type is(kneighbors_classifier)
      call model%fit(x,pre_y)
    type is(decision_tree_classifier)
      call model%fit(x,pre_y)
    class is(multinomial_nb)
      call model%fit(x,pre_y)
    type is(perceptron)
      call model%fit(x,pre_y)
    type is(passive_aggressive_classifier)
      call model%fit(x,pre_y)
    class is(ridge_classifier)
      call model%fit(x,pre_y)
    type is(linear_svc)
      call model%fit(x,pre_y)
    type is(svc)
      call model%fit(x,pre_y)
    class is(linear_regression)
      call model%fit(x,real(pre_y,dp))
    type is(sgd_regressor)
      call model%fit(x,real(pre_y,dp))
    type is(kneighbors_regressor)
      call model%fit(x,real(pre_y,dp))
    type is(decision_tree_regressor)
      call model%fit(x,real(pre_y,dp))
    end select
  end subroutine
  function predict_base(model,q) result(r)
    class(*), intent(in) :: model
    real(dp), intent(in) :: q(:,:)
    real(dp), allocatable :: r(:)
    select type(model)
    type is(logistic_regression)
      r=real(model%predict(q),dp)
    type is(gaussian_nb)
      r=real(model%predict(q),dp)
    type is(sgd_classifier)
      r=real(model%predict(q),dp)
    type is(kneighbors_classifier)
      r=real(model%predict(q),dp)
    type is(decision_tree_classifier)
      r=real(model%predict(q),dp)
    class is(multinomial_nb)
      r=real(model%predict(q),dp)
    type is(perceptron)
      r=real(model%predict(q),dp)
    type is(passive_aggressive_classifier)
      r=real(model%predict(q),dp)
    class is(ridge_classifier)
      r=real(model%predict(q),dp)
    type is(linear_svc)
      r=real(model%predict(q),dp)
    type is(svc)
      r=real(model%predict(q),dp)
    class is(linear_regression)
      r=model%predict(q)
    type is(sgd_regressor)
      r=model%predict(q)
    type is(kneighbors_regressor)
      r=model%predict(q)
    type is(decision_tree_regressor)
      r=model%predict(q)
    end select
  end function
  function binary_confidence(model,q) result(r)
    class(*), intent(in) :: model
    real(dp), intent(in) :: q(:,:)
    real(dp), allocatable :: r(:),a(:,:)
    select type(model)
    type is(logistic_regression)
      a=model%decision_function(q); r=a(:,1)
    type is(gaussian_nb)
      a=model%predict_proba(q); r=a(:,2)
    type is(sgd_classifier)
      a=model%decision_function(q); r=a(:,1)
    type is(kneighbors_classifier)
      a=model%predict_proba(q); r=a(:,2)
    type is(decision_tree_classifier)
      a=model%predict_proba(q); r=a(:,2)
    class is(multinomial_nb)
      a=model%predict_proba(q); r=a(:,2)
    type is(perceptron)
      a=model%decision_function(q); r=a(:,1)
    type is(passive_aggressive_classifier)
      a=model%decision_function(q); r=a(:,1)
    class is(ridge_classifier)
      a=model%decision_function(q); r=a(:,1)
    type is(linear_svc)
      a=model%decision_function(q); r=a(:,1)
    type is(svc)
      a=model%decision_function(q); r=a(:,1)
    class is(linear_regression)
      r=model%predict(q)
    type is(sgd_regressor)
      r=model%predict(q)
    type is(kneighbors_regressor)
      r=model%predict(q)
    type is(decision_tree_regressor)
      r=model%predict(q)
    end select
  end function
  function binary_probability(model,q) result(r)
    class(*), intent(in) :: model
    real(dp), intent(in) :: q(:,:)
    real(dp), allocatable :: r(:),a(:,:)
    select type(model)
    type is(logistic_regression)
      a=model%predict_proba(q)
    type is(gaussian_nb)
      a=model%predict_proba(q)
    type is(sgd_classifier)
      a=model%predict_proba(q)
    type is(kneighbors_classifier)
      a=model%predict_proba(q)
    type is(decision_tree_classifier)
      a=model%predict_proba(q)
    class is(multinomial_nb)
      a=model%predict_proba(q)
    end select
    r=a(:,2)
  end function
  subroutine model_state(model)
    class(*), intent(in) :: model
    integer :: j
    select type(model)
    type is(logistic_regression)
      write(output,*) size(model%classes_)
      write(output,*) model%classes_
      write(output,*) 1,size(model%coef_,1),size(model%coef_,2)
      call matrix(model%coef_); call vector(model%intercept_)
    type is(gaussian_nb)
      write(output,*) size(model%classes_)
      write(output,*) model%classes_
      write(output,*) 2,size(model%theta_,1),size(model%theta_,2)
      call matrix(model%theta_); call matrix(model%var_)
      call vector(model%class_prior_); call vector(model%class_count_)
    type is(sgd_classifier)
      write(output,*) size(model%classes_)
      write(output,*) model%classes_
      write(output,*) 3,size(model%coef_,1),size(model%coef_,2)
      call matrix(model%coef_); call vector([model%intercept_,model%t_])
    type is(kneighbors_classifier)
      write(output,*) size(model%classes_)
      write(output,*) model%classes_
      write(output,*) 6,size(model%search_%x_,1),size(model%search_%x_,2)
      call matrix(model%search_%x_); call vector(real(model%y_,dp))
    type is(decision_tree_classifier)
      write(output,*) size(model%classes_)
      write(output,*) model%classes_
      write(output,*) 5,size(model%classes_),model%node_count_
      do j=1,model%node_count_
        associate(node=>model%nodes_(j))
          write(output,'(5(i0,1x),*(es25.16e3,1x))') &
            node%left,node%right,node%feature,node%n_samples,node%depth,node%threshold,node%impurity,node%probability
        end associate
      end do
    class is(multinomial_nb)
      write(output,*) size(model%classes_)
      write(output,*) model%classes_
      write(output,*) 4,size(model%feature_log_prob_,1),size(model%feature_log_prob_,2)
      call matrix(model%feature_log_prob_); call vector(model%class_log_prior_)
    type is(perceptron)
      write(output,*) size(model%classes_)
      write(output,*) model%classes_
      write(output,*) 1,size(model%coef_,1),size(model%coef_,2)
      call matrix(model%coef_); call vector(model%intercept_)
    type is(passive_aggressive_classifier)
      write(output,*) size(model%classes_)
      write(output,*) model%classes_
      write(output,*) 3,size(model%coef_,1),size(model%coef_,2)
      call matrix(model%coef_); call vector([model%intercept_,model%t_])
    class is(ridge_classifier)
      write(output,*) size(model%classes_)
      write(output,*) model%classes_
      write(output,*) 1,size(model%coef_,1),size(model%coef_,2)
      call matrix(model%coef_); call vector(model%intercept_)
    type is(linear_svc)
      write(output,*) size(model%classes_)
      write(output,*) model%classes_
      write(output,*) 1,size(model%coef_,1),size(model%coef_,2)
      call matrix(model%coef_); call vector(model%intercept_)
    type is(svc)
      write(output,*) size(model%classes_)
      write(output,*) model%classes_
      write(output,*) 8,size(model%x_fit_,1),size(model%x_fit_,2)
      call matrix(model%x_fit_); call vector(model%models_(1)%dual)
      call vector([model%models_(1)%intercept])
    class is(linear_regression)
      write(output,*) 0
      write(output,*) 9,1,size(model%coef_)
      call vector(model%coef_); call vector([model%intercept_])
    type is(sgd_regressor)
      write(output,*) 0
      write(output,*) 11,1,size(model%coef_)
      call vector(model%coef_); call vector([model%intercept_,model%t_])
    type is(kneighbors_regressor)
      write(output,*) 0
      write(output,*) 6,size(model%search_%x_,1),size(model%search_%x_,2)
      call matrix(model%search_%x_); call vector(real(model%y_,dp))
    type is(decision_tree_regressor)
      write(output,*) 0
      write(output,*) 7,0,model%node_count_
      do j=1,model%node_count_
        associate(node=>model%nodes_(j))
          write(output,'(5(i0,1x),3(es25.16e3,1x))') &
            node%left,node%right,node%feature,node%n_samples,node%depth,node%threshold,node%value,node%impurity
        end associate
      end do
    end select
  end subroutine
end program
