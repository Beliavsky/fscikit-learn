program reference_stacking_driver
  use sklearn
  use sklearn_base_dispatch, only: base_proba,base_decision,train_classifier,train_regressor
  use sklearn_weighted_linear, only: weighted_ridge_loo
  implicit none
  type(stacking_classifier) :: clf
  type(stacking_regressor) :: reg
  type(stacking_estimator), allocatable :: bases(:)
  type(index_split), allocatable :: plans(:)
  type(logistic_regression) :: logistic
  type(logistic_regression) :: standalone
  type(gaussian_nb) :: nb
  type(ridge_classifier) :: rc
  type(decision_tree_classifier) :: tc
  type(extra_tree_classifier) :: ec
  type(kneighbors_classifier) :: kc
  type(dummy_classifier) :: dc
  type(linear_svc) :: linear_svm
  type(linear_regression) :: ols
  type(ridge) :: rr
  type(kneighbors_regressor) :: kr
  type(decision_tree_regressor) :: tr
  type(extra_tree_regressor) :: er
  type(dummy_regressor) :: dr
  type(kernel_ridge) :: kernel
  type(ridge_cv) :: loo
  class(*), allocatable :: final
  integer, allocatable :: choices(:),drop(:)
  real(dp), allocatable :: x(:,:),q(:,:),y(:),w(:),value(:,:)
  integer :: mode,n,p,m,b,fc,cv,pass,prefit,weighted,method,ns,seed,fi,input,out,i,j,f,nt,nv,flag
  character(1024) :: path
  character(20) :: name
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) mode,n,p,m,b,fc,cv,pass,prefit,weighted,method,ns,seed,fi
  allocate(x(n,p),q(m,p),y(n),w(n),choices(b),drop(b),bases(b))
  read(input,*) choices; read(input,*) drop
  do i=1,n
    read(input,*) x(i,:)
  end do
  read(input,*) y; read(input,*) w
  do i=1,m
    read(input,*) q(i,:)
  end do
  if(ns>0) then
    allocate(plans(ns))
    do f=1,ns
      read(input,*) nt,nv; allocate(plans(f)%train(nt),plans(f)%test(nv))
      read(input,*) plans(f)%train; read(input,*) plans(f)%test
    end do
  end if
  close(input)
  logistic%c=.7_dp; logistic%tol=1.e-10_dp; logistic%max_iter=500
  rc%alpha=.7_dp; rr%alpha=.7_dp
  tc%max_depth=2; tr%max_depth=2; ec%max_depth=2; er%max_depth=2
  tc%random_state=seed; tr%random_state=seed; ec%random_state=seed; er%random_state=seed
  kc%n_neighbors=3; kr%n_neighbors=3
  linear_svm%c=.7_dp; linear_svm%tol=1.e-10_dp; linear_svm%max_iter=10000
  kernel%kernel='rbf'; kernel%gamma=.4_dp; kernel%alpha=.7_dp
  do j=1,b
    write(name,'(a,i0)') 'model',j
    bases(j)%name=name; bases(j)%drop=drop(j)==1
    if(bases(j)%drop) cycle
    if(mode==0) then
      select case(choices(j))
      case(0)
        allocate(bases(j)%estimator,source=logistic)
      case(1)
        allocate(bases(j)%estimator,source=nb)
      case(2)
        allocate(bases(j)%estimator,source=rc)
      case(3)
        allocate(bases(j)%estimator,source=tc)
      case(4)
        allocate(bases(j)%estimator,source=ec)
      case(5)
        allocate(bases(j)%estimator,source=kc)
      case(6)
        allocate(bases(j)%estimator,source=dc)
      case(7)
        allocate(bases(j)%estimator,source=linear_svm)
      end select
      if(prefit==1) then
        select type(model=>bases(j)%estimator)
        class is(decision_tree_classifier)
          call model%fit(x,int(y),spread(1._dp,1,n))
        class default
          call train_classifier(bases(j)%estimator,x,int(y))
        end select
      end if
    else
      select case(choices(j))
      case(0)
        allocate(bases(j)%estimator,source=ols)
      case(1)
        allocate(bases(j)%estimator,source=rr)
      case(2)
        allocate(bases(j)%estimator,source=kr)
      case(3)
        allocate(bases(j)%estimator,source=tr)
      case(4)
        allocate(bases(j)%estimator,source=er)
      case(5)
        allocate(bases(j)%estimator,source=dr)
      case(6)
        allocate(bases(j)%estimator,source=kernel)
      end select
      if(prefit==1) then
        select type(model=>bases(j)%estimator)
        class is(decision_tree_regressor)
          call model%fit(x,y,spread(1._dp,1,n))
        class default
          call train_regressor(bases(j)%estimator,x,y)
        end select
      end if
    end if
  end do
  if(mode==0) then
    select case(fc)
    case(0)
      allocate(final,source=logistic)
    case(1)
      allocate(final,source=nb)
    case(2)
      allocate(final,source=rc)
    case(3)
      allocate(final,source=tc)
    end select
    clf%cv=cv; clf%passthrough=pass==1; clf%prefit=prefit==1
    select case(method)
    case(1)
      clf%stack_method='predict_proba'
    case(2)
      clf%stack_method='decision_function'
    case(3)
      clf%stack_method='predict'
    end select
  else
    select case(fc)
    case(0)
      allocate(final,source=ols)
    case(1)
      allocate(final,source=rr)
    case(2)
      allocate(final,source=kr)
    case(3)
      allocate(final,source=tr)
    end select
    reg%cv=cv; reg%passthrough=pass==1; reg%prefit=prefit==1
  end if
  call get_command_argument(2,path); open(newunit=out,file=trim(path),status='replace')
  if(mode==3) then
    standalone%tol=1.e-10_dp; standalone%max_iter=500
    if(weighted==1) then
      call standalone%fit(x,int(y),w)
    else
      call standalone%fit(x,int(y))
    end if
    write(out,*) size(standalone%coef_,1)
    call matrix(standalone%coef_); write(out,*) standalone%intercept_
    value=standalone%predict_proba(q); call matrix(value)
    write(out,*) standalone%predict(q)
    close(out); stop
  else if(mode==2) then
    loo%cv=0; loo%fit_intercept=fi==1; loo%alphas=[.1_dp,1._dp,10._dp]
    if(weighted==1) then
      call weighted_ridge_loo(loo,x,y,loo%alphas,loo%alpha_,loo%best_score_,loo%cv_scores_,w)
    else
      call loo%fit(x,y)
    end if
    write(out,*) loo%alpha_,loo%best_score_; write(out,*) loo%cv_scores_
    write(out,*) loo%coef_; write(out,*) loo%intercept_; write(out,*) loo%predict(q)
    close(out); stop
  else if(mode==0) then
    if(weighted==1) then
      call clf%fit(x,int(y),bases,final,w,plans)
    else
      call clf%fit(x,int(y),bases,final,splits=plans)
    end if
    flag=merge(1,0,base_proba(clf%final_estimator_))+2*merge(1,0,base_decision(clf%final_estimator_))
    write(out,*) clf%n_features_in_,clf%n_features_out_,size(clf%estimators_),clf%n_classes_,flag
    write(out,*) clf%classes_; write(out,*) clf%active_indices_; write(out,*) clf%n_feature_outs_
    do j=1,size(clf%estimators_)
      write(out,*) trim(clf%stack_method_(j))
    end do
    write(out,*) clf%predict(q); write(out,*) clf%predict(x)
    if(btest(flag,0)) then
      value=clf%predict_proba(q); call matrix(value)
    end if
    if(btest(flag,1)) then
      value=clf%decision_function(q); write(out,*) size(value,2); call matrix(value)
    end if
    write(out,*) clf%score(x,int(y),w)
    call matrix(clf%training_meta_features_)
    value=clf%transform(q); call matrix(value)
    value=clf%transform(x); call matrix(value)
    call state(clf%final_estimator_)
    do j=1,size(clf%estimators_)
      call state(clf%estimators_(j)%estimator)
    end do
    if(allocated(clf%cv_splits_)) then
      write(out,*) size(clf%cv_splits_)
      do f=1,size(clf%cv_splits_)
        write(out,*) size(clf%cv_splits_(f)%train),size(clf%cv_splits_(f)%test)
        write(out,*) clf%cv_splits_(f)%train; write(out,*) clf%cv_splits_(f)%test
        do j=1,size(clf%estimators_)
          call state(clf%cv_estimators_(f,j)%estimator)
        end do
      end do
    else
      write(out,*) 0
    end if
  else
    if(weighted==1) then
      call reg%fit(x,y,bases,final,w,plans)
    else
      call reg%fit(x,y,bases,final,splits=plans)
    end if
    write(out,*) reg%n_features_in_,reg%n_features_out_,size(reg%estimators_),0,0
    write(out,*) reg%active_indices_; write(out,*) reg%n_feature_outs_
    do j=1,size(reg%estimators_)
      write(out,*) trim(reg%stack_method_(j))
    end do
    write(out,*) reg%predict(q); write(out,*) reg%predict(x); write(out,*) reg%score(x,y,w)
    call matrix(reg%training_meta_features_)
    value=reg%transform(q); call matrix(value)
    value=reg%transform(x); call matrix(value)
    call state(reg%final_estimator_)
    do j=1,size(reg%estimators_)
      call state(reg%estimators_(j)%estimator)
    end do
    if(allocated(reg%cv_splits_)) then
      write(out,*) size(reg%cv_splits_)
      do f=1,size(reg%cv_splits_)
        write(out,*) size(reg%cv_splits_(f)%train),size(reg%cv_splits_(f)%test)
        write(out,*) reg%cv_splits_(f)%train; write(out,*) reg%cv_splits_(f)%test
        do j=1,size(reg%estimators_)
          call state(reg%cv_estimators_(f,j)%estimator)
        end do
      end do
    else
      write(out,*) 0
    end if
  end if
  call matrix(x); call matrix(q); write(out,*) y; write(out,*) w
  close(out)
contains
  subroutine matrix(a)
    real(dp), intent(in) :: a(:,:)
    integer :: row
    do row=1,size(a,1)
      write(out,*) a(row,:)
    end do
  end subroutine
  subroutine state(model)
    class(*), intent(in) :: model
    integer :: node
    select type(model)
    class is(decision_tree_classifier)
      write(out,*) 1,model%node_count_,model%random_state
      write(out,*) model%classes_; write(out,*) model%feature_importances_
      do node=1,model%node_count_
        associate(a=>model%nodes_(node))
          write(out,*) a%left,a%right,a%feature,a%n_samples,a%depth,merge(1,0,a%missing_go_left)
          write(out,*) a%threshold,a%weighted_n_samples,a%impurity
          write(out,*) a%probability
        end associate
      end do
    class is(decision_tree_regressor)
      write(out,*) 1,model%node_count_,model%random_state
      write(out,*) model%feature_importances_
      do node=1,model%node_count_
        associate(a=>model%nodes_(node))
          write(out,*) a%left,a%right,a%feature,a%n_samples,a%depth,merge(1,0,a%missing_go_left)
          write(out,*) a%threshold,a%weighted_n_samples,a%impurity
          write(out,*) a%value
        end associate
      end do
    type is(ridge_cv)
      write(out,*) 3,size(model%coef_),0
      write(out,*) model%coef_; write(out,*) model%intercept_; write(out,*) model%alpha_,model%best_score_
      write(out,*) model%cv_scores_
    type is(logistic_regression)
      write(out,*) 2,size(model%coef_,1),size(model%coef_,2)
      call matrix(model%coef_); write(out,*) model%intercept_
    type is(linear_svc)
      write(out,*) 5,size(model%coef_,1),size(model%coef_,2)
      call matrix(model%coef_); write(out,*) model%intercept_
      write(out,*) model%gradient_norm_
    class is(linear_regression)
      write(out,*) 4,size(model%coef_),0
      write(out,*) model%coef_; write(out,*) model%intercept_
    class default
      write(out,*) 0,0,0
    end select
  end subroutine
end program
