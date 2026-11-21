program reference_threshold_driver
  use sklearn_common, only: dp
  use sklearn_threshold
  use sklearn_linear_model, only: logistic_regression,logistic_regression_cv
  use sklearn_naive_bayes, only: gaussian_nb
  use sklearn_ridge_classification, only: ridge_classifier
  use sklearn_neighbors, only: kneighbors_classifier
  use sklearn_tree, only: decision_tree_classifier
  use sklearn_dummy, only: dummy_classifier
  use sklearn_base_dispatch, only: train_classifier,probabilities,decisions
  implicit none
  type(fixed_threshold_classifier) :: fixed
  type(tuned_threshold_classifier_cv) :: tuned
  type(logistic_regression) :: lr
  type(logistic_regression_cv) :: lrcv
  type(gaussian_nb) :: nb
  type(ridge_classifier) :: ridge
  type(kneighbors_classifier) :: knn
  type(decision_tree_classifier) :: tree
  type(dummy_classifier) :: dummy
  class(*), allocatable :: base,fold_model
  real(dp), allocatable :: x(:,:),query(:,:),weight(:),score_weight(:),grid(:),prob(:,:),decision(:,:)
  real(dp) :: cutoff,fraction
  integer, allocatable :: y(:),truth(:),predicted(:),train(:),test(:)
  integer :: n,p,m,kind,weighted,scored,mode,refit,positive_option,ngrid,nth,cv,store,cutoff_auto,i,j,u,v,positive
  character(20) :: method,scoring
  character(1024) :: input,output
  call get_command_argument(1,input); call get_command_argument(2,output)
  open(newunit=u,file=trim(input),status='old',action='read')
  read(u,*) n,p,m,kind,weighted,scored,mode,refit,positive_option,ngrid,nth,cv,store,cutoff_auto
  read(u,*) method,scoring,cutoff,fraction
  allocate(x(n,p),query(m,p),y(n),truth(m))
  do i=1,n
    read(u,*) x(i,:)
  end do
  read(u,*) y
  if(weighted==1) then
    allocate(weight(n)); read(u,*) weight
  end if
  if(scored==1) then
    allocate(score_weight(n)); read(u,*) score_weight
  end if
  do i=1,m
    read(u,*) query(i,:)
  end do
  read(u,*) truth
  if(ngrid>0) then
    allocate(grid(ngrid)); read(u,*) grid; tuned%thresholds=grid
  end if
  if(mode==2.or.mode==4) then
    allocate(tuned%splits(merge(2,1,mode==2)))
    do i=1,size(tuned%splits)
      read(u,*) j,ngrid
      allocate(tuned%splits(i)%train(j),tuned%splits(i)%test(ngrid))
      read(u,*) tuned%splits(i)%train; read(u,*) tuned%splits(i)%test
    end do
  end if
  close(u)
  lr%c=.7_dp; lr%tol=1.e-12_dp; lr%max_iter=1000
  ridge%alpha=.7_dp; tree%max_depth=3; tree%random_state=0
  lrcv%cs=[.1_dp,1._dp]; lrcv%cv=2; lrcv%tol=1.e-12_dp; lrcv%max_iter=1000
  select case(kind)
  case(1); allocate(base,source=lr)
  case(2); allocate(base,source=nb)
  case(3); allocate(base,source=ridge)
  case(4); allocate(base,source=knn)
  case(5); allocate(base,source=tree)
  case(6); allocate(base,source=dummy)
  case(7); allocate(base,source=lrcv)
  end select
  if(mode==3) call train_classifier(base,x,y,weight)
  fixed%prefit=mode==3; fixed%response_method=method
  if(cutoff_auto==0) fixed%threshold=cutoff
  positive=maxval(y)
  if(positive_option>0) then
    if(positive_option==1) positive=minval(y)
    fixed%pos_label=positive; tuned%pos_label=positive
  else if(any(scoring==[character(20)::'f1','precision','recall','jaccard'])) then
    positive=1
  end if
  call fixed%fit(x,y,base,weight)
  tuned%prefit=mode==3; tuned%refit=refit==1; tuned%response_method=method; tuned%scoring=scoring
  tuned%n_thresholds=nth; tuned%cv=cv; tuned%store_cv_results=store==1
  if(mode==1) tuned%validation_fraction=fraction
  if(scoring=='utility') then
    call tuned%fit(x,y,base,weight,utility,score_weight)
  else
    call tuned%fit(x,y,base,weight,score_weight=score_weight)
  end if
  open(newunit=v,file=trim(output),status='replace',action='write')
  call emit('classes',real(tuned%classes_,dp)); call emit('best',[tuned%best_threshold_,tuned%best_score_])
  call emit('fixed_predict',real(fixed%predict(query),dp)); call emit('predict',real(tuned%predict(query),dp))
  call emit('score',[fixed%score(query,truth),tuned%score(query,truth)])
  call emit('weighted_score',[fixed%score(query,truth,[(1+.1_dp*i,i=1,m)]), &
    tuned%score(query,truth,[(1+.1_dp*i,i=1,m)])])
  if(kind/=3) then
    prob=fixed%predict_proba(query); call emit('fixed_proba',reshape(prob,[size(prob)]))
    prob=tuned%predict_proba(query); call emit('proba',reshape(prob,[size(prob)]))
  end if
  if(kind==1.or.kind==3.or.kind==7) then
    decision=fixed%decision_function(query); call emit('fixed_decision',reshape(decision,[size(decision)]))
    decision=tuned%decision_function(query); call emit('decision',reshape(decision,[size(decision)]))
  end if
  call emit('features',[real(tuned%n_features_in_,dp)])
  call emit('positive',[real(tuned%positive_label_,dp)])
  if(store==1) then
    call emit('thresholds',tuned%thresholds_); call emit('mean_scores',tuned%mean_test_score_)
    call emit('fold_thresholds',reshape(tuned%fold_thresholds_,[size(tuned%fold_thresholds_)]))
    call emit('fold_scores',reshape(tuned%fold_scores_,[size(tuned%fold_scores_)]))
  end if
  do i=1,size(tuned%cv_splits_)
    write(v,'(a,i0,*(1x,i0))') 'train',i,tuned%cv_splits_(i)%train
    write(v,'(a,i0,*(1x,i0))') 'test',i,tuned%cv_splits_(i)%test
    train=tuned%cv_splits_(i)%train; test=tuned%cv_splits_(i)%test
    allocate(fold_model,source=base)
    if(mode/=3) then
      if(allocated(weight)) then
        call train_classifier(fold_model,x(train,:),y(train),weight(train))
      else
        call train_classifier(fold_model,x(train,:),y(train))
      end if
    end if
    if(tuned%response_method_used_=='predict_proba') then
      prob=probabilities(fold_model,x(test,:)); j=2
      if(tuned%positive_label_==tuned%classes_(1)) j=1
      write(v,'(a,i0,*(1x,es26.17))') 'response',i,prob(:,j)
    else
      decision=decisions(fold_model,x(test,:))
      if(tuned%positive_label_==tuned%classes_(1)) decision=-decision
      write(v,'(a,i0,*(1x,es26.17))') 'response',i,decision(:,1)
    end if
    select type(fold_model)
    class is(decision_tree_classifier)
      if(mode==3) train=[(j,j=1,n)]
      write(v,'(a,i0,*(1x,i0))') 'leaf_train',i,fold_model%apply(x(train,:))
      write(v,'(a,i0,*(1x,i0))') 'leaf_test',i,fold_model%apply(x(test,:))
    end select
    deallocate(fold_model)
  end do
  ! A genuine refit replaces classes, width, curve storage and the fitted base.
  tuned%prefit=.false.; tuned%refit=.true.; tuned%store_cv_results=.false.; tuned%validation_fraction=-1
  tuned%scoring='accuracy'; tuned%cv=2; tuned%n_thresholds=7; tuned%response_method='auto'
  if(allocated(tuned%pos_label)) deallocate(tuned%pos_label)
  if(allocated(tuned%splits)) deallocate(tuned%splits)
  if(allocated(tuned%thresholds)) deallocate(tuned%thresholds)
  call tuned%fit(reshape([0._dp,1._dp,2._dp,3._dp,4._dp,5._dp],[6,1]),[9,9,4,9,4,4],lr)
  call emit('refit_classes',real(tuned%classes_,dp)); call emit('refit_features',[real(tuned%n_features_in_,dp)])
  call emit('refit_stored',[merge(1._dp,0._dp,allocated(tuned%thresholds_))]); close(v)
contains
  subroutine emit(name,values)
    character(*), intent(in) :: name
    real(dp), intent(in) :: values(:)
    write(v,'(a,*(1x,es26.17))') name,values
  end subroutine
  real(dp) function utility(truth,prediction,sample_weight) result(value)
    integer, intent(in) :: truth(:),prediction(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp) :: cost(size(truth))
    cost=0
    where(truth/=positive.and.prediction==positive) cost=1
    where(truth==positive.and.prediction/=positive) cost=3
    value=-sum(cost)/size(cost)
    if(present(sample_weight)) value=-sum(cost*sample_weight)/sum(sample_weight)
  end function
end program
