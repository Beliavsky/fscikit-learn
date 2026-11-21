program reference_logistic_cv_driver
  use sklearn_linear_model, only: logistic_regression_cv,logistic_regression
  use sklearn_common, only: dp
  implicit none
  type(logistic_regression_cv) :: model
  type(logistic_regression) :: base
  real(dp), allocatable :: x(:,:),query(:,:),weight(:),test_weight(:),prob(:,:),decision(:,:),base_weight(:)
  integer, allocatable :: y(:),truth(:),predicted(:)
  integer :: input,output,n,p,m,ncs,nfolds,weighted,classmode,intercept,refit,i,j,f,nt,nv
  character(512) :: file
  call get_command_argument(1,file); open(newunit=input,file=trim(file),status='old')
  call get_command_argument(2,file); open(newunit=output,file=trim(file),status='replace')
  read(input,*) n,p,m,ncs,nfolds,weighted,classmode
  read(input,*) model%n_cs,model%cv,model%max_iter,model%tol,intercept,refit,model%scoring
  model%fit_intercept=intercept==1; model%refit=refit==1; model%class_weight_balanced=classmode==1
  if(ncs>0) then
    allocate(model%cs(ncs)); read(input,*) model%cs
  end if
  allocate(x(n,p),y(n),query(m,p),truth(m),test_weight(m))
  do i=1,n
    read(input,*) x(i,:)
  end do
  read(input,*) y
  if(weighted==1) then
    allocate(weight(n)); read(input,*) weight
  end if
  do i=1,m
    read(input,*) query(i,:)
  end do
  read(input,*) truth; read(input,*) test_weight
  if(classmode==2) then
    read(input,*) j
    allocate(model%class_weight(j)); read(input,*) model%class_weight
  end if
  if(nfolds>0) then
    allocate(model%splits(nfolds))
    do f=1,nfolds
      read(input,*) nt,nv; allocate(model%splits(f)%train(nt),model%splits(f)%test(nv))
      read(input,*) model%splits(f)%train; read(input,*) model%splits(f)%test
    end do
  end if
  call model%fit(x,y,weight)
  call emit('classes',real(model%classes_,dp)); call emit('path_classes',real(model%path_classes_,dp))
  call emit('grid',model%cs_); call emit('selected',model%c_); call emit('cw',model%class_weight_)
  call emit('scores',reshape(model%scores_,[size(model%scores_)]))
  call emit('paths',reshape(model%coefs_paths_,[size(model%coefs_paths_)]))
  call emit('init',reshape(model%refit_init_,[size(model%refit_init_)]))
  call emit('coef',reshape(model%coef_,[size(model%coef_)])); call emit('bias',model%intercept_)
  prob=model%predict_proba(query); decision=model%decision_function(query); predicted=model%predict(query)
  call emit('proba',reshape(prob,[size(prob)])); call emit('decision',reshape(decision,[size(decision)]))
  call emit('predict',real(predicted,dp)); call emit('score',[model%score(query,truth)])
  call emit('weighted_score',[model%score(query,truth,test_weight)])
  call emit('best',real(model%best_c_indices_,dp)); call emit('features',[real(model%n_features_in_,dp)])
  call emit('native_iterations',real(reshape(model%n_iter_path_,[size(model%n_iter_path_)]),dp))
  call emit('native_converged',real(reshape(merge(1,0,model%converged_path_),[size(model%converged_path_)]),dp))
  ! Base LogisticRegression now accepts the source optional score weights.
  base%classes_=model%classes_; base%coef_=model%coef_; base%intercept_=model%intercept_
  call emit('base_weighted_accuracy',[base%score(query,truth,test_weight)])
  base_weight=test_weight; base_weight(1)=-.5_dp
  call emit('base_signed_accuracy',[base%score(query,truth,base_weight)])
  do f=1,size(model%cv_splits_)
    write(output,'(a,i0,*(1x,i0))') 'train',f,model%cv_splits_(f)%train
    write(output,'(a,i0,*(1x,i0))') 'test',f,model%cv_splits_(f)%test
  end do
  ! Genuine refit with a new width, class set and configuration.
  if(allocated(model%cs)) deallocate(model%cs)
  if(allocated(model%splits)) deallocate(model%splits)
  if(allocated(model%class_weight)) deallocate(model%class_weight)
  model%cv=2; model%n_cs=2; model%scoring='accuracy'; model%class_weight_balanced=.false.; model%refit=.true.
  call model%fit(reshape([-2._dp,-1._dp,-.5_dp,.5_dp,1._dp,2._dp],[6,1]),[9,9,4,9,4,4])
  call emit('refit_classes',real(model%classes_,dp)); call emit('refit_features',[real(model%n_features_in_,dp)])
  call emit('refit_grid',model%cs_)
contains
  subroutine emit(name,values)
    character(*), intent(in) :: name
    real(dp), intent(in) :: values(:)
    write(output,'(a,*(1x,es25.16e3))') name,values
  end subroutine
end program
