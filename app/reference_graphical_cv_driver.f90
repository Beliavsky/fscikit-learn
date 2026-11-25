program reference_graphical_cv_driver
  use sklearn_covariance, only: graphical_lasso_cv
  use sklearn_common, only: dp
  implicit none
  type(graphical_lasso_cv) :: model
  real(dp), allocatable :: x(:,:),query(:,:),distances(:)
  integer :: input,output,n,p,m,na,nfolds,i,f,ntrain,ntest,centered
  character(512) :: file
  call get_command_argument(1,file); open(newunit=input,file=trim(file),status='old')
  call get_command_argument(2,file); open(newunit=output,file=trim(file),status='replace')
  call get_command_argument(3,file)
  if(len_trim(file)>0) model%mode=trim(file)
  read(input,*) n,p,m,na,nfolds
  read(input,*) model%n_alphas,model%n_refinements,model%cv,model%max_iter,centered, &
    model%tol,model%enet_tol,model%eps
  model%assume_centered=centered==1
  if(na>0) then
    allocate(model%alphas(na)); read(input,*) model%alphas
  end if
  allocate(x(n,p),query(m,p))
  do i=1,n
    read(input,*) x(i,:)
  end do
  do i=1,m
    read(input,*) query(i,:)
  end do
  if(nfolds>0) then
    allocate(model%splits(nfolds))
    do f=1,nfolds
      read(input,*) ntrain,ntest
      allocate(model%splits(f)%train(ntrain),model%splits(f)%test(ntest))
      read(input,*) model%splits(f)%train; read(input,*) model%splits(f)%test
    end do
  end if
  call model%fit(x)
  call emit('alpha',[model%alpha_]); call emit('best',[model%best_score_])
  call emit('grid',model%cv_results_%alphas)
  call emit('splits',reshape(model%cv_results_%split_test_score,[size(model%cv_results_%split_test_score)]))
  call emit('mean',model%cv_results_%mean_test_score); call emit('std',model%cv_results_%std_test_score)
  call emit('location',model%location_); call emit('covariance',reshape(model%covariance_,[p*p]))
  call emit('precision',reshape(model%precision_,[p*p])); call emit('costs',reshape(model%costs_,[size(model%costs_)]))
  call emit('n_iter',[real(model%n_iter_,dp)]); call emit('features',[real(model%n_features_in_,dp)])
  call emit('converged',[real(merge(1,0,model%converged_),dp)])
  call emit('endpoint_cov',reshape(model%cv_results_%empirical_covariance,[size(model%cv_results_%empirical_covariance)]))
  call emit('endpoint_precision',reshape(model%cv_results_%empirical_precision,[size(model%cv_results_%empirical_precision)]))
  call emit('endpoint_test', &
    reshape(model%cv_results_%empirical_test_covariance,[size(model%cv_results_%empirical_test_covariance)]))
  call emit('score',[model%score(query)])
  distances=model%mahalanobis(query); call emit('mahalanobis',distances)
  do f=1,size(model%cv_splits_)
    write(output,'(a,i0,*(1x,i0))') 'train',f,model%cv_splits_(f)%train
    write(output,'(a,i0,*(1x,i0))') 'test',f,model%cv_splits_(f)%test
  end do
contains
  subroutine emit(name,values)
    character(*), intent(in) :: name
    real(dp), intent(in) :: values(:)
    write(output,'(a,*(1x,es25.16e3))') name,values
  end subroutine
end program
