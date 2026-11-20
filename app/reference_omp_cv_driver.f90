program reference_omp_cv_driver
  use sklearn_omp_cv, only: orthogonal_matching_pursuit_cv
  use sklearn_common, only: dp
  implicit none
  type(orthogonal_matching_pursuit_cv) :: model
  real(dp), allocatable :: x(:,:),y(:),query(:,:),prediction(:)
  integer, allocatable :: train(:),test(:)
  integer :: input,output,n,p,nq,nops,k,i,f,custom,nf
  character(512) :: file
  character(24) :: prefix
  call get_command_argument(1,file); open(newunit=input,file=trim(file),status='old')
  call get_command_argument(2,file); open(newunit=output,file=trim(file),status='replace')
  read(input,*) nops
  do k=1,nops
    read(input,*) n,p,nq,model%cv,model%max_iter,model%fit_intercept,custom,nf
    if(allocated(model%test_fold)) deallocate(model%test_fold)
    if(allocated(model%train_mask)) deallocate(model%train_mask,model%test_mask)
    if(custom==1) then
      allocate(model%test_fold(n)); read(input,*) model%test_fold
    else if(custom==2) then
      allocate(model%train_mask(n,nf),model%test_mask(n,nf),train(n),test(n))
      do f=1,nf
        read(input,*) train; read(input,*) test
        model%train_mask(:,f)=train/=0; model%test_mask(:,f)=test/=0
      end do
      deallocate(train,test)
    end if
    if(allocated(x)) deallocate(x,y,query)
    allocate(x(n,p),y(n),query(nq,p))
    do i=1,n
      read(input,*) x(i,:),y(i)
    end do
    do i=1,nq
      read(input,*) query(i,:)
    end do
    call model%fit(x,y); write(prefix,'(i0,a)') k,'_'
    call emit('coef',model%coef_); call emit('intercept',[model%intercept_])
    prediction=model%predict(x); call emit('train',prediction)
    prediction=model%predict(query); call emit('query',prediction)
    call emit('score',[model%score(x,y)]); call emit('best_mse',[model%best_mse_])
    call emit('mse',reshape(model%mse_path_,[size(model%mse_path_)]))
    call emit('steps',real(model%fold_n_iter_,dp)); call emit('train_sizes',real(model%fold_train_sizes_,dp))
    call emit('test_sizes',real(model%fold_test_sizes_,dp))
    call emit('meta',real([model%n_nonzero_coefs_,model%n_iter_,model%n_features_in_,model%n_splits_],dp))
  end do
  close(input); close(output)
contains
  subroutine emit(name,values)
    character(*), intent(in) :: name
    real(dp), intent(in) :: values(:)
    write(output,'(a,*(1x,es25.17))') trim(prefix)//name,values
  end subroutine
end program
