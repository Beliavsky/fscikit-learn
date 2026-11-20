program reference_minibatch_dictionary_driver
  use sklearn_dictionary, only: mini_batch_dictionary_learning,mini_batch_sparse_pca
  use sklearn_randomized_svd_core, only: randomized_svd
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_common, only: dp
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  type(mini_batch_dictionary_learning) :: model
  type(mini_batch_sparse_pca) :: projection
  type(legacy_random_state) :: rng
  real(dp), allocatable :: x(:,:),values(:,:),s(:),u(:,:),vt(:,:)
  real(dp) :: next_uniform,next_normal
  integer :: input,output,kind,nops,op,n,p,i,j,k,init_rows,rank,seed
  character(512) :: path
  character(32) :: prefix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  read(input,*) kind,nops
  do k=1,nops
    read(input,*) op,n,p
    if(kind==1) then
      read(input,*) model%n_components,model%max_iter,model%batch_size,model%random_state,model%max_no_improvement, &
        model%transform_n_nonzero_coefs,model%transform_max_iter
      read(input,*) model%alpha,model%tol,model%transform_alpha
      read(input,*) model%shuffle,model%positive_code,model%positive_dict,model%split_sign
      read(input,*) model%fit_algorithm,model%transform_algorithm
      read(input,*) init_rows
      if(allocated(model%dict_init)) deallocate(model%dict_init)
      if(init_rows>0) then
        allocate(model%dict_init(init_rows,p))
        do j=1,init_rows
          read(input,*) model%dict_init(j,:)
        end do
      end if
    else if(kind==2) then
      read(input,*) projection%n_components,projection%max_iter,projection%batch_size,projection%random_state, &
        projection%max_no_improvement
      read(input,*) projection%alpha,projection%tol,projection%ridge_alpha
      read(input,*) projection%shuffle,projection%method
    else
      read(input,*) rank,seed
    end if
    if(allocated(x)) deallocate(x)
    allocate(x(n,p))
    do i=1,n
      read(input,*) x(i,:)
    end do
    write(prefix,'(i0,a)') k,'_'
    if(kind==1) then
      select case(op)
      case(1); call model%fit(x)
      case(2); call model%partial_fit(x)
      case(3); values=model%fit_transform(x); call matrix('result',values)
      case(4); values=model%transform(x); call matrix('result',values)
      end select
      call matrix('components',model%components_); call matrix('a',model%a_); call matrix('b',model%b_)
      call vector('meta',real([int(model%n_features_in_,int64),int(model%n_components_,int64),int(model%batch_size_,int64), &
        model%n_steps_,model%n_iter_],dp))
      call vector('convergence',[model%ewa_cost_,model%ewa_cost_min_,real(model%no_improvement_,dp)])
    else if(kind==2) then
      select case(op)
      case(1); call projection%fit(x)
      case(3); values=projection%fit_transform(x); call matrix('result',values)
      case(4); values=projection%transform(x); call matrix('result',values)
      case(5); values=projection%inverse_transform(x); call matrix('result',values)
      end select
      call matrix('components',projection%components_); call vector('mean',projection%mean_)
      call vector('meta',[real(projection%n_features_in_,dp),real(projection%n_components_,dp),real(projection%n_iter_,dp)])
    else
      call rng%seed(seed); call randomized_svd(x,rank,rng,s,u,vt)
      call vector('singular',s); call matrix('left',u); call matrix('right',vt)
      next_uniform=rng%uniform(); next_normal=rng%normal(); call vector('rng',[next_uniform,next_normal])
    end if
  end do
  close(input); close(output)
contains
  subroutine matrix(name,a)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:,:)
    call vector(name,reshape(a,[size(a)]))
  end subroutine
  subroutine vector(name,a)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:)
    write(output,'(a,1x,*(es25.16e3,1x))') trim(prefix)//name,a
  end subroutine
end program
