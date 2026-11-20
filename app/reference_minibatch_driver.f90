program reference_minibatch_driver
  use sklearn_minibatch_kmeans, only: mini_batch_kmeans
  use sklearn_common, only: dp
  implicit none
  type(mini_batch_kmeans) :: model
  real(dp), allocatable :: x(:,:),query(:,:),w(:),qw(:),values(:,:)
  integer, allocatable :: labels(:)
  integer :: n,p,nq,op,chunk,input,output,i,j,explicit_centers,step
  character(512) :: path
  character(32) :: prefix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p,nq,op,chunk,explicit_centers
  read(input,*) model%n_clusters,model%n_init,model%max_iter,model%random_state,model%batch_size,model%init_size
  read(input,*) model%tol,model%reassignment_ratio,model%max_no_improvement,model%compute_labels,model%init
  allocate(x(n,p),query(nq,p),w(n),qw(nq))
  do i=1,n
    read(input,*) x(i,:),w(i)
  end do
  do i=1,nq
    read(input,*) query(i,:),qw(i)
  end do
  if(explicit_centers==1) then
    allocate(model%initial_centers(model%n_clusters,p))
    do i=1,model%n_clusters
      read(input,*) model%initial_centers(i,:)
    end do
  end if
  close(input)
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  select case(op)
  case(2,5,6)
    if(op==5) call model%fit(x,w)
    step=0
    do i=1,n,chunk
      j=min(n,i+chunk-1); call model%partial_fit(x(i:j,:),w(i:j)); step=step+1
      write(prefix,'(a,i0,a)') 'step',step,'_'
      call snapshot(trim(prefix))
    end do
    if(op==6) call model%fit(x,w)
  case(3); labels=model%fit_predict(x,w); call vector('fit_predict',real(labels,dp))
  case(4); values=model%fit_transform(x,w); call matrix('fit_transform',values)
  case default; call model%fit(x,w)
  end select
  call snapshot('')
  call vector('predict',real(model%predict(query),dp)); call matrix('transform',model%transform(query))
  call vector('score',[model%score(x),model%score(x,w),model%score(query,qw)])
  close(output)
contains
  subroutine snapshot(prefix)
    character(*), intent(in) :: prefix
    call matrix(prefix//'centers',model%cluster_centers_); call vector(prefix//'counts',model%counts_)
    call vector(prefix//'meta',[real(model%n_features_in_,dp),real(model%n_steps_,dp)])
    if(model%n_iter_>0) call vector(prefix//'n_iter',[real(model%n_iter_,dp)])
    if(allocated(model%labels_)) call vector(prefix//'labels',real(model%labels_,dp))
    if(model%has_inertia_) call vector(prefix//'inertia',[model%inertia_])
  end subroutine
  subroutine matrix(name,a)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:,:)
    call vector(name,reshape(a,[size(a)]))
  end subroutine
  subroutine vector(name,a)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:)
    write(output,'(a,1x,*(es25.16e3,1x))') name,a
  end subroutine
end program
