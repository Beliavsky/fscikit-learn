program reference_bisecting_driver
  use sklearn_bisecting_kmeans, only: bisecting_kmeans
  use sklearn_common, only: dp
  implicit none
  type(bisecting_kmeans) :: model
  real(dp), allocatable :: x(:,:),query(:,:),weights(:),query_weights(:),transformed(:,:)
  integer, allocatable :: fitted_labels(:)
  integer :: n,p,nq,operation,input,output,i
  character(512) :: path
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p,nq,operation
  read(input,*) model%n_clusters,model%n_init,model%max_iter,model%random_state,model%tol
  read(input,*) model%init,model%bisecting_strategy,model%algorithm
  allocate(x(n,p),query(nq,p),weights(n),query_weights(nq))
  do i=1,n
    read(input,*) x(i,:),weights(i)
  end do
  do i=1,nq
    read(input,*) query(i,:),query_weights(i)
  end do
  close(input)
  select case(operation)
  case(2); fitted_labels=model%fit_predict(x,weights)
  case(3); transformed=model%fit_transform(x,weights)
  case default; call model%fit(x,weights)
  end select
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  call matrix('centers',model%cluster_centers_); call vector('labels',real(model%labels_,dp))
  call vector('inertia',[model%inertia_]); call vector('predict',real(model%predict(query),dp))
  call vector('train_predict',real(model%predict(x),dp)); call matrix('transform',model%transform(query))
  call vector('score',[model%score(x),model%score(x,weights),model%score(query,query_weights)])
  call matrix('tree_centers',model%tree_centers_); call matrix('children',real(model%children_,dp))
  call vector('tree_labels',real(model%tree_leaf_labels_,dp)); call vector('tree_counts',real(model%tree_sample_counts_,dp))
  call vector('tree_scores',model%tree_scores_); call vector('meta',real([model%n_features_in_,model%node_count_],dp))
  if(allocated(fitted_labels)) call vector('fit_predict',real(fitted_labels,dp))
  if(allocated(transformed)) call matrix('fit_transform',transformed)
  close(output)
contains
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
