program reference_hdbscan_driver
  use sklearn_hdbscan, only: hdbscan
  use sklearn_common, only: dp
  implicit none
  type(hdbscan) :: model
  real(dp), allocatable :: x(:,:),cuts(:)
  integer, allocatable :: labels(:)
  integer :: n,p,ncuts,input,output,i,operation,cut_min
  character(512) :: path
  character(32) :: name
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p,ncuts,cut_min,operation
  read(input,*) model%min_cluster_size,model%min_samples,model%max_cluster_size,model%alpha, &
    model%cluster_selection_epsilon,model%p,model%allow_single_cluster
  read(input,*) model%metric,model%cluster_selection_method,model%store_centers
  allocate(x(n,p),cuts(ncuts))
  do i=1,n
    read(input,*) x(i,:)
  end do
  read(input,*) cuts
  close(input)
  if(operation==2) then
    labels=model%fit_predict(x)
  else
    call model%fit(x)
  end if
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  call vector('labels',real(model%labels_,dp)); call vector('probabilities',model%probabilities_)
  call matrix('children',real(model%children_,dp)); call vector('distances',model%distances_)
  call vector('sizes',real(model%cluster_sizes_,dp)); call vector('finite',real(model%finite_indices_,dp))
  call vector('core',model%core_distances_); call matrix('mst_edges',real(model%mst_edges_,dp))
  call vector('mst_distances',model%mst_distances_)
  call vector('condensed_parents',real(model%condensed_parents_,dp))
  call vector('condensed_children',real(model%condensed_children_,dp))
  call vector('condensed_sizes',real(model%condensed_sizes_,dp)); call vector('condensed_lambdas',model%condensed_lambdas_)
  call vector('stabilities',model%stabilities_); call vector('selected',real(model%selected_clusters_,dp))
  call vector('meta',real([model%n_features_in_,model%n_clusters_,merge(1,0,model%has_infinite_edges_)],dp))
  if(allocated(model%centroids_)) call matrix('centroids',model%centroids_)
  if(allocated(model%medoids_)) call matrix('medoids',model%medoids_)
  if(allocated(labels)) call vector('fit_predict',real(labels,dp))
  do i=1,ncuts
    labels=model%dbscan_clustering(cuts(i),cut_min)
    write(name,'(a,i0)') 'cut',i; call vector(trim(name),real(labels,dp))
  end do
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
