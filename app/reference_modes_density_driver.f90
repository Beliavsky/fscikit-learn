program reference_modes_density_driver
  use sklearn
  implicit none
  type(mean_shift) :: ms
  type(affinity_propagation) :: ap
  type(spectral_embedding) :: se
  type(spectral_clustering) :: sc
  type(kernel_density) :: kde
  real(dp), allocatable :: x(:,:),continuous(:,:),query(:,:),weighted_query(:,:),g(:,:),weights(:)
  integer :: n,p,i,k,input,output
  character(1024) :: path
  character(16) :: kernels(5)=[character(16)::'gaussian','tophat','epanechnikov','exponential','linear']
  character(20) :: affinities(3)=[character(20)::'rbf','nearest_neighbors','precomputed']
  character(32) :: prefix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p; allocate(x(n,p),continuous(n,p),weights(n))
  do i=1,n
    read(input,*) x(i,:),continuous(i,:)
    weights(i)=mod(i-1,4)
  end do
  close(input); query=x+0.017_dp
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  ms%bandwidth=0.8_dp; call ms%fit(x); call mode_results('mean')
  call vector('mean_predict',real(ms%predict(query),dp))
  ms%bandwidth=-1; call ms%fit(x); call mode_results('mean_auto')
  call vector('mean_auto_bandwidth',[ms%bandwidth_])
  call vector('bandwidth_quantiles',[estimate_bandwidth(x,0.2_dp),estimate_bandwidth(x,0.3_dp), &
    estimate_bandwidth(x,0.5_dp)])
  ms%bandwidth=0.8_dp; ms%cluster_all=.false.; call ms%fit(x,seeds=x([1,n/3+1],:))
  call matrix('mean_orphan_centers',ms%cluster_centers_); call vector('mean_orphan_labels',real(ms%labels_,dp))
  call vector('mean_orphan_predict',real(ms%predict(query),dp))
  ms%max_iter=0; ms%cluster_all=.true.; call ms%fit(x); call mode_results('mean_capped')
  ap%damping=0.7_dp; ap%max_iter=500; ap%random_state=21; call ap%fit(x)
  call vector('ap_indices',real(ap%cluster_centers_indices_,dp)); call matrix('ap_centers',ap%cluster_centers_)
  call vector('ap_labels',real(ap%labels_,dp)); call vector('ap_predict',real(ap%predict(query),dp))
  call matrix('ap_affinity',ap%affinity_matrix_)
  call vector('ap_status',[real(ap%n_iter_,dp),merge(1._dp,0._dp,ap%converged_)])
  ap%affinity='precomputed'; ap%preference=-2.5_dp; g=-pairwise_distances(x,x,'sqeuclidean'); call ap%fit(g)
  call vector('ap_pre_indices',real(ap%cluster_centers_indices_,dp)); call vector('ap_pre_labels',real(ap%labels_,dp))
  call matrix('ap_pre_affinity',ap%affinity_matrix_)
  call vector('ap_pre_status',[real(ap%n_iter_,dp),merge(1._dp,0._dp,ap%converged_)])
  ap%affinity='euclidean'; ap%preference=-1.e8_dp; ap%max_iter=1; call ap%fit(x)
  call vector('ap_capped_labels',real(ap%labels_,dp)); call vector('ap_capped_predict',real(ap%predict(query),dp))
  call vector('ap_capped_status',[real(ap%n_iter_,dp),merge(1._dp,0._dp,ap%converged_)])
  se%gamma=0.4_dp; se%n_neighbors=9; se%n_components=3
  do i=1,3
    se%affinity=affinities(i)
    if(i==3) then
      call se%fit(rbf_kernel(continuous,continuous,0.4_dp))
    else
      call se%fit(continuous)
    end if
    prefix='embedding_'//trim(affinities(i))
    call matrix(trim(prefix)//'_affinity',se%affinity_matrix_)
    call matrix(trim(prefix)//'_gram',matmul(se%embedding_,transpose(se%embedding_)))
    call vector(trim(prefix)//'_components',[real(se%n_connected_components_,dp)])
  end do
  sc%n_clusters=3; sc%gamma=0.3_dp; sc%random_state=21
  do i=1,4
    sc%affinity='rbf'; sc%assign_labels='cluster_qr'
    if(i==2) sc%assign_labels='kmeans'
    if(i==3) sc%affinity='nearest_neighbors'
    if(i==4) sc%affinity='precomputed'
    if(i==4) then
      call sc%fit(rbf_kernel(x,x,0.3_dp))
    else
      call sc%fit(x)
    end if
    write(prefix,'(a,i0)') 'spectral_',i
    call vector(trim(prefix)//'_labels',real(sc%labels_,dp)); call matrix(trim(prefix)//'_affinity',sc%affinity_matrix_)
    call matrix(trim(prefix)//'_gram',matmul(sc%embedding_,transpose(sc%embedding_)))
    call vector(trim(prefix)//'_components',[real(sc%n_connected_components_,dp)])
  end do
  query=continuous+0.017_dp; kde%bandwidth=0.9_dp
  weighted_query=query
  do i=1,n
    if(weights(i)==0) weighted_query(i,:)=continuous(mod(i,n)+1,:)+0.017_dp
  end do
  do k=1,5
    kde%kernel=kernels(k); call kde%fit(continuous)
    call vector('density_'//trim(kernels(k)),kde%score_samples(query))
    call vector('density_score_'//trim(kernels(k)),[kde%score(query)])
    call kde%fit(continuous,sample_weight=weights)
    call vector('weighted_'//trim(kernels(k)),kde%score_samples(weighted_query))
    call vector('weighted_score_'//trim(kernels(k)),[kde%score(weighted_query)])
  end do
  kde%kernel='gaussian'
  kde%bandwidth_rule='scott'; call kde%fit(continuous)
  call vector('density_scott',kde%score_samples(query)); call vector('bandwidth_scott',[kde%bandwidth_])
  kde%bandwidth_rule='silverman'; call kde%fit(continuous)
  call vector('density_silverman',kde%score_samples(query)); call vector('bandwidth_silverman',[kde%bandwidth_])
  close(output)
contains
  subroutine mode_results(prefix)
    character(*), intent(in) :: prefix
    call matrix(prefix//'_centers',ms%cluster_centers_)
    call vector(prefix//'_labels',real(ms%labels_,dp)); call vector(prefix//'_iterations',[real(ms%n_iter_,dp)])
  end subroutine
  subroutine vector(name,values)
    character(*), intent(in) :: name
    real(dp), intent(in) :: values(:)
    write(output,'(a,*(1x,es25.16e3))') name,values
  end subroutine
  subroutine matrix(name,values)
    character(*), intent(in) :: name
    real(dp), intent(in) :: values(:,:)
    call vector(name,reshape(values,[size(values)]))
  end subroutine
end program
