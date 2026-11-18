program reference_graph_driver
  use sklearn
  use sklearn_common, only: unique_labels
  implicit none
  type(radius_neighbors_classifier) :: rc
  type(radius_neighbors_regressor) :: rr
  type(nearest_centroid) :: nc
  type(local_outlier_factor) :: lof
  type(kernel_pca) :: kp,pre
  type(isomap) :: iso
  type(label_propagation) :: lp
  type(label_spreading) :: ls
  real(dp), allocatable :: x(:,:),target(:),query(:,:),g(:,:),z(:,:)
  integer, allocatable :: labels(:),semi(:),classes(:)
  integer :: n,p,i,j,input,output
  character(1024) :: path
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p; allocate(x(n,p),target(n),labels(n))
  do i=1,n
    read(input,*) x(i,:),target(i),labels(i)
  end do
  close(input); query=x+0.013_dp
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  rc%radius=1.3_dp; call rc%fit(x,labels)
  call matrix('radius_class_uniform',rc%predict_proba(query))
  rc%weights='distance'; call matrix('radius_class_distance',rc%predict_proba(query))
  rr%radius=1.3_dp; call rr%fit(x,target); call vector('radius_reg_uniform',rr%predict(query))
  rr%weights='distance'; call vector('radius_reg_distance',rr%predict(query))
  call nc%fit(x,labels); call matrix('centroid_mean',nc%centroids_)
  call vector('centroid_predict',real(nc%predict(query),dp))
  call vector('centroid_std',nc%within_class_std_dev_)
  call matrix('centroid_deviations',nc%deviations_)
  nc%shrink_threshold=0.3_dp; call nc%fit(x,labels); call matrix('centroid_shrink',nc%centroids_)
  nc%metric='manhattan'; nc%shrink_threshold=0; call nc%fit(x,labels)
  call matrix('centroid_median',nc%centroids_); call vector('centroid_manhattan',real(nc%predict(query),dp))
  lof%n_neighbors=7; call lof%fit(x); call vector('lof_train',lof%negative_outlier_factor_)
  lof%contamination=0.15_dp; call lof%fit(x); call vector('lof_offset',[lof%offset_])
  lof%novelty=.true.; call vector('lof_query',lof%score_samples(query))
  call vector('lof_decision',lof%decision_function(query))
  call vector('lof_predict',real(lof%predict(query),dp))
  kp%n_components=3; kp%kernel='rbf'; kp%gamma=0.2_dp
  z=kp%fit_transform(x); call matrix('kpca_rbf_fit',z)
  call matrix('kpca_rbf_query',kp%transform(query)); call vector('kpca_rbf_values',kp%eigenvalues_)
  pre%kernel='precomputed'; pre%n_components=3; g=rbf_kernel(x,x,0.2_dp)
  z=pre%fit_transform(g); call matrix('kpca_precomputed',pre%transform(rbf_kernel(query,x,0.2_dp)))
  kp%kernel='linear'; call kp%fit(x); call matrix('kpca_linear',kp%transform(query))
  kp%kernel='poly'; kp%degree=2; call kp%fit(x); call matrix('kpca_poly',kp%transform(query))
  kp%kernel='cosine'; call kp%fit(x); call matrix('kpca_cosine',kp%transform(query))
  iso%n_neighbors=8; iso%n_components=2; call iso%fit(x)
  call matrix('isomap_distances',iso%dist_matrix_); call matrix('isomap_fit',iso%embedding_)
  call matrix('isomap_query',iso%transform(query)); call vector('isomap_error',[iso%reconstruction_error()])
  iso%n_neighbors=1; call iso%fit(x); call matrix('isomap_completed_distances',iso%dist_matrix_)
  semi=labels; classes=unique_labels(labels)
  do i=1,n
    if(mod(i-1,4)/=0) semi(i)=-1
  end do
  do i=1,size(classes)
    do j=1,n
      if(labels(j)==classes(i)) exit
    end do
    semi(j)=classes(i)
  end do
  lp%gamma=0.2_dp; lp%max_iter=3000; lp%tol=1.e-10_dp; call lp%fit(x,semi)
  call matrix('propagation_rbf_train',lp%label_distributions_)
  call matrix('propagation_rbf_query',lp%predict_proba(query))
  call vector('propagation_rbf_status',[real(lp%n_iter_,dp),merge(1._dp,0._dp,lp%converged_)])
  ls%gamma=0.2_dp; ls%max_iter=3000; ls%tol=1.e-10_dp; call ls%fit(x,semi)
  call matrix('spreading_rbf_train',ls%label_distributions_)
  call matrix('spreading_rbf_query',ls%predict_proba(query))
  call vector('spreading_rbf_status',[real(ls%n_iter_,dp),merge(1._dp,0._dp,ls%converged_)])
  lp%kernel='knn'; lp%n_neighbors=7; call lp%fit(x,semi)
  call matrix('propagation_knn_train',lp%label_distributions_)
  call matrix('propagation_knn_query',lp%predict_proba(query))
  call vector('propagation_knn_status',[real(lp%n_iter_,dp),merge(1._dp,0._dp,lp%converged_)])
  ls%kernel='knn'; ls%n_neighbors=7; call ls%fit(x,semi)
  call matrix('spreading_knn_train',ls%label_distributions_)
  call matrix('spreading_knn_query',ls%predict_proba(query))
  call vector('spreading_knn_status',[real(ls%n_iter_,dp),merge(1._dp,0._dp,ls%converged_)])
  close(output)
contains
  subroutine vector(name,values)
    character(*), intent(in) :: name
    real(dp), intent(in) :: values(:)
    write(output,'(a,1x,*(es25.16e3,1x))') name,values
  end subroutine
  subroutine matrix(name,values)
    character(*), intent(in) :: name
    real(dp), intent(in) :: values(:,:)
    call vector(name,reshape(values,[size(values)]))
  end subroutine
end program
