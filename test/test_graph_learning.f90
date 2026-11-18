program test_graph_learning
  use sklearn_common
  use sklearn_metrics, only: rbf_kernel,linear_kernel
  use sklearn_kernel_pca
  use sklearn_manifold
  use sklearn_semi_supervised
  implicit none
  type(kernel_pca) :: kp,pre
  type(isomap) :: iso
  type(label_propagation) :: lp
  type(label_spreading) :: ls
  real(dp) :: x(12,2),query(2,2)
  real(dp), allocatable :: z(:,:),other(:,:),g(:,:),prob(:,:)
  integer :: i,labels(12),full(12)
  do i=1,12
    x(i,:)=[real(i,dp),sin(real(i,dp))]
    labels(i)=-1; full(i)=merge(3,8,i<=6)
  end do
  kp%kernel='rbf'; kp%gamma=0.1_dp; kp%n_components=3; z=kp%fit_transform(x)
  call require(maxval(abs(kp%transform(x)-z))<1.e-10_dp,'kernel PCA fit/transform consistency')
  g=rbf_kernel(x,x,0.1_dp); pre%kernel='precomputed'; pre%n_components=3; other=pre%fit_transform(g)
  call require(maxval(abs(other-z))<1.e-10_dp,'precomputed kernel equivalence')
  x(:,2)=0; kp%kernel='linear'; kp%n_components=0; z=kp%fit_transform(x)
  call require(size(z,2)==1,'kernel PCA drops null-space with all components')
  x=0; z=kp%fit_transform(x); call require(size(z,2)==0,'zero kernel PCA')
  do i=1,12
    x(i,:)=[real(i,dp),sin(real(i,dp))]
  end do
  iso%n_neighbors=3; iso%n_components=2; z=iso%fit_transform(x)
  call require(iso%n_connected_components_==1,'Isomap connected graph')
  call require(maxval(abs(iso%dist_matrix_-transpose(iso%dist_matrix_)))<1.e-12_dp,'Isomap distances symmetric')
  call require(iso%reconstruction_error()>=0,'Isomap reconstruction error')
  other=iso%transform(x); call require(maxval(abs(other-z))<1.e-9_dp,'Isomap training transform')
  iso%n_neighbors=1; call iso%fit(x)
  call require(all(iso%dist_matrix_<huge(1._dp)/4),'Isomap component completion')
  labels(1)=3; labels(12)=8; lp%gamma=0.1_dp; lp%tol=1.e-10_dp; call lp%fit(x,labels)
  call require(lp%converged_,'LabelPropagation convergence')
  call require(lp%transduction_(1)==3.and.lp%transduction_(12)==8,'LabelPropagation clamps labels')
  call require(all(abs(sum(lp%label_distributions_,dim=2)-1)<1.e-12_dp),'diffusion normalization')
  query=x([1,12],:); call require(all(lp%predict(query)==[3,8]),'label inference')
  ls%gamma=0.1_dp; ls%tol=1.e-10_dp; ls%max_iter=500; call ls%fit(x,labels)
  call require(ls%converged_,'LabelSpreading convergence')
  lp%kernel='knn'; lp%n_neighbors=4; call lp%fit(x,labels)
  ls%kernel='knn'; ls%n_neighbors=4; call ls%fit(x,labels)
  prob=ls%predict_proba(query)
  call require(all(abs(sum(prob,dim=2)-1)<1.e-12_dp),'KNN diffusion inference')
  print *, 'kernel PCA, Isomap and label diffusion: PASS'
end program
