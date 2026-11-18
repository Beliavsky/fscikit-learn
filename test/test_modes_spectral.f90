program test_modes_spectral
  use sklearn
  use sklearn_common, only: require
  implicit none
  type(mean_shift) :: ms
  type(affinity_propagation) :: ap
  type(spectral_clustering) :: sc
  type(spectral_embedding) :: se
  real(dp) :: x(60,2),trivial(6,2),graph(6,6),degree(60),gram(2,2),eye(2,2)
  real(dp), allocatable :: z(:,:),a(:,:)
  integer :: truth(60),i,c
  do i=1,60
    c=(i-1)/20; truth(i)=c
    x(i,:)=[3._dp*(c-1),merge(3._dp,-3._dp,c==1)]+0.06_dp*[sin(real(i,dp)),cos(1.7_dp*i)]
  end do
  ms%bandwidth=0.5_dp; call ms%fit(x)
  call require(size(ms%cluster_centers_,1)==3.and.adjusted_rand_score(truth,ms%labels_)==1,'MeanShift modes')
  call require(ms%converged_,'MeanShift converged')
  ms%bandwidth=-1; call ms%fit(x)
  call require(adjusted_rand_score(truth,ms%labels_)==1.and.ms%bandwidth_>0,'MeanShift automatic bandwidth')
  ms%bandwidth=0.5_dp; ms%cluster_all=.false.; call ms%fit(x,seeds=x([1,21],:))
  call require(all(ms%labels_(41:)==-1).and.all(ms%predict(x)/=-1),'MeanShift orphan labels versus prediction')
  ms%max_iter=0; call ms%fit(x)
  call require(ms%n_iter_==0.and..not.ms%converged_,'MeanShift capped updates')
  ap%preference=-10; ap%damping=0.7_dp; ap%max_iter=500; call ap%fit(x)
  call require(ap%converged_.and.adjusted_rand_score(truth,ap%labels_)==1,'AffinityPropagation clusters')
  call require(all(ap%predict(x)==ap%labels_),'AffinityPropagation nearest-exemplar prediction')
  trivial=0; ms%bandwidth=-1; call ms%fit(trivial)
  call require(ms%bandwidth_==0.and.size(ms%cluster_centers_,1)==1,'MeanShift zero automatic bandwidth')
  ap%preference=huge(1._dp); call ap%fit(trivial)
  call require(size(ap%cluster_centers_indices_)==1.and.ap%n_iter_==0,'equal affinities one cluster')
  ap%preference=1; call ap%fit(trivial)
  call require(size(ap%cluster_centers_indices_)==6.and.ap%n_iter_==0,'equal affinities separate clusters')
  ap%preference=-1.e12_dp; ap%max_iter=1; call ap%fit(x)
  call require(.not.ap%converged_.and.all(ap%labels_==-1).and.size(ap%cluster_centers_indices_)==0, &
    'AffinityPropagation empty-exemplar cap')
  call require(all(ap%predict(x)==-1),'empty-exemplar prediction')
  sc%n_clusters=3; sc%gamma=0.3_dp; sc%assign_labels='cluster_qr'; call sc%fit(x)
  call require(adjusted_rand_score(truth,sc%labels_)==1,'spectral QR clustering')
  sc%assign_labels='kmeans'; call sc%fit(x)
  call require(adjusted_rand_score(truth,sc%labels_)==1,'spectral kmeans clustering')
  se%affinity='rbf'; se%gamma=0.3_dp; z=se%fit_transform(x); a=se%affinity_matrix_
  degree=sum(a,dim=2)
  do i=1,60
    degree(i)=degree(i)-a(i,i)
  end do
  gram=matmul(transpose(z),z*spread(degree,2,2)); eye=0; eye(1,1)=1; eye(2,2)=1
  call require(maxval(abs(gram-eye))<1.e-10_dp,'spectral degree-weighted orthonormality')
  call require(maxval(abs(sum(z*spread(degree,2,2),dim=1)))<1.e-7_dp,'spectral nontrivial eigenvectors')
  graph=0
  do i=1,3
    graph(2*i-1:2*i,2*i-1:2*i)=1
  end do
  se%affinity='precomputed'; call se%fit(graph)
  call require(se%n_connected_components_==3,'spectral disconnected components')
  graph=0.25_dp*graph; call se%fit(graph); z=se%embedding_
  do i=1,6
    graph(i,i)=1.e100_dp
  end do
  call se%fit(graph); a=matmul(z,transpose(z))
  call require(maxval(abs(matmul(se%embedding_,transpose(se%embedding_))-a))<1.e-12_dp, &
    'spectral normalization ignores large self loops without degree cancellation')
  print *, 'mode and spectral clustering: PASS'
end program
