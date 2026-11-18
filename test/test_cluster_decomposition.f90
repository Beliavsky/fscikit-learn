program test_cluster_decomposition
  use sklearn_common
  use sklearn_metrics, only: adjusted_rand_score
  use sklearn_cluster
  use sklearn_decomposition
  implicit none
  type(kmeans) :: km
  type(dbscan) :: db
  type(agglomerative_clustering) :: ag
  type(pca) :: pc
  type(truncated_svd) :: svd
  type(nmf) :: nm
  real(dp) :: x(6,2),positive(6,2),init(2,2)
  real(dp), allocatable :: z(:,:),reconstructed(:,:)
  integer :: y(6)
  x(:,1)=[-3._dp,-2._dp,-1._dp,7._dp,8._dp,9._dp]; x(:,2)=2*x(:,1)+3
  y=[0,0,0,1,1,1]; init=x([1,4],:)
  km%n_clusters=2; call km%fit(x,init)
  call require(adjusted_rand_score(y,km%labels_)==1,'kmeans partition')
  call require(abs(km%inertia_-20)<1.e-10_dp,'kmeans inertia')
  call require(all(km%predict(x)==km%labels_),'kmeans predict')
  db%eps=2.3_dp; db%min_samples=2; call db%fit(x)
  call require(adjusted_rand_score(y,db%labels_)==1,'DBSCAN partition')
  call ag%fit(x); call require(adjusted_rand_score(y,ag%labels_)==1,'Ward partition')
  ag%linkage='average'; call ag%fit(x)
  call require(adjusted_rand_score(y,ag%labels_)==1,'average linkage partition')
  pc%n_components=1; z=pc%fit_transform(x); reconstructed=pc%inverse_transform(z)
  call require(maxval(abs(reconstructed-x))<1.e-10_dp,'PCA rank-one reconstruction')
  call require(abs(sum(pc%explained_variance_ratio_)-1)<1.e-12_dp,'PCA explained variance')
  pc%whiten=.true.; z=pc%transform(x)
  call require(abs(sum(z**2)/5-1)<1.e-12_dp,'PCA whitening')
  svd%n_components=2; z=svd%fit_transform(x)
  call require(maxval(abs(svd%inverse_transform(z)-x))<1.e-10_dp,'SVD reconstruction')
  positive(:,1)=[1._dp,2._dp,3._dp,4._dp,5._dp,6._dp]; positive(:,2)=2*positive(:,1)
  nm%n_components=1; z=nm%fit_transform(positive)
  call require(nm%reconstruction_err_<1.e-8_dp,'NMF rank-one reconstruction')
  call require(all(z>=0).and.all(nm%components_>=0),'NMF nonnegativity')
  z=nm%transform(positive)
  call require(maxval(abs(nm%inverse_transform(z)-positive))<1.e-8_dp,'NMF transform')
  print *, 'clustering and decomposition: PASS'
end program
