program test_weighted_kmeans
  use sklearn_common,only:dp,require
  use sklearn_cluster,only:kmeans,k_means,kmeans_result
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  implicit none
  type(kmeans)::model
  type(kmeans_result)::result
  type(csr_matrix)::sparse
  real(dp)::x(4,2),initial(2,2),w(4),expected(2,2)
  real(dp),allocatable::distance(:,:)
  integer,allocatable::before(:),after(:)
  integer::n
  x(:,1)=[0._dp,2._dp,10._dp,12._dp];x(:,2)=[0._dp,0._dp,1._dp,1._dp]
  initial(1,:)=[0._dp,0._dp];initial(2,:)=[10._dp,1._dp];w=[1._dp,3._dp,1._dp,1._dp]
  expected(1,:)=[1.5_dp,0._dp];expected(2,:)=[11._dp,1._dp]
  call random_seed(size=n);allocate(before(n),after(n));call random_seed(get=before)
  model%n_clusters=2;call model%fit(x,initial,sample_weight=w)
  call random_seed(get=after);call require(all(before==after),'KMeans does not alter the process-global RNG')
  call require(all(model%cluster_centers_==expected),'weighted Lloyd means')
  call require(all(model%labels_==[0,0,1,1]).and.model%n_iter_==2,'labels and strict stopping count')
  call require(model%inertia_==5.and.model%score(x,w)==-5,'weighted inertia and score')
  sparse=csr_from_dense(x);result=k_means(sparse,2,sample_weight=w,init_centers=initial)
  call require(all(result%centers==expected).and.result%inertia==5,'CSR weighted Lloyd agrees with analytic means')
  call require(all(model%predict(sparse)==model%labels_),'CSR query labels')
  distance=model%transform(sparse);call require(all(shape(distance)==[4,2]),'CSR query distance shape')
  model%n_init=0;model%init='random';call model%fit(sparse,sample_weight=w)
  call require(model%n_init_effective_==10,'random auto initialization uses ten trials')
  model%init='k-means++';call model%fit(x,sample_weight=w)
  call require(model%n_init_effective_==1,'kmeans++ auto initialization uses one trial')
  print *,'Weighted dense/CSR KMeans, query methods and private seeded RNG passed'
end program
