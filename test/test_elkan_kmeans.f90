program test_elkan_kmeans
  use sklearn_common,only:dp,require
  use sklearn_cluster,only:kmeans,k_means,kmeans_result
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  implicit none
  type(kmeans)::model
  type(kmeans_result)::lloyd,elkan,single
  type(csr_matrix)::s
  real(dp)::x(4,2),initial(2,2),w(4),expected(2,2)
  integer::predicted(4)
  x(:,1)=[0._dp,2._dp,10._dp,12._dp];x(:,2)=[0._dp,0._dp,1._dp,1._dp]
  initial=x([1,3],:);w=[1._dp,3._dp,1._dp,1._dp]
  expected(1,:)=[1.5_dp,0._dp];expected(2,:)=[11._dp,1._dp]
  elkan=k_means(x,2,sample_weight=w,init_centers=initial,algorithm='elkan')
  call require(all(elkan%centers==expected).and.elkan%inertia==5,'analytic weighted Elkan solution')
  s=csr_from_dense(x);model%n_clusters=2;model%algorithm='elkan'
  call model%fit(s,initial,sample_weight=w)
  call require(all(model%cluster_centers_==expected).and.model%n_iter_==2,'CSR Elkan center updates and strict stopping')
  w=0
  elkan=k_means(x,2,sample_weight=w,init_centers=initial,algorithm='elkan')
  lloyd=k_means(x,2,sample_weight=w,init_centers=initial,algorithm='lloyd')
  call require(all(elkan%labels==[0,0,1,1]),'Elkan bounds retain tied labels after zero-weight center collapse')
  call require(all(lloyd%labels==0),'Lloyd assigns coincident centers to the first index')
  call require(all(elkan%centers==lloyd%centers).and.elkan%inertia==0,'zero-weight algorithms share centers and inertia')
  model%initial_centers=initial;call model%fit(x,sample_weight=w);predicted=model%predict(x)
  call require(all(model%labels_==[0,0,1,1]).and.all(predicted==0),'source fit/predict tie distinction is preserved')
  single=k_means(x,1,algorithm='elkan',random_state=7)
  lloyd=k_means(x,1,algorithm='lloyd',random_state=7)
  call require(all(single%centers==lloyd%centers).and.single%n_iter==lloyd%n_iter,'one-cluster fallback to Lloyd')
  print *,'Dense/CSR Elkan bounds, weighted means, source ties and fallback passed'
end program
