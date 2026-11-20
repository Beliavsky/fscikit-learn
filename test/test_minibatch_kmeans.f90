program test_minibatch_kmeans
  use sklearn, only: dp,mini_batch_kmeans
  use sklearn_common, only: require
  implicit none
  type(mini_batch_kmeans) :: model,copy
  real(dp) :: x(4,1),w(4),y(2,1)
  real(dp), allocatable :: distances(:,:),before(:,:),counts(:)
  integer, allocatable :: labels(:)
  x(:,1)=[-3._dp,-1._dp,1._dp,3._dp]; w=[1._dp,2._dp,1._dp,2._dp]
  model%n_clusters=2; model%reassignment_ratio=0
  model%initial_centers=reshape([-2._dp,2._dp],[2,1])
  call model%partial_fit(x,w)
  call require(maxval(abs(model%cluster_centers_(:,1)-[-5._dp/3,7._dp/3]))<1.e-12_dp,'minibatch first weighted means')
  call require(all(model%counts_==3).and.model%n_steps_==1,'minibatch first weighted counts/steps')
  y(:,1)=[-2._dp,2._dp]; call model%partial_fit(y,[2._dp,1._dp])
  call require(maxval(abs(model%cluster_centers_(:,1)-[-1.8_dp,2.25_dp]))<1.e-12_dp,'minibatch running weighted means')
  call require(all(model%counts_==[5._dp,4._dp]),'minibatch running counts')
  labels=model%predict(y); call require(all(labels==[0,1]),'minibatch zero-based prediction')
  distances=model%transform(y)
  call require(abs(distances(1,1)-.2_dp)<1.e-12_dp,'minibatch Euclidean transform')
  call require(abs(model%score(y,[2._dp,1._dp])+.1425_dp)<1.e-12_dp,'minibatch weighted score')
  before=model%cluster_centers_; counts=model%counts_; call model%partial_fit(x,0*w)
  call require(all(model%cluster_centers_==before).and.all(model%counts_==counts),'minibatch zero weights preserve state')
  call model%partial_fit(reshape([-2._dp],[1,1]))
  call require(model%n_steps_==4,'minibatch later batches may have fewer rows than clusters')
  model=mini_batch_kmeans(); model%n_clusters=1; model%batch_size=2; model%max_iter=4
  model%max_no_improvement=-1; model%reassignment_ratio=0; model%random_state=42
  call model%fit(x,w); copy=model; call model%fit(x,w)
  call require(all(copy%cluster_centers_==model%cluster_centers_),'minibatch fit resets seeded optimization')
  call require(model%n_steps_==8.and.model%n_iter_==4,'minibatch epoch and batch counters')
  copy=model; call copy%partial_fit(y); call model%partial_fit(y)
  call require(all(copy%cluster_centers_==model%cluster_centers_),'minibatch copied stream state')
  model=mini_batch_kmeans(); model%n_clusters=1; model%compute_labels=.false.; model%max_iter=1
  call model%fit(x)
  call require(.not.allocated(model%labels_).and..not.model%has_inertia_,'minibatch one-step inertia unavailable')
  model%max_iter=3; call model%fit(x)
  call require(model%has_inertia_.and..not.allocated(model%labels_),'minibatch smoothed inertia without labels')
  model=mini_batch_kmeans(); model%n_clusters=1; distances=model%fit_transform(x)
  call require(all(shape(distances)==[4,1]),'minibatch fit_transform shape')
  model=mini_batch_kmeans(); model%n_clusters=1; model%max_iter=1000000000
  model%max_no_improvement=0; call model%fit(0*x)
  call require(model%n_steps_==2.and.model%n_iter_==2,'minibatch wide iteration budget before early stopping')
  print *, 'MiniBatchKMeans tests passed'
end program
