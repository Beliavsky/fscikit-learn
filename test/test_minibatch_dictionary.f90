program test_minibatch_dictionary
  use sklearn_dictionary, only: mini_batch_dictionary_learning,mini_batch_sparse_pca
  use sklearn_randomized_svd_core, only: randomized_svd
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_common, only: dp,require,dense_svd
  implicit none
  type(mini_batch_dictionary_learning) :: model,copy,padded
  type(mini_batch_sparse_pca) :: projection
  type(legacy_random_state) :: rng
  real(dp) :: x(2,2),data(4,3),sv(3),left(4,3),right(3,3)
  real(dp), allocatable :: codes(:,:),z(:,:),reconstructed(:,:),s(:),u(:,:),vt(:,:)
  x=0; x(1,1)=1; x(2,2)=1
  model%n_components=2; model%fit_algorithm='cd'; model%alpha=.2_dp; model%dict_init=x
  model%transform_algorithm='lasso_cd'; model%transform_alpha=.2_dp; model%max_iter=0
  model%batch_size=999; model%tol=0; model%max_no_improvement=-1
  call model%partial_fit(x)
  call require(maxval(abs(model%components_-x))<1.e-14_dp,'MiniBatchDictionaryLearning atom projection')
  call require(maxval(abs(model%a_-.32_dp*x))<1.e-14_dp,'MiniBatchDictionaryLearning analytic first statistic A')
  call require(maxval(abs(model%b_-.4_dp*x))<1.e-14_dp,'MiniBatchDictionaryLearning analytic first statistic B')
  codes=model%transform(x)
  call require(maxval(abs(codes-.8_dp*x))<1.e-14_dp,'MiniBatchDictionaryLearning analytic soft threshold')
  copy=model; call model%partial_fit(2*x); call copy%partial_fit(2*x)
  call require(maxval(abs(model%a_-1.812_dp*x))<1.e-13_dp,'MiniBatchDictionaryLearning forgetting recurrence A')
  call require(maxval(abs(model%b_-2.04_dp*x))<1.e-13_dp,'MiniBatchDictionaryLearning forgetting recurrence B')
  call require(all(model%components_==copy%components_),'MiniBatchDictionaryLearning copied stream')
  call require(model%n_steps_==2.and.model%n_iter_==0,'MiniBatchDictionaryLearning partial counters')
  model%shuffle=.false.; model%max_iter=1000000000; model%batch_size=2; model%tol=.001_dp
  call model%fit(x)
  call require(model%n_steps_==2.and.model%n_iter_==2,'MiniBatchDictionaryLearning huge budget early stop')
  model%tol=0; model%max_no_improvement=0; call model%fit(x)
  call require(model%n_steps_==2,'MiniBatchDictionaryLearning zero patience stops after warmup')
  padded%n_components=3; padded%max_iter=0; padded%dict_init=x(:1,:)
  call padded%fit(x)
  call require(all(shape(padded%components_)==[3,2]).and.all(padded%components_(2:,:)==0), &
    'MiniBatchDictionaryLearning initializer padding')
  call require(padded%n_steps_==0.and.all(padded%a_==0),'MiniBatchDictionaryLearning initialization only')
  data=reshape([1._dp,2._dp,3._dp,4._dp,2._dp,-1._dp,0._dp,3._dp,1._dp,4._dp,2._dp,-2._dp],[4,3])
  call rng%seed(42); call randomized_svd(data,2,rng,s,u,vt); call dense_svd(data,sv,left,right)
  call require(maxval(abs(s-sv(:2)))<1.e-11_dp,'Randomized SVD exact full sampled range')
  call require(maxval(abs(matmul(transpose(u),u)-x))<1.e-12_dp,'Randomized SVD left orthogonality')
  call require(maxval(abs(matmul(vt,transpose(vt))-x))<1.e-12_dp,'Randomized SVD right orthogonality')
  projection%n_components=2; projection%max_iter=3; projection%tol=0; projection%max_no_improvement=-1
  z=projection%fit_transform(0*data+1)
  reconstructed=projection%inverse_transform(z)
  call require(all(projection%components_==0).and.all(z==0),'MiniBatchSparsePCA constant features')
  call require(all(reconstructed==1),'MiniBatchSparsePCA mean reconstruction')
  call require(all(shape(projection%components_)==[2,3]).and.projection%n_iter_==3,'MiniBatchSparsePCA state')
  projection%ridge_alpha=0; z=projection%transform(0*data+1)
  call require(all(z==0),'MiniBatchSparsePCA singular zero-ridge projection')
  print *, 'Mini-batch dictionary learning and SparsePCA tests passed'
end program
