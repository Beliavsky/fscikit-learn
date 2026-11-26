program test_neighbor_frozen_outputs
 use sklearn_common,only:dp,require
 use sklearn_neighbors,only:kneighbors_regressor
 use sklearn_frozen,only:frozen_estimator
 implicit none
 type(kneighbors_regressor),target::model
 type(frozen_estimator)::frozen
 real(dp)::x(6,1),y(6,2),query(2,1),truth(2,2)
 real(dp),allocatable::prediction(:,:)
 x(:,1)=[0._dp,1._dp,2._dp,3._dp,4._dp,5._dp]
 y(:,1)=2*x(:,1)+1;y(:,2)=-x(:,1);query(:,1)=[.5_dp,1.5_dp]
 truth(:,1)=[2._dp,4._dp];truth(:,2)=[-.5_dp,-1.5_dp]
 call frozen%set_params(model)
 call require(.not.frozen%is_fitted(),'unfitted neighbor wrapper')
 model%n_neighbors=2;call model%fit(x,y)
 call require(frozen%is_fitted(),'joint neighbor wrapper fitted detection')
 prediction=frozen%predict_matrix(query)
 call require(maxval(abs(prediction-truth))<1.e-12_dp,'frozen joint neighbor prediction')
 call require(frozen%score(query,truth,sample_weight=[1._dp,3._dp])>1-1.e-12_dp,'frozen weighted joint R2')
 call frozen%fit();call require(model%n_outputs_==2,'frozen fit preserves joint targets')
 call model%fit(x,y(:,:1))
 call require(frozen%score(query,truth(:,1),sample_weight=[1._dp,3._dp])>1-1.e-12_dp, &
  'frozen vector-truth score for column-target fit')
 call model%fit(x,y(:,1));prediction=frozen%predict_matrix(query)
 call require(size(prediction,2)==1.and.maxval(abs(prediction(:,1)-truth(:,1)))<1.e-12_dp, &
  'frozen follows scalar refit')
 print *,'Frozen joint neighbor fitted state, prediction, scoring and borrowed refits passed'
end program
