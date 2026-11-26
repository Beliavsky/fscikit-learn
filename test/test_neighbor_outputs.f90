program test_neighbor_outputs
 use sklearn_common,only:dp,require
 use sklearn_neighbors,only:kneighbors_regressor,radius_neighbors_regressor
 use,intrinsic::ieee_arithmetic
 use,intrinsic::ieee_exceptions
 implicit none
 type(kneighbors_regressor)::knn
 type(radius_neighbors_regressor)::rad
 real(dp)::x(6,1),targets(6,2),q(2,1),truth(2,2),d(6,6),qd(2,6)
 real(dp),allocatable::prediction(:,:),single(:)
 integer::i
 logical::modes(size(ieee_all)),flags(size(ieee_all)),after_modes(size(ieee_all)),after_flags(size(ieee_all))
 x(:,1)=[0._dp,1._dp,2._dp,3._dp,4._dp,5._dp]
 targets(:,1)=2*x(:,1)+1;targets(:,2)=-x(:,1);q(:,1)=[.5_dp,1.5_dp]
 truth(:,1)=[2._dp,4._dp];truth(:,2)=[-.5_dp,-1.5_dp]
 knn%n_neighbors=2;call knn%fit(x,targets)
 call require(knn%n_outputs_==2.and.knn%n_samples_fit_==6.and.knn%matrix_target_,'neighbor target metadata')
 call ieee_set_flag(ieee_all,.false.);call ieee_get_flag(ieee_all,flags);call ieee_get_halting_mode(ieee_all,modes)
 prediction=knn%predict_matrix(q)
 call ieee_get_flag(ieee_all,after_flags);call ieee_get_halting_mode(ieee_all,after_modes)
 call require(all(flags.eqv.after_flags).and.all(modes.eqv.after_modes),'neighbor prediction preserves IEEE state')
 call require(maxval(abs(prediction-truth))<1.e-12_dp,'joint neighbor targets')
 call require(knn%score(q,truth,sample_weight=[1._dp,3._dp])>1-1.e-12_dp,'weighted multioutput neighbor R2')
 prediction=knn%predict_matrix()
 call require(maxval(abs(prediction(1,:)-[4._dp,-1.5_dp]))<1.e-12_dp,'training predictions exclude the fitted sample')
 knn%weights='distance';prediction=knn%predict_matrix(x(:1,:))
 call require(all(prediction(1,:)==targets(1,:)),'zero-distance neighbors dominate')
 rad%radius=.6_dp;call rad%fit(x,targets);prediction=rad%predict_matrix(q)
 call require(maxval(abs(prediction-truth))<1.e-12_dp,'joint radius targets')
 rad%radius=0;prediction=rad%predict_matrix()
 call require(all(ieee_is_nan(prediction)),'zero-radius training predictions exclude self')
 prediction=rad%predict_matrix(x(:1,:));call require(all(prediction(1,:)==targets(1,:)),'zero-radius explicit query')
 do i=1,6
  d(:,i)=abs(x(:,1)-x(i,1));qd(:,i)=abs(q(:,1)-x(i,1))
 end do
 knn%metric='precomputed';knn%weights='uniform';call knn%fit(d,targets)
 prediction=knn%predict_matrix(qd);call require(maxval(abs(prediction-truth))<1.e-12_dp,'precomputed multioutput regression')
 rad%metric='precomputed';rad%radius=ieee_value(0._dp,ieee_positive_inf);call rad%fit(d,targets)
 prediction=rad%predict_matrix(qd)
 call require(maxval(abs(prediction(1,:)-[6._dp,-2.5_dp]))<1.e-12_dp,'infinite-radius regression')
 knn%metric='minkowski';call knn%fit(x,targets(:,1));single=knn%predict(q)
 call require(allocated(knn%y_).and..not.knn%matrix_target_.and.all(single==truth(:,1)),'scalar refit resets target state')
 prediction=knn%predict_matrix(q)
 call require(size(prediction,2)==1.and.all(prediction(:,1)==single),'matrix helper after scalar fit')
 print *,'Joint neighbor targets, training exclusion, weighting, radius bounds, precomputed data and refits passed'
end program
