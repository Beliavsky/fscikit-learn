program test_radius_classification_outputs
 use sklearn_common,only:dp,require
 use sklearn_neighbors,only:radius_neighbors_classifier,neighbor_probability
 use,intrinsic::ieee_arithmetic
 use,intrinsic::ieee_exceptions
 implicit none
 type(radius_neighbors_classifier)::model
 type(neighbor_probability),allocatable::prob(:)
 real(dp)::x(6,1),q(2,1),d(6,6),qd(2,6),tiny_query(2,2)
 integer::y(6,3),i
 integer,allocatable::prediction(:,:),single(:)
 real(dp),allocatable::scalar_prob(:,:)
 logical::modes(size(ieee_all)),flags(size(ieee_all)),after_modes(size(ieee_all)),after_flags(size(ieee_all))
 x(:,1)=[0._dp,1._dp,2._dp,3._dp,4._dp,5._dp];q(:,1)=[.1_dp,4.9_dp]
 y(:,1)=[-3,-3,-3,7,7,7];y(:,2)=[0,0,2,2,9,9];y(:,3)=42
 model%radius=.6_dp;call model%fit(x,y)
 call ieee_get_flag(ieee_all,flags);call ieee_get_halting_mode(ieee_all,modes)
 prob=model%predict_proba_multioutput(q);prediction=model%predict_matrix(q)
 call ieee_get_flag(ieee_all,after_flags);call ieee_get_halting_mode(ieee_all,after_modes)
 call require(all(flags.eqv.after_flags).and.all(modes.eqv.after_modes),'radius classifier IEEE state')
 call require(all(prediction(1,:)==[-3,0,42]).and.all(prediction(2,:)==[7,9,42]),'joint radius predictions')
 call require(all(prob(2)%classes==[0,2,9]),'radius class order')
 model%outlier_mode='most_frequent';model%radius=0;call model%fit(x,y)
 prediction=model%predict_matrix();call require(all(prediction(:,1)==-3),'most frequent tied class chooses smallest')
 call require(all(prediction(:,2)==0).and.all(prediction(:,3)==42),'per-output most frequent outliers')
 model%outlier_mode='label';model%outlier_labels=[-99,2,42];call model%fit(x,y)
 prob=model%predict_proba_multioutput();prediction=model%predict_matrix()
 call require(all(prob(1)%values==0).and.all(prediction(:,1)==-99),'unknown outlier has zero probabilities')
 call require(all(prob(2)%values(:,2)==1).and.all(prediction(:,2)==2),'known output outlier probability')
 prob=model%predict_proba_multioutput(x(:1,:));call require(prob(1)%values(1,1)==1,'explicit zero-radius query')
 do i=1,6
  d(:,i)=abs(x(:,1)-x(i,1));qd(:,i)=abs(q(:,1)-x(i,1))
 end do
 model%metric='precomputed';model%radius=ieee_value(0._dp,ieee_positive_inf)
 model%weights='distance';call model%fit(d,y);prediction=model%predict_matrix(qd)
 call require(all(prediction(1,:)==[-3,0,42]).and.all(prediction(2,:)==[7,9,42]),'precomputed radius weighting')
 model%metric='minkowski';model%outlier_mode='error';call model%fit(x,reshape([0,0,0,1,1,1],[6,1]))
 single=model%predict(q);scalar_prob=model%predict_proba(q)
 call require(.not.model%outputs_2d_.and.all(single==[0,1]),'column fit restores scalar API')
 call require(model%score(q,[0,1],sample_weight=[1._dp,3._dp])==1,'weighted radius accuracy')
 call model%fit(x,reshape([0,0,0,1,1,1,1,1,1,0,0,0],[6,2]))
 call require(model%score(q,reshape([0,1,1,0],[2,2]),sample_weight=[1._dp,3._dp])==1,'weighted radius subset accuracy')
 model%metric='precomputed';model%radius=1;model%outlier_mode='label';model%outlier_label=-99
 deallocate(model%outlier_labels)
 call model%fit(reshape([0._dp,1._dp,1._dp,0._dp],[2,2]),[-3,7])
 tiny_query=1;tiny_query(1,1)=1.e-320_dp;tiny_query(2,2)=1.e-320_dp
 call ieee_get_flag(ieee_all,flags);call ieee_get_halting_mode(ieee_all,modes)
 scalar_prob=model%predict_proba(tiny_query);single=model%predict(tiny_query)
 call ieee_get_flag(ieee_all,after_flags);call ieee_get_halting_mode(ieee_all,after_modes)
 call require(all(flags.eqv.after_flags).and.all(modes.eqv.after_modes),'radius overflow preserves IEEE state')
 call require(ieee_is_nan(scalar_prob(1,1)).and.ieee_is_nan(scalar_prob(2,2)),'radius reciprocal overflow retained')
 call require(all(single==[-3,7]),'radius first-NaN class prediction')
 print *,'Joint radius labels, outlier modes, probabilities, scoring, precomputed data, IEEE and refits passed'
end program
