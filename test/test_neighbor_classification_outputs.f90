program test_neighbor_classification_outputs
 use sklearn_common,only:dp,require
 use sklearn_neighbors,only:kneighbors_classifier,neighbor_probability
 use,intrinsic::ieee_exceptions
 implicit none
 type(kneighbors_classifier)::model
 type(neighbor_probability),allocatable::prob(:)
 real(dp)::x(6,1),q(2,1),d(6,6),qd(2,6)
 integer::y(6,3),i
 integer,allocatable::prediction(:,:),single(:)
 real(dp),allocatable::scalar_prob(:,:)
 logical::modes(size(ieee_all)),flags(size(ieee_all)),after_modes(size(ieee_all)),after_flags(size(ieee_all))
 x(:,1)=[0._dp,1._dp,2._dp,3._dp,4._dp,5._dp];q(:,1)=[.1_dp,4.9_dp]
 y(:,1)=[-3,-3,-3,7,7,7];y(:,2)=[0,0,2,2,9,9];y(:,3)=42
 model%n_neighbors=2;call model%fit(x,y)
 call require(model%outputs_2d_.and.model%n_outputs_==3,'joint classifier metadata')
 call require(.not.allocated(model%classes_).and..not.allocated(model%y_),'scalar classifier state cleared')
 call ieee_get_flag(ieee_all,flags);call ieee_get_halting_mode(ieee_all,modes)
 prob=model%predict_proba_multioutput(q);prediction=model%predict_matrix(q)
 call ieee_get_flag(ieee_all,after_flags);call ieee_get_halting_mode(ieee_all,after_modes)
 call require(all(flags.eqv.after_flags).and.all(modes.eqv.after_modes),'classifier IEEE state')
 call require(all(prediction(1,:)==[-3,0,42]).and.all(prediction(2,:)==[7,9,42]),'joint classifier predictions')
 call require(all(prob(2)%classes==[0,2,9]).and.all(shape(prob(2)%values)==[2,3]),'ragged classes')
 call require(all(prob(1)%values(:,1)==[1._dp,0._dp]),'joint classifier probabilities')
 model%n_neighbors=1;prediction=model%predict_matrix()
 call require(prediction(3,2)==0,'classifier training query excludes self with stable tied index')
 model%weights='distance';prob=model%predict_proba_multioutput(x(:1,:))
 call require(prob(1)%values(1,1)==1,'classifier zero-distance weighting')
 do i=1,6
  d(:,i)=abs(x(:,1)-x(i,1));qd(:,i)=abs(q(:,1)-x(i,1))
 end do
 model%metric='precomputed';call model%fit(d,y)
 prediction=model%predict_matrix(qd);call require(all(prediction(1,:)==y(1,:)),'precomputed joint classifier')
 model%metric='minkowski';call model%fit(x,reshape([0,0,0,1,1,1],[6,1]))
 call require(.not.model%outputs_2d_.and.allocated(model%classes_),'column fit restores scalar API')
 single=model%predict(q);scalar_prob=model%predict_proba(q)
 call require(all(single==[0,1]).and.size(scalar_prob,2)==2,'scalar API after refit')
 call require(model%score(q,[0,1],sample_weight=[1._dp,3._dp])==1,'weighted scalar accuracy')
 call model%fit(x,reshape([0,0,0,1,1,1,1,1,1,0,0,0],[6,2]))
 call require(model%score(q,reshape([0,1,1,0],[2,2]),sample_weight=[1._dp,3._dp])==1,'weighted subset accuracy')
 print *,'Joint KNN classes, probabilities, votes, self exclusion, precomputed geometry and refits passed'
end program
