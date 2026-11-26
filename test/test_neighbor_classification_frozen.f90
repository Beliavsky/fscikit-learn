program test_neighbor_classification_frozen
 use sklearn_common,only:dp,require
 use sklearn_neighbors,only:kneighbors_classifier
 use sklearn_frozen,only:frozen_estimator
 implicit none
 type(kneighbors_classifier),target::model
 type(frozen_estimator)::frozen
 real(dp)::x(4,1)
 integer,allocatable::prediction(:)
 x(:,1)=[0._dp,1._dp,2._dp,3._dp]
 call frozen%set_params(model)
 call require(.not.frozen%is_fitted(),'unfitted joint classifier wrapper')
 model%n_neighbors=1;call model%fit(x,reshape([0,0,1,1,2,2,4,4],[4,2]))
 call require(frozen%is_fitted(),'joint classifier fitted state')
 call frozen%fit();call require(model%n_outputs_==2,'frozen fit preserves joint labels')
 call model%fit(x,[0,0,1,1]);prediction=frozen%predict_class(x)
 call require(all(prediction==[0,0,1,1]),'borrowed scalar classifier refit')
 print *,'Frozen joint classifier fitted state and borrowed refit passed'
end program
