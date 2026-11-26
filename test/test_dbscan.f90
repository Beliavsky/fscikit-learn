program test_dbscan
 use sklearn,only:dbscan
 use sklearn_common,only:dp,require
 use sklearn_sparse,only:csr_matrix,csr_from_dense,csr_from_arrays
 implicit none
 type(dbscan)::model
 type(csr_matrix)::sparse
 real(dp)::x(5,1),weights(5)
 real(dp),allocatable::large_weights(:)
 integer,allocatable::labels(:)
 integer::i
 x(:,1)=[0._dp,.1_dp,.2_dp,2._dp,4._dp]
 weights=[2._dp,-1._dp,2._dp,3._dp,0._dp]
 model%eps=.15_dp;model%min_samples=3
 labels=model%fit_predict(x,weights)
 call require(all(labels==[0,0,0,1,-1]),'signed DBSCAN density and isolated weighted core')
 call require(all(model%core_sample_indices_==[2,4]),'weighted DBSCAN core indices')
 call require(size(model%components_,1)==2,'owned dense DBSCAN core rows')
 sparse=csr_from_dense(x);labels=model%fit_predict(sparse,weights)
 call require(all(labels==[0,0,0,1,-1]),'sparse signed DBSCAN')
 call require(allocated(model%components_csr_).and..not.allocated(model%components_),'sparse core storage')
 model%min_samples=2;labels=model%fit_predict(x)
 call require(all(labels==[0,0,0,-1,-1]),'DBSCAN refit resets labels and weights')
 call require(.not.allocated(model%components_csr_),'dense refit clears sparse components')
 model%metric='precomputed'
 sparse=csr_from_arrays(3,3,[1,2,3,3],[2,1],[.1_dp,.1_dp])
 labels=model%fit_predict(sparse)
 call require(all(labels==[0,0,-1]),'sparse precomputed DBSCAN inserts implicit zero diagonal')
 call require(all(sparse%indptr==[1,2,3,3]).and.sparse%nnz()==2,'precomputed input remains owned by caller')
 model%metric='cosine';model%eps=.5_dp
 sparse=csr_from_dense(reshape([0._dp,0._dp],[1,2]))
 labels=model%fit_predict(reshape([0._dp,0._dp],[1,2]),[3._dp])
 call require(all(labels==[0]),'dense cosine zero-row self neighbor is retained')
 labels=model%fit_predict(sparse,[3._dp])
 call require(all(labels==[-1]),'CSR cosine training copy does not force a zero diagonal')
 labels=model%fit_predict(sparse)
 call require(all(labels==[-1]),'unweighted singleton is noise')
 call require(allocated(model%components_).and..not.allocated(model%components_csr_),'Source empty components storage')
 call require(size(model%components_,1)==0.and.size(model%components_,2)==2,'empty components retain feature width')
 model%metric='precomputed';model%eps=.1_dp;model%min_samples=1
 sparse=csr_from_arrays(2000,2000,[1,(i+1,i=2,2001)],[2,3,(i,i=2,2000)],spread(0._dp,1,2001))
 allocate(large_weights(2000));large_weights=0;large_weights(:3)=[1.e16_dp,-1.e16_dp,1._dp]
 labels=model%fit_predict(sparse,large_weights)
 call require(all(labels(:3)==0),'fast sparse diagonal insertion preserves cancellation-sensitive core')
 call require(all(model%core_sample_indices_==[1,3]),'fast insertion Source core indices')
 call require(all(model%components_csr_%indices(:3)==[1,2,3]),'fast insertion preserves sorted row structure')
 print *,'Weighted dense/CSR DBSCAN, sparse precomputed self edges and refits passed'
end program
