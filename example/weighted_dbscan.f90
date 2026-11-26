program weighted_dbscan
 use sklearn,only:dp,dbscan
 implicit none
 type(dbscan)::model
 real(dp)::x(5,1)
 integer,allocatable::labels(:)
 x(:,1)=[0._dp,.1_dp,.2_dp,2._dp,4._dp]
 model%eps=.15_dp;model%min_samples=3
 labels=model%fit_predict(x,sample_weight=[2._dp,-1._dp,2._dp,3._dp,0._dp])
 print *,'Cluster labels (noise = -1):',labels
 print *,'Core row indices:',model%core_sample_indices_
end program
