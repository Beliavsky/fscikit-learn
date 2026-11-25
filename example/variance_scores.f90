program variance_scores
 use sklearn,only:dp,r2_score,explained_variance_score,r2_score_raw
 implicit none
 real(dp)::y(3,2),p(3,2)
 y(:,1)=[1._dp,2._dp,3._dp];y(:,2)=2*y(:,1)
 p=y+1
 print *, 'R2:',r2_score(y,p)
 print *, 'Explained variance:',explained_variance_score(y,p)
 print *, 'Per-output R2:',r2_score_raw(y,p)
 print *, 'Variance-weighted R2:',r2_score(y,p,multioutput='variance_weighted')
end program
