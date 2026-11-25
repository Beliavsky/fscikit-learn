program tweedie_scores
 use sklearn,only:dp,mean_tweedie_deviance,d2_tweedie_score
 implicit none
 real(dp)::y(4),p(4),w(4)
 y=[.5_dp,1._dp,2.5_dp,7._dp];p=[1._dp,1._dp,5._dp,3.5_dp];w=[1._dp,2._dp,0._dp,3._dp]
 print *, 'Normal D2:',d2_tweedie_score(y,p)
 print *, 'Poisson D2:',d2_tweedie_score(y,p,power=1._dp)
 print *, 'Gamma D2:',d2_tweedie_score(reshape(y,[4,1]),p,power=2._dp)
 print *, 'Weighted compound deviance:',mean_tweedie_deviance(y,p,w,power=1.5_dp)
end program
