program likelihood_ratios
 use sklearn,only:dp,class_likelihood_ratios
 implicit none
 real(dp)::ratios(2)
 ratios=class_likelihood_ratios([0,1,0,1,0],[1,1,0,0,0])
 print *, 'Positive and negative likelihood ratios:',ratios
 ratios=class_likelihood_ratios([0,1,0,1,0],[1,1,0,0,0], &
  sample_weight=[2._dp,3._dp,4._dp,5._dp,6._dp])
 print *, 'Weighted likelihood ratios:',ratios
end program
