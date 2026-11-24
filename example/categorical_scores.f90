program categorical_scores
 use sklearn,only:dp,balanced_accuracy_score,matthews_corrcoef,cohen_kappa_score
 implicit none
 integer::y(6),p(6)
 real(dp)::w(6)
 y=[0,1,2,2,1,0];p=[0,2,2,1,1,0];w=[1._dp,2._dp,3._dp,4._dp,5._dp,6._dp]
 print *, 'Weighted balanced accuracy:',balanced_accuracy_score(y,p,sample_weight=w)
 print *, 'Weighted Matthews correlation:',matthews_corrcoef(y,p,sample_weight=w)
 print *, 'Quadratic kappa:',cohen_kappa_score(y,p,weights='quadratic',sample_weight=w)
 print *, 'Ordered quadratic kappa:',cohen_kappa_score(y,p,labels=[2,0,1],weights='quadratic',sample_weight=w)
end program
