program subset_accuracy
 use sklearn,only:dp,accuracy_score,zero_one_loss,csr_matrix
 use sklearn_sparse,only:csr_from_dense
 implicit none
 logical::truth(3,2),guess(3,2)
 type(csr_matrix)::sparse_guess
 truth=reshape([.true.,.true.,.false.,.false.,.true.,.false.],[3,2])
 guess=truth;guess(2,1)=.false.;sparse_guess=csr_from_dense(merge(1._dp,0._dp,guess))
 print *, 'Weighted subset accuracy:',accuracy_score(truth,guess,sample_weight=[1._dp,4._dp,2._dp])
 print *, 'Weighted zero-one count:',zero_one_loss(truth,guess,normalize=.false.,sample_weight=[1._dp,4._dp,2._dp])
 print *, 'Mixed dense/CSR accuracy:',accuracy_score(truth,sparse_guess)
end program
