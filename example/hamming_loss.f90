program hamming_loss_example
 use sklearn,only:dp,hamming_loss,csr_from_dense
 implicit none
 integer::y(2,3),p(2,3)
 y=reshape([0,1,1,0,0,1],[2,3]);p=reshape([1,1,1,1,0,0],[2,3])
 print *, 'Dense multilabel Hamming loss:',hamming_loss(y,p)
 print *, 'Weighted CSR Hamming loss:',hamming_loss(csr_from_dense(real(y,dp)), &
  csr_from_dense(real(p,dp)),sample_weight=[1._dp,3._dp])
end program
