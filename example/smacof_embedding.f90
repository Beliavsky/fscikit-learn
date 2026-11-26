program smacof_embedding
 use sklearn,only:dp,smacof,smacof_result
 implicit none
 type(smacof_result)::fit
 real(dp)::d(3,3)
 d=reshape([0._dp,1._dp,3._dp,1._dp,0._dp,2._dp,3._dp,2._dp,0._dp],[3,3])
 fit=smacof(d,n_components=2,n_init=4,random_state=42,eps=1.e-8_dp)
 print *,'Stress: ',fit%stress
 print *,'Iterations: ',fit%n_iter
 print *,'Embedding: ',fit%embedding
end program
