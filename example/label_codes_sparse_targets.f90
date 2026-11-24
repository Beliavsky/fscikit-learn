program label_codes_sparse_targets
 use sklearn,only:label_binarize,label_binarize_csr,csr_matrix,dp
 use,intrinsic::iso_fortran_env,only:int64
 implicit none
 type(csr_matrix)::target,encoded
 integer(int64),allocatable::dense(:,:)
 target%n_rows=2;target%n_cols=3
 target%indptr=[1,4,5];target%indices=[3,3,1,2];target%data=[1._dp,1._dp,0._dp,1._dp]
 encoded=label_binarize_csr(target,[2,0,1],pos_label=3)
 dense=label_binarize(target,[0,1,2],neg_label=-2,pos_label=0)
 print *, 'Encoded stored columns:',encoded%indices
 print *, 'Encoded stored values:',encoded%data
 print *, 'Dense first row:',dense(1,:)
end program
