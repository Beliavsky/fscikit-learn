program label_codes
 use sklearn,only:label_binarize,label_binarize_csr,csr_matrix
 implicit none
 integer,allocatable::dense(:,:)
 type(csr_matrix)::sparse
 dense=label_binarize([1,6,99],[1,6,4,2],neg_label=-1,pos_label=3)
 print *, 'First class:',dense(:,1)
 print *, 'Second class:',dense(:,2)
 sparse=label_binarize_csr([1,6,99],[1,6,4,2])
 print *, 'Sparse shape and stored count:',sparse%n_rows,sparse%n_cols,sparse%nnz()
 print *, 'Sparse columns:',sparse%indices
end program
