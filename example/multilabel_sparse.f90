program multilabel_sparse
 use sklearn,only:multi_label_binarizer,label_set,csr_matrix
 implicit none
 type(multi_label_binarizer)::model
 type(label_set)::labels(2)
 type(label_set),allocatable::restored(:)
 type(csr_matrix)::features
 labels(1)%values=[9,-8,9];labels(2)%values=[16,0]
 features=model%fit_transform_csr(labels);restored=model%inverse_transform(features)
 print *, 'Fitted classes:',model%classes_
 print *, 'Stored columns:',features%indices
 print *, 'First restored row:',restored(1)%values
end program
