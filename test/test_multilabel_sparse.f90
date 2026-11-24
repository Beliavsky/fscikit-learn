program test_multilabel_sparse
 use sklearn,only:multi_label_binarizer,label_set,csr_matrix,dp
 use sklearn_common,only:require
 implicit none
 type(multi_label_binarizer)::model
 type(label_set),allocatable::train(:),decoded(:)
 type(csr_matrix)::a,b,z
 integer,allocatable::before(:),after(:)
 integer::n,i
 call random_seed(size=n);allocate(before(n),after(n));call random_seed(get=before)
 allocate(train(2));train(1)%values=[9,-8,9];train(2)%values=[16,0]
 a=model%fit_transform_csr(train)
 call require(all(model%classes_==[-8,0,9,16]),'automatic sorted classes')
 call require(all(a%indices==[3,1,4,2]),'fit_transform preserves encounter-index set order')
 b=model%transform_csr(train)
 call require(all(b%indices==[1,3,2,4]),'transform uses fitted class-index set order')
 call require(all(a%to_dense()==b%to_dense()),'fit and query dense identities')
 call require(all(train(1)%values==[9,-8,9]),'fit preserves caller ragged labels')
 z%n_rows=1;z%n_cols=4;z%indptr=[1,4];z%indices=[4,1,4];z%data=[0._dp,1._dp,1._dp]
 decoded=model%inverse_transform(z)
 call require(all(decoded(1)%values==[16,-8,16]),'CSR inverse includes stored zeros and duplicates')
 decoded(1)%values=99
 call require(all(model%classes_==[-8,0,9,16]),'inverse values own their storage')
 model%classes=[16,9,0,-8];a=model%fit_transform_csr(train)
 call require(all(a%indices==[2,4,1,3]),'constructor class ordering')
 a%data=7;b=model%transform_csr(train);call require(all(b%data==1),'CSR result owns its data')
 deallocate(model%classes,train);n=100000;allocate(train(n))
 do i=1,n
  train(i)%values=[i]
 end do
 a=model%fit_transform_csr(train)
 call require(a%n_rows==n.and.a%n_cols==n.and.a%nnz()==n,'large automatic discovery stays sparse')
 call require(all(a%indices==[(i,i=1,n)]),'large fitted class mapping')
 call random_seed(get=after);call require(all(before==after),'multilabel sparse operations preserve caller RNG')
 print *, 'MultiLabelBinarizer CSR tests PASS'
end program
