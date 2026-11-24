program test_label_binarize_sparse
 use sklearn,only:label_binarize,label_binarize_csr,csr_matrix,dp
 use sklearn_common,only:require
 use,intrinsic::iso_fortran_env,only:int64
 use,intrinsic::ieee_arithmetic,only:ieee_is_negative
 implicit none
 type(csr_matrix)::y,z,saved
 integer(int64),allocatable::dense(:,:)
 integer,allocatable::classes(:),before(:),after(:)
 integer::i,n
 call random_seed(size=n);allocate(before(n),after(n));call random_seed(get=before)
 y%n_rows=2;y%n_cols=3;y%indptr=[1,4,5];y%indices=[3,3,1,2];y%data=[1._dp,1._dp,-0._dp,1._dp];saved=y
 dense=label_binarize(y,[0,1,2],neg_label=-2,pos_label=0)
 call require(all(dense(1,:)==[0_int64,-2_int64,4_int64]),'positive-zero switch follows duplicate coalescing')
 z=label_binarize_csr(y,[2,0,1],pos_label=3)
 call require(all(z%indices==[1,1,2,3]).and.all(z%data==3),'stored zero recoding and raw duplicate order')
 call require(all(y%data==saved%data).and.all(y%indices==saved%indices).and.all(y%indptr==saved%indptr), &
  'CSR targets remain unchanged')
 z%data=10;z=label_binarize_csr(y,[0,1,2]);call require(all(z%data==y%data),'CSR output owns its data')
 call require(.not.ieee_is_negative(z%data(3)).and.ieee_is_negative(y%data(3)), &
  'source integer cast normalizes signed zero without modifying input')
 y%n_rows=1;y%n_cols=2;y%indptr=[1,13];y%indices=spread(1,1,12);y%data=spread(1e18_dp,1,12)
 dense=label_binarize(y,[0,1]);call require(dense(1,1)==-huge(0_int64)-1_int64,'source int64 out-of-range cast')
 n=100000;y%n_rows=n;y%n_cols=n;y%indptr=[(i,i=1,n+1)];y%indices=[(i,i=1,n)];y%data=spread(1._dp,1,n)
 classes=[(i,i=n,1,-1)];z=label_binarize_csr(y,classes)
 call require(z%n_rows==n.and.z%n_cols==n.and.z%nnz()==n,'large CSR target stays compact')
 call require(all(z%indices==classes),'large CSR class permutation')
 call random_seed(get=after);call require(all(before==after),'CSR binarization preserves caller RNG')
 print *, 'CSR target label_binarize tests PASS'
end program
