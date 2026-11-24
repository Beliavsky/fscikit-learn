program test_label_binarize
 use sklearn,only:label_binarize,label_binarize_csr,csr_matrix
 use sklearn_common,only:require
 implicit none
 integer,allocatable::z(:,:),large(:),before(:),after(:)
 integer::x(3,3),saved(3,3),i,seed_size
 type(csr_matrix)::s
 call random_seed(size=seed_size);allocate(before(seed_size),after(seed_size));call random_seed(get=before)
 z=label_binarize([1,6,99],[1,6,4,2],-3,7)
 call require(all(z(:,1)==[7,-3,-3]).and.all(z(:,2)==[-3,7,-3]),'class order and unknown labels')
 z=label_binarize([4,8,4],[8,4],-2,0)
 call require(all(z(:,1)==[0,-2,0]),'binary class ordering and zero positive label')
 z=label_binarize([1,1],[1],-5,9)
 call require(all(z==-5),'single-class output is negative')
 x=reshape([1,0,1,0,1,0,1,1,0],[3,3]);saved=x
 z=label_binarize(x,[2,0,1]);s=label_binarize_csr(x,[2,0,1])
 call require(all(real(z)==s%to_dense()),'dense and CSR indicator values')
 call require(all(z(:,1)==x(:,3)).and.all(z(:,2)==x(:,1)), 'indicator source permutation')
 call require(all(x==saved),'caller indicator values preserved')
 s%data=12;s=label_binarize_csr(x,[2,0,1])
 call require(all(s%data==1),'CSR results own their storage')
 z=label_binarize([0,1,2],[integer::])
 call require(size(z,1)==3.and.size(z,2)==0,'multiclass empty class output')
 large=[(i,i=1,100000)];s=label_binarize_csr(large,large(100000:1:-1))
 call require(s%n_cols==100000.and.s%nnz()==100000,'compact large sparse output')
 call require(all(s%indices==large(100000:1:-1)),'large sparse class ordering')
 call random_seed(get=after);call require(all(before==after),'free binarization preserves caller RNG')
 print *, 'Free label_binarize tests PASS'
end program
