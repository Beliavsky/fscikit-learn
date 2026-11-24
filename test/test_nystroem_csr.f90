program test_nystroem_csr
 use sklearn_common,only:dp,require
 use sklearn_sparse,only:csr_matrix,csr_from_arrays,csr_from_dense
 use sklearn_kernel_approximation,only:nystroem
 implicit none
 type(nystroem)::model,dense_model
 type(csr_matrix)::x,q,big,original
 real(dp)::a(5,3)
 real(dp),allocatable::z(:,:),d(:,:),first(:,:),basis(:,:)
 integer::i
 do i=1,5
  a(i,:)=[real(i,dp),sin(real(i,dp)),0._dp]
 end do
 x=csr_from_dense(a);original=x;model%n_components=3;dense_model%n_components=3
 z=model%fit_transform(x);d=dense_model%fit_transform(a)
 call require(all(model%component_indices_==[3,1,2]),'CSR source landmark permutation')
 call require(maxval(abs(z-d))<1.e-9_dp,'canonical CSR and dense Nystroem')
 call require(allocated(model%components_csr_).and..not.allocated(model%components_),'CSR landmark storage')
 first=model%transform(a);z=model%transform(x)
 call require(maxval(abs(first-z))<1.e-9_dp,'mixed dense/CSR Nystroem queries')
 call model%fit(a)
 call require(allocated(model%components_).and..not.allocated(model%components_csr_),'dense refit clears CSR')
 call model%fit(x)
 call require(allocated(model%components_csr_).and..not.allocated(model%components_),'CSR refit clears dense')
 call require(all(x%indptr==original%indptr).and.all(x%data==original%data),'CSR caller input preserved')
 ! Duplicate landmarks stay raw for RBF, then canonicalize on a live Laplacian query.
 q=csr_from_arrays(2,2,[1,5,9],[2,1,1,2,2,1,1,2],[0._dp,1._dp,2._dp,0._dp,1._dp,2._dp,3._dp,0._dp])
 model=nystroem();model%n_components=2;call model%fit(q)
 call require(model%components_csr_%nnz()==8,'RBF retains duplicate landmark entries')
 model%kernel='laplacian';z=model%transform(q)
 call require(model%components_csr_%nnz()==4.and.q%nnz()==8,'Laplacian canonicalizes only owned landmarks')
 basis=model%components_csr_%to_dense();model%kernel='rbf';z=model%transform(q)
 call require(all(model%components_csr_%to_dense()==basis),'later kernel retains canonical landmarks')
 ! An accidental dense input copy would allocate 80 GB here.
 big%n_rows=100000;big%n_cols=100000
 allocate(big%indptr(100001),big%indices(1),big%data(1))
 big%indptr=2;big%indptr(1)=1;big%indices=100000;big%data=2
 model=nystroem();model%n_components=2;call model%fit(big)
 call require(model%components_csr_%nnz()<=1,'large sparse fit retains sparse basis')
 q=csr_from_arrays(2,100000,[1,2,2],[100000],[2._dp]);z=model%transform(q)
 call require(all(shape(z)==[2,2]),'large sparse query emits small dense feature map')
 print *, 'Nystroem CSR landmarks, mixed queries, refits, live kernels and sparse shapes: PASS'
end program
