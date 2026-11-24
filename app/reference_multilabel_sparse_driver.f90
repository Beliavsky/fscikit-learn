program reference_multilabel_sparse_driver
 use sklearn_label_binarization,only:multi_label_binarizer,label_set
 use sklearn_sparse,only:csr_matrix
 use sklearn_common,only:dp
 implicit none
 type(multi_label_binarizer)::model
 type(label_set),allocatable::train(:),query(:),decoded(:)
 type(csr_matrix)::a,b,inverse
 integer,allocatable::classes(:),lengths(:),flat(:)
 integer::n,m,c,ctor,method,invert,refit,i,k,ni,pi,stored,pos
 read(*,*) n,m,c,ctor,method,invert,refit
 allocate(classes(c),train(n),query(m));read(*,*) classes
 if(ctor==1) model%classes=classes
 do i=1,n
  read(*,*) k;allocate(train(i)%values(k));read(*,*) train(i)%values
 end do
 do i=1,m
  read(*,*) k;allocate(query(i)%values(k));read(*,*) query(i)%values
 end do
 if(invert>=3) then
  read(*,*) ni,pi,stored;inverse%n_rows=ni;inverse%n_cols=pi
  allocate(inverse%indptr(ni+1),inverse%indices(stored),inverse%data(stored))
  read(*,*) inverse%indptr
  read(*,*) inverse%indices
  read(*,*) inverse%data
 end if
 if(method==1) then
  a=model%fit_transform_csr(train)
 else
  call model%fit(train);a=model%transform_csr(train)
 end if
 print *, 'classes',model%classes_
 call emit('train',a)
 print *, 'legacy_train',model%transform(train)
 b=model%transform_csr(query);call emit('query',b)
 if(invert==1) inverse=a
 if(invert==2) inverse=b
 if(invert>0) then
  decoded=model%inverse_transform(inverse);allocate(lengths(size(decoded)))
  do i=1,size(decoded)
   lengths(i)=size(decoded(i)%values)
  end do
  allocate(flat(sum(lengths)));pos=1
  do i=1,size(decoded)
   k=lengths(i);flat(pos:pos+k-1)=decoded(i)%values;pos=pos+k
  end do
  print *, 'inverse_size',lengths
  print *, 'inverse_flat',flat
 end if
 if(refit==1) then
  call model%fit(query);print *, 'refit_classes',model%classes_
  b=model%transform_csr(train);call emit('refit',b)
 end if
contains
 subroutine emit(prefix,z)
  character(*),intent(in)::prefix
  type(csr_matrix),intent(in)::z
  print *, prefix//'_shape',z%n_rows,z%n_cols
  print *, prefix//'_ptr',z%indptr
  print *, prefix//'_idx',z%indices
  print *, prefix//'_data',z%data
  print *, prefix//'_dense',z%to_dense()
 end subroutine
end program
