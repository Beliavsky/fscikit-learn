program reference_label_binarize_sparse_driver
 use sklearn_label_binarize
 use sklearn_common,only:dp
 use sklearn_sparse,only:csr_matrix
 use,intrinsic::iso_fortran_env,only:int64
 implicit none
 type(csr_matrix)::y,z
 integer,allocatable::classes(:)
 integer(int64),allocatable::dense(:,:)
 integer::n,p,nc,nnz,negative,positive,sparse
 character(32)::invalid
 if(command_argument_count()>0) then
  call get_command_argument(1,invalid)
  y%n_rows=1;y%n_cols=2
  if(invalid/='unallocated'.and.invalid/='negative_shape') then
   y%indptr=[1,2];y%indices=[1];y%data=[1._dp]
  end if
  select case(trim(invalid))
  case('negative_shape');y%n_rows=-1
  case('bad_pointer');y%indptr(2)=4
  case('bad_index');y%indices(1)=3
  end select
  z=label_binarize_csr(y,[0,1]);error stop 'invalid CSR accepted'
 end if
 read(*,*) n,p,nc,nnz,negative,positive,sparse
 y%n_rows=n;y%n_cols=p
 allocate(classes(nc),y%indptr(n+1),y%indices(nnz),y%data(nnz))
 read(*,*) classes
 read(*,*) y%indptr
 read(*,*) y%indices
 read(*,*) y%data
 if(sparse==1) then
  z=label_binarize_csr(y,classes,negative,positive)
  print *, 'shape',z%n_rows,z%n_cols
  print *, 'ptr',z%indptr
  print *, 'idx',z%indices
  print *, 'data',z%data
  print *, 'dense',z%to_dense()
 else
  dense=label_binarize(y,classes,negative,positive)
  print *, 'shape',shape(dense)
  print *, 'dense',dense
 end if
 print *, 'input_ptr',y%indptr
 print *, 'input_idx',y%indices
 print *, 'input_data',y%data
end program
