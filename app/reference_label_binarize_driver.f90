program reference_label_binarize_driver
 use sklearn_label_binarize
 use sklearn_common,only:dp
 use sklearn_sparse,only:csr_matrix
 implicit none
 integer::n,p,nc,negative,positive,sparse,rank,i
 integer,allocatable::y(:,:),classes(:),z(:,:)
 type(csr_matrix)::s
 read(*,*) n,p,nc,negative,positive,sparse,rank
 allocate(y(n,p),classes(nc));read(*,*) classes
 do i=1,n
  read(*,*) y(i,:)
 end do
 if(sparse==1) then
  if(rank==1) then
   s=label_binarize_csr(y(:,1),classes,negative,positive)
  else
   s=label_binarize_csr(y,classes,negative,positive)
  end if
  print *, 'shape',s%n_rows,s%n_cols
  print *, 'ptr',s%indptr
  print *, 'idx',s%indices
  print *, 'data',s%data
  print *, 'dense',s%to_dense()
 else
  if(rank==1) then
   z=label_binarize(y(:,1),classes,negative,positive)
  else
   z=label_binarize(y,classes,negative,positive)
  end if
  print *, 'shape',shape(z)
  print *, 'dense',z
 end if
end program
