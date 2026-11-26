program reference_dbscan_driver
 use sklearn_common,only:dp
 use sklearn_dbscan,only:dbscan
 use sklearn_sparse,only:csr_matrix
 implicit none
 type(dbscan)::model
 type(csr_matrix)::sparse
 real(dp),allocatable::x(:,:),weights(:),components(:,:)
 integer::n,p,storage,weighted,nnz,i,status
 do
  read(*,*,iostat=status)n,p,storage,weighted,model%metric,model%algorithm,model%eps,model%p, &
   model%min_samples,model%leaf_size
  if(status/=0)exit
  allocate(x(n,p),weights(n))
  do i=1,n;read(*,*)x(i,:);end do
  read(*,*)weights
  if(storage==1)then
   read(*,*)nnz
   sparse%n_rows=n;sparse%n_cols=p
   if(allocated(sparse%data))deallocate(sparse%indptr,sparse%indices,sparse%data)
   allocate(sparse%indptr(n+1),sparse%indices(nnz),sparse%data(nnz))
   read(*,*)sparse%indptr
   read(*,*)sparse%indices
   read(*,*)sparse%data
   if(weighted==1)then
    call model%fit(sparse,weights)
   else
    call model%fit(sparse)
   end if
   if(allocated(model%components_csr_))then
    components=model%components_csr_%to_dense()
   else
    components=model%components_
   end if
  else
   if(weighted==1)then
    call model%fit(x,weights)
   else
    call model%fit(x)
   end if
   components=model%components_
  end if
  write(*,'(i0,1x,i0,1x,a)')size(model%core_sample_indices_),model%n_features_in_,trim(model%fit_method_)
  write(*,'(*(i0,1x))')model%labels_
  write(*,'(*(i0,1x))')model%core_sample_indices_
  write(*,'(*(es26.17e3,1x))')transpose(components)
  if(allocated(model%components_csr_))then
   write(*,'(*(es26.17e3,1x))')model%components_csr_%data
   write(*,'(*(i0,1x))')model%components_csr_%indices
   write(*,'(*(i0,1x))')model%components_csr_%indptr
  else
   write(*,*)
   write(*,*)
   write(*,*)
  end if
  deallocate(x,weights)
 end do
end program
