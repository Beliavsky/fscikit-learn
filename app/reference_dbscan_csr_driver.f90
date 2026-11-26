program reference_dbscan_csr_driver
 use sklearn_common,only:dp
 use sklearn_dbscan,only:dbscan
 use sklearn_sparse,only:csr_matrix
 implicit none
 type(dbscan)::model
 type(csr_matrix)::x
 real(dp),allocatable::weights(:)
 integer::n,p,nnz,weighted,status
 do
  read(*,*,iostat=status)n,p,nnz,weighted,model%eps,model%min_samples
  if(status/=0)exit
  x%n_rows=n;x%n_cols=p;model%metric='precomputed'
  allocate(x%indptr(n+1),x%indices(nnz),x%data(nnz),weights(n))
  read(*,*)x%indptr
  read(*,*)x%indices
  read(*,*)x%data
  read(*,*)weights
  if(weighted==1)then
   call model%fit(x,weights)
  else
   call model%fit(x)
  end if
  write(*,'(i0,1x,i0)')size(model%core_sample_indices_),model%n_features_in_
  write(*,'(*(i0,1x))')model%labels_
  write(*,'(*(i0,1x))')model%core_sample_indices_
  if(allocated(model%components_csr_))then
   write(*,'(*(es26.17e3,1x))')model%components_csr_%data
   write(*,'(*(i0,1x))')model%components_csr_%indices
   write(*,'(*(i0,1x))')model%components_csr_%indptr
  else
   write(*,*)
   write(*,*)
   write(*,*)
  end if
  deallocate(x%indptr,x%indices,x%data,weights)
 end do
end program
