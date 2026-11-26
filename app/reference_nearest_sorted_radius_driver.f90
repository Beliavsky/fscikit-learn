program reference_nearest_sorted_radius_driver
 use sklearn_common,only:dp
 use sklearn_neighbors,only:nearest_neighbors
 use sklearn_sparse,only:csr_matrix
 implicit none
 type(nearest_neighbors)::model
 type(csr_matrix)::graph
 real(dp),allocatable::x(:,:),q(:,:)
 integer::n,p,m,fit_k,leaf,implicit_query,exclude,sorted,i,status
 real(dp)::power,radius
 character(16)::metric,algorithm,mode
 do
  read(*,*,iostat=status)n,p,m,fit_k,leaf,implicit_query,exclude,metric,power,algorithm,radius,sorted,mode
  if(status/=0)exit
  allocate(x(n,p),q(m,p))
  do i=1,n;read(*,*)x(i,:);end do
  do i=1,m;read(*,*)q(i,:);end do
  model%metric=metric;model%p=power;model%algorithm=algorithm
  model%leaf_size=leaf;model%n_neighbors=fit_k;call model%fit(x)
  if(implicit_query==1)then
   if(exclude==1)then
    graph=model%radius_neighbors_graph(radius=radius,mode=mode,sort_results=sorted==1)
   else
    graph=model%radius_neighbors_graph(model%x_,radius,mode,sorted==1)
   end if
  else
   graph=model%radius_neighbors_graph(q,radius,mode,sorted==1)
  end if
  write(*,'(a)')trim(model%fit_method_)
  write(*,'(*(i0,1x))')graph%n_rows,graph%n_cols,graph%nnz()
  write(*,'(*(i0,1x))')graph%indptr
  write(*,'(*(i0,1x))')graph%indices
  write(*,'(*(es26.17e3,1x))')graph%data
  deallocate(x,q)
 end do
end program
