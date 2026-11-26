program reference_neighbor_query_driver
 use sklearn_common,only:dp
 use sklearn_neighbor_query,only:neighbor_query_state
 use sklearn_sparse,only:csr_matrix
 implicit none
 type(neighbor_query_state)::model
 type(csr_matrix)::graph
 real(dp),allocatable::x(:,:),q(:,:),d(:,:)
 integer,allocatable::ids(:,:)
 real(dp)::power,radius
 integer::n,p,m,k,fit_k,leaf,family,implicit_query,exclude,i,status
 character(16)::metric,algorithm
 do
  read(*,*,iostat=status)n,p,m,k,fit_k,leaf,family,implicit_query,exclude,metric,power,algorithm,radius
  if(status/=0)exit
  allocate(x(n,p),q(m,p))
  do i=1,n;read(*,*)x(i,:);end do
  do i=1,m;read(*,*)q(i,:);end do
  call model%fit(x,metric,power,algorithm,leaf,fit_k)
  if(family==0)then
   if(implicit_query==1)then
    call model%kneighbors(x,distances=d,indices=ids,n_neighbors=k,exclude_self=exclude==1)
   else
    call model%kneighbors(x,q,d,ids,k,exclude==1)
   end if
   write(*,'(a)')trim(model%fit_method())
   write(*,'(*(es26.17e3,1x))')transpose(d)
   write(*,'(*(i0,1x))')transpose(ids)
  else
   if(implicit_query==1)then
    graph=model%radius_graph(x,radius=radius,exclude_self=exclude==1)
   else
    graph=model%radius_graph(x,q,radius,exclude==1)
   end if
   write(*,'(a)')trim(model%fit_method())
   write(*,'(*(i0,1x))')graph%n_rows,graph%n_cols,graph%nnz()
   write(*,'(*(i0,1x))')graph%indptr
   write(*,'(*(i0,1x))')graph%indices
   write(*,'(*(es26.17e3,1x))')graph%data
  end if
  deallocate(x,q)
 end do
end program
