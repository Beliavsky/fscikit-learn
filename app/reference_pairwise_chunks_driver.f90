module chunk_reference_support
 use sklearn_common,only:dp
 use sklearn_sparse,only:csr_from_dense
 use sklearn_pairwise_chunks
 use,intrinsic::ieee_arithmetic,only:ieee_is_nan
 implicit none
 integer::metric_calls=0
 type,extends(pairwise_chunk_reducer)::summary_reducer
  integer::mode=0,calls=0
  real(dp)::total=0
 contains
  procedure::apply=>reduce_summary
 end type
contains
 real(dp)function asymmetric(x,y)result(value)
  real(dp),intent(in)::x(:),y(:)
  metric_calls=metric_calls+1
  value=sum((x-y)**2)+.01_dp*x(1)+.03_dp*y(1)+metric_calls*1.e-5_dp
 end function
 subroutine reduce_summary(self,block,reduced)
  class(summary_reducer),intent(inout)::self
  type(distance_block),intent(inout)::block
  type(distance_reduction),intent(out)::reduced
  real(dp),allocatable::d(:,:)
  integer::i,j,best
  if(block%is_sparse)then
   d=block%sparse%to_dense()
  else
   d=block%values
  end if
  self%calls=self%calls+1;self%total=self%total+sum(d)
  select case(self%mode)
  case(1)
   allocate(reduced%tuple(2));reduced%tuple(1)%real_vector=sum(d,dim=2)
   allocate(reduced%tuple(2)%integer_vector(size(d,1)))
   do i=1,size(d,1)
    best=1
    do j=2,size(d,2)
     if(ieee_is_nan(d(i,best)))cycle
     if(ieee_is_nan(d(i,j)))then
      best=j
     else if(d(i,j)<d(i,best))then
      best=j
     end if
    end do
    reduced%tuple(2)%integer_vector(i)=best-1
   end do
  case(3);reduced%sparse=csr_from_dense(real(merge(1,0,d<1._dp),dp))
  case(4);allocate(reduced%tuple(0))
  case(5);reduced%real_matrix=d
  case(6);reduced%integer_matrix=merge(1,0,d<1._dp)
  end select
 end subroutine
end module
program reference_pairwise_chunks_driver
 use sklearn_common,only:dp
 use sklearn_sparse,only:csr_matrix,csr_from_dense
 use sklearn_pairwise_chunks
 use chunk_reference_support
 implicit none
 real(dp),allocatable,target::x(:,:),y(:,:)
 type(csr_matrix),target::sx,sy
 type(pairwise_distance_iterator)::it
 type(distance_chunk)::chunk
 type(summary_reducer)::reducer
 real(dp)::memory,pvalue
 real(dp),allocatable::power,w(:),v(:),vi(:,:)
 logical,allocatable::square
 integer::code,n,m,p,spx,spy,identity,parameters,squared,mode,ios,i,j
 character(20)::metric
 character(20),parameter::names(24)=[character(20)::'euclidean','sqeuclidean','l2','manhattan', &
 'chebyshev','minkowski','cosine','nan_euclidean','haversine','braycurtis','canberra','correlation', &
 'hamming','dice','jaccard','rogerstanimoto','russellrao','sokalmichener','sokalsneath','yule', &
 'cityblock','seuclidean','mahalanobis','precomputed']
 do
  read(*,*,iostat=ios)code,n,m,p,spx,spy,identity,memory,pvalue,parameters,squared,mode
  if(ios/=0)exit
  allocate(x(n,p),y(m,p))
  do i=1,n
   read(*,*)x(i,:)
  end do
  do i=1,m
   read(*,*)y(i,:)
  end do
  sx=csr_from_dense(x);sy=csr_from_dense(y)
  if(code==6)power=pvalue
  if(parameters==1)then
   if(code==6)w=[(real(j,dp)/p+.5_dp,j=1,p)]
   if(code==22)v=[(real(j,dp)/p+.5_dp,j=1,p)]
   if(code==23)then
    allocate(vi(p,p));vi=.1_dp
    do j=1,p
     vi(j,j)=2
    end do
   end if
  end if
  if(squared==1)square=.true.
  metric_calls=0;reducer%mode=mode;reducer%calls=0;reducer%total=0
  if(code>0)then
   metric=names(code)
   if(spx==0.and.spy==0)then
    if(identity==1)then
     it=pairwise_distances_chunked(x,metric=metric,working_memory=memory,p=power,w=w,v=v,vi=vi,squared=square)
    else if(identity==2)then
     it=pairwise_distances_chunked(x,x,metric,memory,power,square,w,v,vi)
    else
     it=pairwise_distances_chunked(x,y,metric,memory,power,square,w,v,vi)
    end if
   else if(spx==1.and.spy==1)then
    if(identity==1)then
     it=pairwise_distances_chunked(sx,metric=metric,working_memory=memory,p=power,w=w,v=v,vi=vi,squared=square)
    else if(identity==2)then
     it=pairwise_distances_chunked(sx,sx,metric,memory,power,square,w,v,vi)
    else
     it=pairwise_distances_chunked(sx,sy,metric,memory,power,square,w,v,vi)
    end if
   else if(spx==0)then
    it=pairwise_distances_chunked(x,sy,metric,memory,power,square,w,v,vi)
   else
    it=pairwise_distances_chunked(sx,y,metric,memory,power,square,w,v,vi)
   end if
  else
   if(spx==0.and.spy==0)then
    if(identity==1)then
     it=pairwise_distances_chunked(x,metric=asymmetric,working_memory=memory)
    else if(identity==2)then
     it=pairwise_distances_chunked(x,x,asymmetric,memory)
    else
     it=pairwise_distances_chunked(x,y,asymmetric,memory)
    end if
   else if(spx==1.and.spy==1)then
    if(identity==1)then
     it=pairwise_distances_chunked(sx,metric=asymmetric,working_memory=memory)
    else if(identity==2)then
     it=pairwise_distances_chunked(sx,sx,asymmetric,memory)
    else
     it=pairwise_distances_chunked(sx,sy,asymmetric,memory)
    end if
   else if(spx==0)then
    it=pairwise_distances_chunked(x,sy,asymmetric,memory)
   else
    it=pairwise_distances_chunked(sx,y,asymmetric,memory)
   end if
  end if
  write(*,*)it%chunk_rows(),merge(1,0,it%exceeds_working_memory())
  do
   if(mode==0)then
    if(.not.it%next(chunk))exit
   else
    if(.not.it%next(chunk,reducer))exit
   end if
   write(*,'(a,2(1x,i0))')'BLOCK',chunk%distance%start,chunk%distance%n_rows
   if(mode==0)then
    call print_block(chunk%distance)
   else
    call print_reduced(chunk%reduced)
   end if
  end do
  write(*,'(a,2(1x,i0),1x,es26.17e3)')'END',metric_calls,reducer%calls,reducer%total
  deallocate(x,y)
  if(allocated(power))deallocate(power)
  if(allocated(w))deallocate(w)
  if(allocated(v))deallocate(v)
  if(allocated(vi))deallocate(vi)
  if(allocated(square))deallocate(square)
 end do
contains
 subroutine print_block(block)
  type(distance_block),intent(in)::block
  write(*,*)merge(1,0,block%is_sparse)
  if(block%is_sparse)then
   call print_csr(block%sparse)
  else
   call matrix(block%values)
  end if
 end subroutine
 subroutine matrix(a)
  real(dp),intent(in)::a(:,:)
  write(*,'(*(es26.17e3,1x))')a
 end subroutine
 subroutine print_csr(a)
  type(csr_matrix),intent(in)::a
  write(*,'(*(i0,1x))')a%indptr
  write(*,'(*(i0,1x))')a%indices
  write(*,'(*(es26.17e3,1x))')a%data
 end subroutine
 subroutine print_reduced(value)
  type(distance_reduction),intent(in)::value
  select case(mode)
  case(1)
   write(*,'(*(es26.17e3,1x))')value%tuple(1)%real_vector
   write(*,'(*(i0,1x))')value%tuple(2)%integer_vector
  case(3);call print_csr(value%sparse)
  case(5);call matrix(value%real_matrix)
  case(6);write(*,'(*(i0,1x))')value%integer_matrix
  case default;write(*,'(a)')'NONE'
  end select
 end subroutine
end program
