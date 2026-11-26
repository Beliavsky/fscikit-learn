program reference_neighbor_gram_driver
 use sklearn_common,only:dp
 use sklearn_neighbor_gram,only:neighbor_row_norm,neighbor_middle_term
 implicit none
 real(dp),allocatable::x(:,:),y(:,:),nx(:),ny(:),middle(:,:)
 integer::m,n,p,i,j,status
 do
  read(*,*,iostat=status)m,n,p
  if(status/=0)exit
  allocate(x(m,p),y(n,p),nx(m),ny(n),middle(m,n))
  do i=1,m;read(*,*)x(i,:);nx(i)=neighbor_row_norm(x(i,:));end do
  do j=1,n;read(*,*)y(j,:);ny(j)=neighbor_row_norm(y(j,:));end do
  do j=1,n
   do i=1,m;middle(i,j)=neighbor_middle_term(x(i,:),y(j,:),m,n,i,j);end do
  end do
  write(*,'(*(es26.17e3,1x))')nx
  write(*,'(*(es26.17e3,1x))')ny
  write(*,'(*(es26.17e3,1x))')transpose(middle)
  deallocate(x,y,nx,ny,middle)
 end do
end program
