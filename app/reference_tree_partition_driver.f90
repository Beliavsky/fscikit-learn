program reference_tree_partition_driver
 use sklearn_common,only:dp
 use sklearn_tree_partition,only:partition_node
 implicit none
 real(dp),allocatable::x(:,:),low(:,:),high(:,:)
 integer,allocatable::order(:),spans(:,:)
 integer::n,p,leaf,levels,ratio,nodes,i,status
 do
  read(*,*,iostat=status)n,p,leaf
  if(status/=0)exit
  allocate(x(n,p));do i=1,n;read(*,*)x(i,:);end do
  levels=1;ratio=max(1,(n-1)/leaf)
  do while(ratio>=2);ratio=ratio/2;levels=levels+1;end do
  nodes=2**levels-1
  allocate(order(n),spans(nodes,3),low(nodes,p),high(nodes,p));order=[(i,i=1,n)]
  call grow(1,1,n)
  write(*,'(*(i0,1x))')order
  write(*,'(*(i0,1x))')transpose(spans)
  write(*,'(*(es26.17e3,1x))')transpose(low)
  write(*,'(*(es26.17e3,1x))')transpose(high)
  deallocate(x,order,spans,low,high)
 end do
contains
 recursive subroutine grow(node,first,last)
  integer,intent(in)::node,first,last
  integer::feature,middle
  low(node,:)=minval(x(order(first:last),:),dim=1);high(node,:)=maxval(x(order(first:last),:),dim=1)
  spans(node,:)=[first,last,merge(1,0,2*node>nodes)]
  if(2*node>nodes)return
  feature=maxloc(high(node,:)-low(node,:),dim=1);middle=first+(last-first+1)/2
  call partition_node(x(:,feature),order(first:last),middle-first+1)
  call grow(2*node,first,middle-1);call grow(2*node+1,middle,last)
 end subroutine
end program
