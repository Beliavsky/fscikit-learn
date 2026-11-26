program reference_neighbor_tree_driver
 use sklearn_common,only:dp
 use sklearn_neighbor_tree,only:neighbor_kd_tree
 implicit none
 type(neighbor_kd_tree)::tree
 real(dp),allocatable::x(:,:),q(:,:),bounds(:,:,:),distance(:),radii(:)
 integer,allocatable::order(:),spans(:,:),indices(:),ops(:),counts(:)
 integer::n,p,m,leaf,specs,i,j,status
 real(dp)::power
 character(16)::metric
 do
  read(*,*,iostat=status)n,p,m,leaf,specs,metric,power
  if(status/=0)exit
  allocate(x(n,p),q(m,p),radii(specs),ops(specs),counts(specs))
  do i=1,specs;read(*,*)ops(i),counts(i),radii(i);end do
  do i=1,n;read(*,*)x(i,:);end do
  do i=1,m;read(*,*)q(i,:);end do
  call tree%fit(x,metric,power,leaf);call tree%get_arrays(order,spans,bounds)
  write(*,'(*(i0,1x))')order
  write(*,'(*(i0,1x))')transpose(spans)
  write(*,'(*(es26.17e3,1x))')transpose(bounds(1,:,:))
  write(*,'(*(es26.17e3,1x))')transpose(bounds(2,:,:))
  do i=1,specs
   do j=1,m
    if(ops(i)==0)then
     call tree%query(q(j,:),counts(i),distance,indices)
    else
     call tree%query_radius(q(j,:),radii(i),distance,indices)
    end if
    write(*,'(i0)')size(indices)
    write(*,'(*(i0,1x))')indices
    write(*,'(*(es26.17e3,1x))')distance
   end do
  end do
  deallocate(x,q,radii,ops,counts)
 end do
end program
