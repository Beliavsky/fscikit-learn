program reference_optics_graph_driver
  use sklearn_optics,only:compute_optics_graph,optics_graph_result
  use sklearn_common,only:dp,require
  implicit none
  type(optics_graph_result)::graph
  real(dp),allocatable::x(:,:),copy(:,:)
  real(dp)::size_fraction,max_eps,p
  character(32)::metric
  integer::n,d,k,mode,i,ios
  do
    read(*,*,iostat=ios)n,d,k,mode
    if(ios/=0)exit
    read(*,*)metric,size_fraction,max_eps,p
    allocate(x(n,d))
    do i=1,n
      read(*,*)x(i,:)
    end do
    copy=x
    if(mode==2)then
      graph=compute_optics_graph(x,k,max_eps,distance=callback)
    else if(mode==1)then
      graph=compute_optics_graph(x,size_fraction,max_eps,metric,p)
    else
      graph=compute_optics_graph(x,k,max_eps,metric,p)
    end if
    call require(all(x==copy),'OPTICS graph changed input')
    write(*,'(*(i0,1x))')graph%ordering
    write(*,'(*(es26.17e3,1x))')graph%core_distances
    write(*,'(*(es26.17e3,1x))')graph%reachability
    write(*,'(*(i0,1x))')graph%predecessor
    write(*,'(i0,1x,l1)')graph%min_samples,graph%all_unreachable
    deallocate(x)
  end do
contains
  function callback(a,b)result(value)
    real(dp),intent(in)::a(:),b(:)
    real(dp)::value
    value=sum(abs(a-b))+sum(abs(a-b)**2)/8
  end function
end program
