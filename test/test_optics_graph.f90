program test_optics_graph
  use sklearn,only:dp,optics,compute_optics_graph,optics_graph_result,cluster_optics_dbscan
  use sklearn_common,only:require
  use,intrinsic::ieee_arithmetic,only:ieee_is_finite
  implicit none
  type(optics_graph_result)::graph,other
  type(optics)::model
  real(dp)::x(6,2),d(3,3)
  real(dp),allocatable::large(:,:)
  integer,allocatable::labels(:)
  integer::i
  x(1,:)=[1._dp,2._dp];x(2,:)=[2._dp,5._dp];x(3,:)=[3._dp,6._dp]
  x(4,:)=[8._dp,7._dp];x(5,:)=[8._dp,8._dp];x(6,:)=[7._dp,3._dp]
  graph=compute_optics_graph(x,2)
  call require(all(graph%ordering==[1,2,3,6,4,5]),'public OPTICS source example order')
  call require(all(graph%predecessor==[-1,1,2,6,4,3]),'public OPTICS source example predecessors')
  call require(abs(graph%core_distances(1)-sqrt(10._dp))<1.e-12_dp,'public OPTICS source example core')
  labels=cluster_optics_dbscan(graph%reachability,graph%core_distances,graph%ordering,2._dp)
  call require(all(labels==[-1,0,0,1,1,-1]),'graph reused by DBSCAN extraction')
  other=compute_optics_graph(x,1._dp)
  call require(other%min_samples==6,'fraction one resolves to all samples')
  other=compute_optics_graph(x,.01_dp,distance=distance)
  call require(other%min_samples==2,'fraction callback and lower clamp')
  model%min_samples=2;model%metric='manhattan';call model%fit(x)
  other=compute_optics_graph(x,2,metric='manhattan')
  call require(all(model%ordering_==other%ordering),'streaming estimator and function agree')
  call require(all(model%predecessor_==other%predecessor),'streaming predecessor agreement')
  d=1.e300_dp
  do i=1,3
    d(i,i)=0
  end do
  other=compute_optics_graph(d,2,metric='precomputed')
  call require(other%all_unreachable.and.all(.not.ieee_is_finite(other%core_distances)), &
    'source decimal rounding overflow becomes infinity without trapping')
  model%metric='precomputed';call model%fit(d)
  call require(all(.not.ieee_is_finite(model%core_distances_)).and.all(model%labels_==-1), &
    'estimator source decimal rounding overflow')
  allocate(large(8192,1))
  large(:,1)=[(real(i,dp),i=1,size(large,1))]
  other=compute_optics_graph(large,2,max_eps=0._dp,metric='manhattan')
  call require(other%all_unreachable.and.all(other%predecessor==-1),'large streamed graph unreachable')
  call require(all(other%ordering==[(i,i=1,size(large,1))]),'large streamed graph index ties')
  print *,'OPTICS graph tests passed; avoided a 512 MiB distance matrix'
contains
  function distance(a,b)result(value)
    real(dp),intent(in)::a(:),b(:)
    real(dp)::value
    value=sum(abs(a-b))
  end function
end program
