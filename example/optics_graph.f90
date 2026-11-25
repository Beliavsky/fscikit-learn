program optics_graph
  use sklearn,only:dp,compute_optics_graph,optics_graph_result,cluster_optics_dbscan
  implicit none
  type(optics_graph_result)::graph
  real(dp)::x(6,1)
  integer,allocatable::labels(:)
  x(:,1)=[0._dp,.1_dp,.2_dp,3._dp,3.1_dp,3.2_dp]
  graph=compute_optics_graph(x,2,metric='manhattan')
  labels=cluster_optics_dbscan(graph%reachability,graph%core_distances,graph%ordering,.25_dp)
  print *,'Ordering:',graph%ordering
  print *,'DBSCAN labels:',labels
end program
