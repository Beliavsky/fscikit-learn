program functional_neighbor_graphs
 use sklearn,only:dp,csr_matrix,csr_from_arrays,kneighbors_graph,radius_neighbors_graph,sort_graph_by_row_values
 implicit none
 real(dp)::x(4,1)
 type(csr_matrix)::near,within,ordered,distances
 x(:,1)=[0._dp,1._dp,3._dp,7._dp]
 near=kneighbors_graph(x,1,mode='distance')
 within=radius_neighbors_graph(x,2._dp,include_self='auto')
 print *,'Nearest distance graph:',near%to_dense()
 print *,'Radius connectivity graph:',within%to_dense()
 distances=csr_from_arrays(2,2,[1,3,5],[2,1,1,2],[1._dp,0._dp,1._dp,0._dp])
 ordered=sort_graph_by_row_values(distances,copy=.true.)
 print *,'Distance-ordered CSR entries:',ordered%data
 print *,'Corresponding columns:',ordered%indices
end program
