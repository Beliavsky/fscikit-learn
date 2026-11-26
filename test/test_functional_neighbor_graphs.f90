program test_functional_neighbor_graphs
 use sklearn_neighbors,only:nearest_neighbors,kneighbors_classifier,kneighbors_regressor, &
  radius_neighbors_classifier,radius_neighbors_regressor,kneighbors_graph,radius_neighbors_graph, &
  sort_graph_by_row_values,csr_matrix,csr_from_dense,csr_from_arrays
 use sklearn_common,only:dp,require
 use,intrinsic::ieee_arithmetic
 use,intrinsic::ieee_exceptions
 implicit none
 type(nearest_neighbors)::nn
 type(kneighbors_classifier)::kc
 type(kneighbors_regressor)::kr
 type(radius_neighbors_classifier)::rc
 type(radius_neighbors_regressor)::rr
 type(csr_matrix)::g,s,t,original
 real(dp)::x(4,1),d(4,4),z(4,2)
 logical::sorted,flags(size(ieee_all)),modes(size(ieee_all)),after_flags(size(ieee_all)),after_modes(size(ieee_all))
 integer::i
 x(:,1)=[0._dp,1._dp,3._dp,7._dp]
 g=kneighbors_graph(x,1,mode='distance')
 call require(all(g%indices==[2,1,2,3]).and.all(g%data==[1._dp,1._dp,2._dp,4._dp]),'functional k distances')
 g=kneighbors_graph(x,1,include_self='auto')
 call require(all(g%indices==[1,2,3,4]).and.all(g%data==1),'auto connectivity includes self')
 g=kneighbors_graph(x,1,mode='distance',include_self='auto')
 call require(all(g%data>0),'auto distance excludes self')
 g=radius_neighbors_graph(x,2._dp,mode='distance')
 call require(all(g%indptr==[1,2,4,5,5]).and.all(g%indices==[2,1,3,2]),'functional radius neighborhoods')
 call require(all(g%data==[1._dp,1._dp,2._dp,2._dp]),'functional radius values')
 g=radius_neighbors_graph(x,0._dp,include_self='auto')
 call require(g%nnz()==4.and.all(g%indices==[1,2,3,4]),'auto zero-radius connectivity')
 s=csr_from_dense(x);g=kneighbors_graph(s,1,mode='distance')
 call require(all(g%data==[1._dp,1._dp,2._dp,4._dp]),'CSR feature graph')
 do i=1,4
  d(i,:)=abs(x(i,1)-x(:,1))
 end do
 s=csr_from_dense(d,.true.);g=kneighbors_graph(s,1,metric='precomputed',mode='distance')
 call require(all(g%indices==[2,1,2,3]).and.all(g%data==[1._dp,1._dp,2._dp,4._dp]),'stored precomputed graph')
 s=csr_from_arrays(3,3,[1,4,6,9],[3,1,3,2,2,3,1,1],[1._dp,1._dp,2._dp,1._dp,2._dp,1._dp,-1._dp,2._dp])
 g=kneighbors_graph(s,1,metric='manhattan',mode='distance')
 call require(all(g%data==[6._dp,7._dp,6._dp]),'raw CSR Manhattan buffer geometry')
 g=kneighbors_graph(s,1,metric='euclidean',mode='distance')
 call require(maxval(abs(g%data-[2._dp,sqrt(11._dp),2._dp]))<1.e-12_dp,'raw CSR Euclidean stored norms')
 g=radius_neighbors_graph(s,2._dp,metric='manhattan',include_self=.true.)
 call require(g%nnz()==3,'raw CSR radius buffer geometry')
 call nn%fit(x);g=kneighbors_graph(nn,1)
 call require(nn%n_features_in_==1.and.nn%n_samples_fit_==4.and.g%nnz()==4,'fitted nearest graph metadata')
 call kc%fit(x,[0,0,1,1]);g=kneighbors_graph(kc,1,include_self='auto')
 call require(all(g%indices==[1,2,3,4]),'fitted classifier graph')
 call kr%fit(x,x(:,1));g=kneighbors_graph(kr,1)
 call require(all(g%indices==[2,1,2,3]),'fitted regressor graph')
 call rc%fit(x,[0,0,1,1]);g=radius_neighbors_graph(rc,0._dp,include_self=.true.)
 call require(g%nnz()==4,'fitted radius classifier graph')
 call rr%fit(x,x(:,1));g=radius_neighbors_graph(rr,0._dp,include_self='auto')
 call require(g%nnz()==4,'fitted radius regressor graph')
 z(:,1)=[0._dp,2._dp,1._dp,4._dp];z(:,2)=[0._dp,1._dp,2._dp,4._dp]
 call nn%fit(z);nn%p=1;g=kneighbors_graph(nn,1,mode='distance',p=1._dp)
 call require(abs(g%data(1)-sqrt(5._dp))<1.e-12_dp,'fitted geometry retained after public parameter change')
 s=csr_from_arrays(3,3,[1,5,8,9],[3,2,2,1,3,1,2,2],[2._dp,0._dp,0._dp,1._dp,3._dp,1._dp,2._dp,-1._dp])
 original=s;t=sort_graph_by_row_values(s,copy=.true.,was_sorted=sorted)
 call require(.not.sorted.and.all(s%data==original%data),'sort copy preserves input')
 call require(all(t%indices==[2,2,1,3,1,2,3,2]),'stable value sort preserves tied columns')
 t=sort_graph_by_row_values(s)
 call require(all(s%data==t%data).and.all(s%indices==t%indices),'default sort mutates input')
 t=sort_graph_by_row_values(s,was_sorted=sorted);call require(sorted,'sorted fast path')
 s=csr_from_arrays(2,2,[1,1,3],[1,2],[2._dp,1._dp])
 t=sort_graph_by_row_values(s,was_sorted=sorted)
 call require(sorted.and.all(t%data==[2._dp,1._dp]),'Source leading-empty-row sortedness rule')
 s=csr_from_arrays(1,4,[1,5],[1,2,3,4], &
  [ieee_value(0._dp,ieee_quiet_nan),2._dp,1._dp,ieee_value(0._dp,ieee_quiet_nan)])
 call ieee_get_flag(ieee_all,flags);call ieee_get_halting_mode(ieee_all,modes)
 t=sort_graph_by_row_values(s)
 call ieee_get_flag(ieee_all,after_flags);call ieee_get_halting_mode(ieee_all,after_modes)
 call require(all(flags.eqv.after_flags).and.all(modes.eqv.after_modes),'sorting preserves IEEE state')
 call require(all(t%indices==[3,2,1,4]).and.all(ieee_is_nan(t%data(3:))),'stable NaN value ordering')
 print *,'Functional dense/CSR graphs, fitted adapters, include_self, stable sorting and IEEE state passed'
end program
