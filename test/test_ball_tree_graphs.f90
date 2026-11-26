program test_ball_tree_graphs
 use sklearn_common,only:dp,require
 use sklearn_neighbor_functions,only:kneighbors_graph,radius_neighbors_graph
 use sklearn_sparse,only:csr_matrix
 use,intrinsic::ieee_arithmetic
 use,intrinsic::ieee_exceptions
 implicit none
 type(csr_matrix)::g
 real(dp)::x(2,2),zero(12,2),inf
 logical::flags(size(ieee_all)),modes(size(ieee_all)),after_flags(size(ieee_all)),after_modes(size(ieee_all))
 inf=ieee_value(0._dp,ieee_positive_inf)
 x(1,:)=[1._dp,-1._dp];x(2,:)=[-1._dp,1._dp]
 g=radius_neighbors_graph(x,1._dp,metric='braycurtis',mode='distance',include_self=.true.)
 call require(g%nnz()==4.and.all(g%data==[0._dp,1._dp,1._dp,0._dp]),'Source Bray-Curtis reduction denominator')
 x(1,:)=[0._dp,0._dp];x(2,:)=[0._dp,1._dp]
 call ieee_get_flag(ieee_all,flags);call ieee_get_halting_mode(ieee_all,modes)
 g=radius_neighbors_graph(x,inf,metric='haversine',mode='distance',include_self=.true.)
 call ieee_get_flag(ieee_all,after_flags);call ieee_get_halting_mode(ieee_all,after_modes)
 call require(g%nnz()==0,'Source brute haversine infinite reduced cutoff')
 call require(all(flags.eqv.after_flags).and.all(modes.eqv.after_modes),'graph preserves IEEE state')
 zero=0
 g=radius_neighbors_graph(zero(:9,:),0._dp,metric='russellrao',include_self=.true.)
 call require(g%nnz()==9,'Source Boolean brute diagonal shortcut')
 g=radius_neighbors_graph(zero,0._dp,metric='russellrao',include_self=.true.)
 call require(g%nnz()==0,'Source automatic BallTree radius geometry')
 g=kneighbors_graph(zero(:9,:),4,metric='dice',mode='distance',include_self=.true.)
 call require(g%nnz()==36.and.count(ieee_is_nan(g%data))==27,'Source NaN Dice fallback selection')
 print *,'Automatic BallTree graphs and numeric/Boolean brute fallbacks passed'
end program
