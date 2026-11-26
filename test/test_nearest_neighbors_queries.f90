program test_nearest_neighbors_queries
 use sklearn_common,only:dp,require
 use sklearn_neighbors,only:nearest_neighbors,kneighbors_graph,radius_neighbors_graph
 use sklearn_sparse,only:csr_matrix
 use,intrinsic::ieee_arithmetic
 implicit none
 type(nearest_neighbors)::nn
 type(csr_matrix)::g
 real(dp)::x(4,1),q(1,1),zero(12,2)
 real(dp),allocatable::d(:,:),dense(:,:)
 integer::i
 integer,allocatable::ids(:,:)
 logical,allocatable::mask(:,:)
 x(:,1)=[0._dp,1._dp,3._dp,7._dp];q=5
 nn%n_neighbors=1;call nn%fit(x)
 call require(nn%fit_method_=='kd_tree','Source auto backend snapshot')
 call nn%kneighbors(distances=d,indices=ids)
 call require(all(ids(:,1)==[2,1,2,3]).and.all(d(:,1)==[1._dp,1._dp,2._dp,4._dp]),'implicit self exclusion')
 call nn%kneighbors(nn%x_,d,ids)
 call require(all(ids(:,1)==[1,2,3,4]).and.all(d==0),'explicit fitted data includes own samples')
 nn%metric='invalid';nn%algorithm='invalid';nn%p=-1;nn%leaf_size=-1
 call nn%kneighbors(q,d,ids)
 call require(ids(1,1)==3.and.d(1,1)==2,'queries retain fitted geometry after parameter changes')
 nn%metric='minkowski';nn%algorithm='brute';nn%p=ieee_value(0._dp,ieee_positive_inf);nn%leaf_size=30
 call nn%fit(x);call nn%kneighbors(q,d,ids)
 call require(ids(1,1)==3.and.d(1,1)==2,'infinite Minkowski uses Chebyshev geometry')
 nn%p=.5_dp;call nn%fit(x);call nn%kneighbors(q,d,ids)
 call require(ids(1,1)==3.and.abs(d(1,1)-2)<1.e-14_dp,'fractional Minkowski brute queries')
 nn%metric='l2';nn%p=2;call nn%fit(x)
 g=nn%kneighbors_graph_csr(q,mode='distance');dense=nn%kneighbors_graph(q,mode='distance')
 call require(g%nnz()==1.and.g%indices(1)==3.and.dense(1,3)==2,'CSR and dense bound graph adapters')
 zero=0;nn%metric='russellrao';nn%algorithm='auto';nn%n_neighbors=5;call nn%fit(zero)
 g=radius_neighbors_graph(nn,0._dp,metric='russellrao',include_self=.true.)
 call require(nn%fit_method_=='ball_tree'.and.g%nnz()==0,'fitted functional graph uses fitted BallTree')
 nn%algorithm='brute'
 g=radius_neighbors_graph(nn,0._dp,metric='russellrao',include_self=.true.)
 call require(g%nnz()==0,'public algorithm changes do not replace the fitted backend')
 call nn%fit(zero);mask=nn%radius_neighbors(nn%x_,0._dp)
 call require(count(mask)==12,'explicit own data retains Boolean pairwise diagonal shortcut')
 mask=nn%radius_neighbors(radius=0._dp)
 call require(count(mask)==0,'implicit radius excludes own samples')
 g=kneighbors_graph(nn,1,metric='russellrao',include_self=.true.)
 call require(g%nnz()==12.and.all(g%indices==[(i,i=1,12)]),'fitted brute KNN graph')
 print *,'NearestNeighbors backends, snapshots, implicit queries and graph adapters passed'
end program
