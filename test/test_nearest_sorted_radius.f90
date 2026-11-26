program test_nearest_sorted_radius
 use sklearn_common,only:dp,require
 use sklearn_neighbors,only:nearest_neighbors
 use sklearn_sparse,only:csr_matrix
 use,intrinsic::ieee_arithmetic
 implicit none
 type(nearest_neighbors)::nn
 type(csr_matrix)::g,unsorted
 real(dp)::x(3,1),q(1,1),roots(65,2),origin(1,2),cosine(65,16),infinity
 real(dp),allocatable::d(:,:)
 integer,allocatable::ids(:,:)
 integer::i
 character(9),parameter::algorithms(4)=[character(9)::'auto','brute','kd_tree','ball_tree']
 infinity=ieee_value(0._dp,ieee_positive_inf)
 x(:,1)=[3._dp,1._dp,2._dp];q=0;nn%n_neighbors=1
 do i=1,size(algorithms)
  nn%algorithm=trim(algorithms(i));nn%metric='euclidean';call nn%fit(x)
  g=nn%radius_neighbors_graph(q,infinity,'distance',sort_results=.true.)
  call require(all(g%indices==[2,3,1]).and.all(g%data==[1._dp,2._dp,3._dp]),'sorted distance graph')
  g=nn%radius_neighbors_graph(q,infinity,'connectivity',sort_results=.true.)
  unsorted=nn%radius_neighbors_graph(q,infinity,'connectivity',sort_results=.false.)
  call require(all(g%indices==unsorted%indices).and.all(g%data==1),'connectivity ignores sorting like Source')
  g=nn%radius_neighbors_graph(radius=0._dp,mode='distance',sort_results=.true.)
  call require(g%nnz()==0,'sorted implicit graph removes own samples')
  g=nn%radius_neighbors_graph(nn%x_,0._dp,'distance',sort_results=.true.)
  call require(g%nnz()==3.and.all(g%data==0),'explicit own data retains zero-distance edges')
 end do
 roots=0;roots(:,1)=1;roots(1:65:2,2)=sqrt(epsilon(1._dp));origin=0
 nn%algorithm='brute';call nn%fit(roots)
 g=nn%radius_neighbors_graph(origin,1._dp,'distance',sort_results=.true.)
 call require(g%nnz()==32.and.all(mod(g%indices,2)==0),'radius filters reduced distances before rounding')
 g=nn%radius_neighbors_graph(origin,1._dp+epsilon(1._dp),'distance',sort_results=.true.)
 call require(g%nnz()==65.and.all(g%data==1),'rounded square roots remain tied')
 call require(all(mod(g%indices(:32),2)==0).and.all(mod(g%indices(33:),2)==1), &
  'numeric radius sorting uses reduced keys before square roots')
 ! Two integer rows isolate the NumPy SYRK tail-tile rounding regression.
 ! Actual sklearn 1.6.1 / NumPy 2.2.6 cosine distance: 0.7491273969978728.
 cosine=0
 cosine(5,:)=[0,1,0,-2,0,0,1,0,0,-1,2,0,0,0,0,0]
 cosine(65,:)=[-1,1,1,0,0,0,1,0,-1,-1,0,1,1,0,1,2]
 nn%metric='cosine';call nn%fit(cosine)
 call nn%kneighbors(distances=d,indices=ids,n_neighbors=1)
 call require(ids(5,1)==65.and.d(5,1)==0.7491273969978728_dp,'cosine self-product SYRK rounding')
 print *,'Sorted NearestNeighbors radius graphs and cosine self-products passed'
end program
