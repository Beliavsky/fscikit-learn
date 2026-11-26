program reference_nearest_neighbors_invalid_driver
 use sklearn_common,only:dp
 use sklearn_neighbors,only:nearest_neighbors
 use sklearn_sparse,only:csr_matrix
 use,intrinsic::ieee_arithmetic
 implicit none
 type(nearest_neighbors)::nn
 type(csr_matrix)::g
 real(dp)::x(4,2),q(1,2),square(4,4)
 real(dp),allocatable::d(:,:)
 integer,allocatable::ids(:,:)
 character(40)::argument
 x=reshape([0._dp,1._dp,2._dp,3._dp,1._dp,2._dp,3._dp,4._dp],[4,2]);q=0;square=0
 call get_command_argument(1,argument)
 select case(trim(argument))
 case('neighbors_zero');nn%n_neighbors=0
 case('neighbors_negative');nn%n_neighbors=-1
 case('leaf_zero');nn%leaf_size=0
 case('leaf_negative');nn%leaf_size=-1
 case('p_zero');nn%p=0
 case('p_negative');nn%p=-1
 case('p_nan');nn%p=ieee_value(0._dp,ieee_quiet_nan)
 case('algorithm');nn%algorithm='invalid'
 case('metric');nn%metric='invalid'
 case('brute_p_alias');nn%algorithm='brute';nn%metric='p'
 case('brute_infinity_alias');nn%algorithm='brute';nn%metric='infinity'
 case('kd_cosine');nn%algorithm='kd_tree';nn%metric='cosine'
 case('ball_cosine');nn%algorithm='ball_tree';nn%metric='cosine'
 case('kd_fractional');nn%algorithm='kd_tree';nn%p=.5_dp
 case('ball_fractional');nn%algorithm='ball_tree';nn%p=.5_dp
 case('radius_negative');nn%radius=-1
 case('radius_nan');nn%radius=ieee_value(0._dp,ieee_quiet_nan)
 case('fit_empty');call nn%fit(x(:0,:));error stop 'invalid accepted'
 case('fit_zero_width');call nn%fit(x(:,:0));error stop 'invalid accepted'
 case('fit_nan');x(1,1)=ieee_value(0._dp,ieee_quiet_nan)
 case('fit_infinity');x(1,1)=ieee_value(0._dp,ieee_positive_inf)
 case('fit_precomputed_width');nn%metric='precomputed'
 case('fit_precomputed_negative')
  nn%metric='precomputed';square(1,2)=-1;call nn%fit(square);error stop 'invalid accepted'
 case('unfitted_knn');call nn%kneighbors(q,d,ids);error stop 'invalid accepted'
 case('unfitted_radius');g=nn%radius_neighbors_graph(q);error stop 'invalid accepted'
 case('unfitted_graph');d=nn%kneighbors_graph(q);error stop 'invalid accepted'
 end select
 call nn%fit(x)
 select case(trim(argument))
 case('query_zero');call nn%kneighbors(q,d,ids,0)
 case('query_negative');call nn%kneighbors(q,d,ids,-1)
 case('query_limit');call nn%kneighbors(q,d,ids,5)
 case('query_self_limit');call nn%kneighbors(distances=d,indices=ids,n_neighbors=4)
 case('query_width');call nn%kneighbors(reshape([0._dp,0._dp,0._dp],[1,3]),d,ids,1)
 case('query_empty');call nn%kneighbors(q(:0,:),d,ids,1)
 case('query_nan');q(1,1)=ieee_value(0._dp,ieee_quiet_nan);call nn%kneighbors(q,d,ids,1)
 case('query_infinity');q(1,1)=ieee_value(0._dp,ieee_positive_inf);call nn%kneighbors(q,d,ids,1)
 case('knn_mode');g=nn%kneighbors_graph_csr(q,mode='invalid',n_neighbors=1)
 case('radius_mode');g=nn%radius_neighbors_graph(q,mode='invalid')
 case('brute_negative_radius');nn%algorithm='brute';call nn%fit(x);g=nn%radius_neighbors_graph(q,-1._dp)
 end select
 error stop 'invalid accepted'
end program
