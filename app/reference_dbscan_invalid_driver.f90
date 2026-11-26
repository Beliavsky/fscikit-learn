program reference_dbscan_invalid_driver
 use sklearn_dbscan,only:dbscan
 use sklearn_common,only:dp,require
 use sklearn_sparse,only:csr_matrix,csr_from_dense
 use,intrinsic::ieee_arithmetic
 implicit none
 type(dbscan)::model
 type(csr_matrix)::sparse
 real(dp)::x(3,2),bad,inf
 character(40)::name
 call get_command_argument(1,name)
 x=1;bad=ieee_value(0._dp,ieee_quiet_nan);inf=ieee_value(0._dp,ieee_positive_inf)
 select case(trim(name))
 case('eps_zero');model%eps=0
 case('eps_negative');model%eps=-1
 case('eps_nan');model%eps=bad
 case('min_zero');model%min_samples=0
 case('leaf_zero');model%leaf_size=0
 case('p_negative');model%p=-1
 case('p_nan');model%p=bad
 case('algorithm');model%algorithm='unknown'
 case('metric');model%metric='unknown'
 case('weights_nan');call model%fit(x,[bad,1._dp,1._dp])
 case('weights_inf');call model%fit(x,[inf,1._dp,1._dp])
 case('weights_count');call model%fit(x,[1._dp])
 case('input_nan');x(1,1)=bad
 case('input_inf');x(1,1)=inf
 case('input_empty');call model%fit(x(:0,:))
 case('precomputed_shape');model%metric='precomputed'
 case('precomputed_negative')
  model%metric='precomputed';call model%fit(reshape([-1._dp],[1,1]))
 case('sparse_negative')
  model%metric='precomputed';sparse=csr_from_dense(reshape([-1._dp],[1,1]));call model%fit(sparse)
 case('sparse_empty');sparse=csr_from_dense(x(:0,:));call model%fit(sparse)
 case default;call require(.false.,'unknown DBSCAN invalid case')
 end select
 call model%fit(x)
 call require(.false.,'invalid DBSCAN input was accepted')
end program
