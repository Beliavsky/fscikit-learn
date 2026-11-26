program reference_neighbor_regression_invalid_driver
 use sklearn_common,only:dp
 use sklearn_neighbors,only:kneighbors_regressor,radius_neighbors_regressor
 use,intrinsic::ieee_arithmetic
 implicit none
 type(kneighbors_regressor)::knn
 type(radius_neighbors_regressor)::rad
 real(dp)::x(12,2),y(12,2),nan,value
 real(dp),allocatable::matrix(:,:),vector(:)
 integer::i,which
 character(24)::arg
 call get_command_argument(1,arg);read(arg,*)which
 do i=1,12
  x(i,:)=[sin(real(i,dp)),cos(.7_dp*i)];y(i,:)=[real(i,dp),-real(i,dp)]
 end do
 nan=ieee_value(0._dp,ieee_quiet_nan);knn%n_neighbors=3
 select case(which)
 case(1);knn%n_neighbors=0
 case(2);knn%weights='invalid'
 case(3);knn%metric='invalid'
 case(4);knn%p=0
 case(5);knn%p=nan
 case(6);knn%metric='precomputed'
 case(7);knn%metric='precomputed';call knn%fit(reshape([-1._dp,0._dp,0._dp,1._dp],[2,2]),y(:2,:));stop
 case(8);call knn%fit(x,y(:11,:));stop
 case(9);call knn%fit(x,y(:,:0));stop
 case(10);y(1,1)=nan
 case(11);x(1,1)=nan
 case(12);call knn%fit(x(:,:0),y);stop
 case(13);matrix=knn%predict_matrix(x);stop
 case(16);knn%n_neighbors=13
 case(17);knn%n_neighbors=12
 case(21);rad%radius=-1;call rad%fit(x,y);stop
 case(22);rad%radius=nan;call rad%fit(x,y);stop
 case(24);call knn%fit(x(:0,:),y(:0,:));stop
 case(29);knn%p=-1;knn%metric='euclidean'
 end select
 call knn%fit(x,y)
 select case(which)
 case(14);matrix=knn%predict_matrix(x(:,:1))
 case(15);x(1,1)=nan;matrix=knn%predict_matrix(x)
 case(16);matrix=knn%predict_matrix(x)
 case(17);matrix=knn%predict_matrix()
 case(18);vector=knn%predict(x)
 case(19);value=knn%score(x,y(:,1))
 case(20);rad%radius=0;call rad%fit(x,y);value=rad%score(x+10,y)
 case(23)
  knn%metric='precomputed';call knn%fit(reshape([0._dp,1._dp,1._dp,0._dp],[2,2]),y(:2,:))
  matrix=knn%predict_matrix(reshape([-1._dp,1._dp],[1,2]))
 case(25);value=knn%score(x,y,sample_weight=[1._dp])
 case(26);matrix=knn%predict_matrix(x(:0,:))
 case(27);call rad%fit(x,y);matrix=rad%predict_matrix(x(:0,:))
 case(28);call rad%fit(x,y);value=rad%score(x,y,sample_weight=[1._dp])
 end select
end program
