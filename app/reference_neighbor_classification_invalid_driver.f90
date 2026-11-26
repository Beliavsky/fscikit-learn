program reference_neighbor_classification_invalid_driver
 use sklearn_common,only:dp
 use sklearn_neighbors,only:kneighbors_classifier,neighbor_probability
 use,intrinsic::ieee_arithmetic
 implicit none
 type(kneighbors_classifier)::model
 type(neighbor_probability),allocatable::prob(:)
 real(dp)::x(12,2),d(12,12),score
 real(dp),allocatable::values(:,:)
 integer::y(12,3),i,j,which
 integer,allocatable::prediction(:),matrix(:,:)
 character(16)::arg
 call get_command_argument(1,arg);read(arg,*)which
 do i=1,12
  x(i,:)=[sin(real(i,dp)),cos(.7_dp*i)];y(i,:)=[mod(i,2),mod(i+1,2),mod(i,2)]
 end do
 model%n_neighbors=3
 select case(which)
 case(1);model%n_neighbors=0
 case(2);model%weights='invalid'
 case(3);model%metric='invalid'
 case(4);model%p=0
 case(5);model%p=ieee_value(0._dp,ieee_quiet_nan)
 case(6);model%metric='precomputed'
 case(7)
  d=0;d(1,1)=-1;model%metric='precomputed';call model%fit(d,y);stop
 case(8);call model%fit(x,y(:11,:));stop
 case(9);call model%fit(x,y(:,:0));stop
 case(10);x(1,1)=ieee_value(0._dp,ieee_quiet_nan)
 case(11);prediction=model%predict(x);stop
 case(14);model%n_neighbors=13
 case(15);model%n_neighbors=12
 case(18);y(:,2)=[(mod(i,3),i=1,12)]
 end select
 if(which==16.or.which==17.or.which==18)then
  call model%fit(x,y)
 else
  call model%fit(x,y(:,1))
 end if
 select case(which)
 case(12);prediction=model%predict(x(:,:1))
 case(13);x(1,1)=ieee_value(0._dp,ieee_quiet_nan);prediction=model%predict(x)
 case(14);prediction=model%predict(x)
 case(15);prediction=model%predict()
 case(16);prediction=model%predict(x)
 case(17);values=model%predict_proba(x)
 case(18);score=model%score(x,y)
 case(19);score=model%score(x,y(:,1),sample_weight=[1._dp])
 case(20);prediction=model%predict(x(:0,:))
 case(21)
  do j=1,12
   d(:,j)=abs(x(:,1)-x(j,1))
  end do
  model%metric='precomputed';call model%fit(d,y(:,1));d(1,1)=-1;prediction=model%predict(d)
 end select
 print *,'unexpected acceptance',which
end program
