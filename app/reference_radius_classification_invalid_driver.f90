program reference_radius_classification_invalid_driver
 use sklearn_common,only:dp
 use sklearn_neighbors,only:radius_neighbors_classifier
 use,intrinsic::ieee_arithmetic
 implicit none
 type(radius_neighbors_classifier)::model
 real(dp)::x(12,2),d(12,12),score
 real(dp),allocatable::values(:,:)
 integer::y(12,3),i,j,which
 integer,allocatable::prediction(:)
 character(16)::arg
 call get_command_argument(1,arg);read(arg,*)which
 do i=1,12
  x(i,:)=[sin(real(i,dp)),cos(.7_dp*i)];y(i,:)=[mod(i,2),mod(i+1,2),mod(i,2)]
 end do
 model%radius=1;model%outlier_mode='error'
 select case(which)
 case(1);model%radius=-1
 case(2);model%radius=ieee_value(0._dp,ieee_quiet_nan)
 case(3);model%weights='invalid'
 case(4);model%metric='invalid'
 case(5);model%p=0
 case(6);model%p=ieee_value(0._dp,ieee_quiet_nan)
 case(7);model%metric='precomputed'
 case(8)
  d=0;d(1,1)=-1;model%metric='precomputed';call model%fit(d,y);stop
 case(9);call model%fit(x,y(:11,:));stop
 case(10);call model%fit(x,y(:,:0));stop
 case(11);x(1,1)=ieee_value(0._dp,ieee_quiet_nan)
 case(12);prediction=model%predict(x);stop
 case(18);y(:,2)=[(mod(i,3),i=1,12)]
 case(21);model%outlier_mode='label';model%outlier_labels=[0,1]
 case(22);model%radius=0
 case(23);call model%fit(x(:0,:),y(:0,:));stop
 case(24);call model%fit(x(:,:0),y);stop
 case(25);model%outlier_mode='invalid'
 case(26);model%metric='euclidean';model%p=-1
 end select
 if(any(which==[16,17,18,21]))then
  call model%fit(x,y)
 else
  call model%fit(x,y(:,1))
 end if
 select case(which)
 case(13);prediction=model%predict(x(:,:1))
 case(14);x(1,1)=ieee_value(0._dp,ieee_quiet_nan);prediction=model%predict(x)
 case(15);prediction=model%predict(x(:0,:))
 case(16);prediction=model%predict(x)
 case(17);values=model%predict_proba(x)
 case(18);score=model%score(x,y)
 case(19);score=model%score(x,y(:,1),sample_weight=[1._dp])
 case(20)
  do j=1,12
   d(:,j)=abs(x(:,1)-x(j,1))
  end do
  model%metric='precomputed';call model%fit(d,y(:,1));d(1,1)=-1;prediction=model%predict(d)
 case(22);prediction=model%predict(x+100)
 end select
 print *,'unexpected acceptance',which
end program
