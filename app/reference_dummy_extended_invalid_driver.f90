program reference_dummy_extended_invalid_driver
  use sklearn_common,only:dp,require
  use sklearn_dummy
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  use,intrinsic::ieee_arithmetic
  implicit none
  type(dummy_classifier)::classifier
  type(dummy_regressor)::regressor
  type(csr_matrix)::sparse
  real(dp)::x(3,1),y(3),weights(3),nan
  real(dp),allocatable::pred(:),proba(:,:)
  integer::labels(3),which
  integer,allocatable::actual(:)
  character(16)::arg
  call get_command_argument(1,arg);read(arg,*)which
  x=0;y=[1._dp,2._dp,3._dp];labels=[-7,4,19];weights=1;nan=ieee_value(0._dp,ieee_quiet_nan)
  select case(which)
  case(1);actual=classifier%predict(x)
  case(2);pred=regressor%predict(x)
  case(3);classifier%strategy='unknown';call classifier%fit(x,labels)
  case(4);regressor%strategy='unknown';call regressor%fit(x,y)
  case(5);classifier%random_state=-1;call classifier%fit(x,labels)
  case(6);call classifier%fit(x,labels,[1._dp])
  case(7);call regressor%fit(x,y,[1._dp])
  case(8);weights(1)=nan;call classifier%fit(x,labels,weights)
  case(9);weights(1)=ieee_value(0._dp,ieee_positive_inf);call regressor%fit(x,y,weights)
  case(10);y(2)=nan;call regressor%fit(x,y)
  case(11);call classifier%fit(x,labels(:2))
  case(12);call regressor%fit(x,y(:2))
  case(13);regressor%quantile=-1;call regressor%fit(x,y)
  case(14);regressor%quantile=nan;call regressor%fit(x,y)
  case(15);regressor%strategy='constant';regressor%constant=nan;call regressor%fit(x,y)
  case(16);classifier%strategy='constant';classifier%constant=99;call classifier%fit(x,labels)
  case(17);classifier%strategy='constant';call classifier%fit(x,spread(labels,2,2))
  case(18);regressor%strategy='constant';call regressor%fit(x,spread(y,2,2))
  case(19);classifier%strategy='constant';classifier%constant_values=[-7];call classifier%fit(x,spread(labels,2,2))
  case(20);regressor%strategy='constant';regressor%constant_values=[1._dp];call regressor%fit(x,spread(y,2,2))
  case(21);call classifier%fit(x,spread(labels,2,2));actual=classifier%predict(x)
  case(22);call regressor%fit(x,spread(y,2,2));pred=regressor%predict(x)
  case(23);call classifier%fit(x,spread(labels,2,2));proba=classifier%predict_proba(x)
  case(24);call regressor%fit(x,y,[0._dp,0._dp,0._dp])
  case(25)
    classifier%strategy='stratified';call classifier%fit(x,labels,[0._dp,0._dp,0._dp])
    proba=classifier%predict_proba(x)
  case(26)
    classifier%strategy='stratified';call classifier%fit(x,labels,[-1._dp,1._dp,1._dp])
    actual=classifier%predict(x)
  case(27);call regressor%fit(x,y,1._dp)
  case(28);sparse=csr_from_dense(x);sparse%indptr(1)=0;call classifier%fit(sparse,labels)
  case default;error stop 'unknown dummy guard'
  end select
  print *,'dummy guard unexpectedly returned'
end program
