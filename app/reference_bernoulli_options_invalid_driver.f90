program reference_bernoulli_options_invalid_driver
  use sklearn_common,only:dp
  use sklearn_naive_bayes,only:bernoulli_nb
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  use,intrinsic::ieee_arithmetic
  implicit none
  type(bernoulli_nb)::model
  type(csr_matrix)::sparse
  real(dp)::x(3,1),w(3),nan
  real(dp),allocatable::values(:,:)
  integer::y(3),which
  character(16)::arg
  call get_command_argument(1,arg);read(arg,*)which
  x(:,1)=[-.5_dp,.2_dp,.7_dp];y=[-7,4,4];w=1;nan=ieee_value(0._dp,ieee_quiet_nan)
  model%binarize=.false.
  select case(which)
  case(1);values=model%predict_proba(x)
  case(2);call model%partial_fit(x,y)
  case(3);x(1,1)=nan;call model%fit(x,y)
  case(4);x(1,1)=ieee_value(0._dp,ieee_positive_inf);call model%fit(x,y)
  case(5);w(1)=-1;call model%fit(x,y,w)
  case(6);w(1)=nan;call model%fit(x,y,w)
  case(7);call model%fit(x,y,[1._dp])
  case(8);call model%fit(x,y(:2))
  case(9);model%alpha=-1;call model%fit(x,y)
  case(10);model%alpha=nan;call model%fit(x,y)
  case(11);model%binarize=.true.;model%binarize_threshold=-1;call model%fit(x,y)
  case(12);model%binarize=.true.;model%binarize_threshold=nan;call model%fit(x,y)
  case(13);model%binarize=.true.;model%binarize_threshold=-1;sparse=csr_from_dense(x);call model%fit(sparse,y)
  case(14);call model%fit(x,y);values=model%predict_proba(spread(x(:,1),2,2))
  case(15);call model%fit(x,y);x(1,1)=nan;values=model%predict_proba(x)
  case(16);sparse=csr_from_dense(x);sparse%indptr(1)=0;call model%fit(sparse,y)
  case(17);call model%partial_fit(x,y,[-7,4]);call model%partial_fit(x,y,[-7,4,19])
  case(18);call model%partial_fit(x,y,[-7])
  case(19);x(1,1)=-huge(1._dp);w(1)=2;call model%fit(x,y,w)
  case(20);x=-huge(1._dp);y=-7;call model%fit(x,y)
  case default;error stop 'unknown Bernoulli guard'
  end select
  print *,'Bernoulli guard unexpectedly returned'
end program
