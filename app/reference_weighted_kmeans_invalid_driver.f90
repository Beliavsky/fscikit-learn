program reference_weighted_kmeans_invalid_driver
  use sklearn_common,only:dp
  use sklearn_cluster,only:kmeans,k_means,kmeans_result
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  use,intrinsic::ieee_arithmetic
  implicit none
  type(kmeans)::model
  type(kmeans_result)::result
  type(csr_matrix)::s
  real(dp)::x(4,2),initial(2,2),w(4),q(4,3),score
  integer,allocatable::labels(:)
  integer::mode
  character(16)::arg
  call get_command_argument(1,arg);read(arg,*)mode
  x(:,1)=[0._dp,1._dp,3._dp,4._dp];x(:,2)=x(:,1);initial=x([1,4],:);w=1;q=1
  model%n_clusters=2
  select case(mode)
  case(1)
    result=k_means(x(:0,:),2)
  case(2)
    result=k_means(x(:,:0),2)
  case(3)
    result=k_means(x,0)
  case(4)
    result=k_means(x,5)
  case(5)
    result=k_means(x,2,n_init=-1)
  case(6)
    result=k_means(x,2,max_iter=0)
  case(7)
    result=k_means(x,2,tol=ieee_value(0._dp,ieee_quiet_nan))
  case(8)
    result=k_means(x,2,tol=-1._dp)
  case(9)
    result=k_means(x,2,random_state=-1)
  case(10)
    result=k_means(x,2,init='random_bad')
  case(11)
    result=k_means(x,2,init_centers=x)
  case(12)
    initial(1,1)=ieee_value(0._dp,ieee_quiet_nan);result=k_means(x,2,init_centers=initial)
  case(13)
    result=k_means(x,2,sample_weight=w(:3))
  case(14)
    w(1)=ieee_value(0._dp,ieee_quiet_nan);result=k_means(x,2,sample_weight=w)
  case(15)
    w(1)=ieee_value(0._dp,ieee_positive_inf);result=k_means(x,2,sample_weight=w)
  case(16)
    w=0;result=k_means(x,2,sample_weight=w,init='random')
  case(17)
    w(1)=-1;result=k_means(x,2,sample_weight=w,init='random')
  case(18)
    w=0;w(1)=1;result=k_means(x,2,sample_weight=w,init='random')
  case(19)
    w=0;result=k_means(x,2,sample_weight=w)
  case(20)
    labels=model%predict(x)
  case(21)
    call model%fit(x,initial);labels=model%predict(q)
  case(22)
    call model%fit(x,initial);x(1,1)=ieee_value(0._dp,ieee_quiet_nan);labels=model%predict(x)
  case(23)
    call model%fit(x,initial);score=model%score(x,w(:3))
  case(24)
    call model%fit(x,initial);w(1)=ieee_value(0._dp,ieee_quiet_nan);score=model%score(x,w)
  case(25)
    result=k_means(s,2)
  case(26)
    s=csr_from_dense(x);s%indices(1)=0;result=k_means(s,2)
  case(27)
    s=csr_from_dense(x);s%data(1)=ieee_value(0._dp,ieee_quiet_nan);result=k_means(s,2)
  case(28)
    s=csr_from_dense(x);s%data(1)=ieee_value(0._dp,ieee_positive_inf);result=k_means(s,2)
  case(29)
    call model%fit(x,initial);s=csr_from_dense(q);labels=model%predict(s)
  case(30)
    result=k_means(x,2,algorithm='invalid')
  case(31)
    s=csr_from_dense(x(:0,:));result=k_means(s,2)
  case(32)
    s=csr_from_dense(x(:,:0));result=k_means(s,2)
  case(33)
    call model%fit(x,initial);x(1,1)=ieee_value(0._dp,ieee_positive_inf);labels=model%predict(x)
  case(34)
    initial(1,1)=ieee_value(0._dp,ieee_positive_inf);result=k_means(x,2,init_centers=initial)
  case(35)
    result=k_means(x,2,tol=ieee_value(0._dp,ieee_positive_inf))
  end select
  print *,'guard unexpectedly returned'
end program
