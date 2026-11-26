program reference_mixture_invalid_driver
 use sklearn_mixture,only:gaussian_mixture
 use sklearn_common,only:dp
 use,intrinsic::ieee_arithmetic
 implicit none
 type(gaussian_mixture)::model
 real(dp)::x(12,2),nan
 real(dp),allocatable::scores(:),samples(:,:)
 integer,allocatable::labels(:)
 integer::i,case_id
 character(24)::arg
 call get_command_argument(1,arg);read(arg,*)case_id
 do i=1,12
  x(i,:)=[sin(real(i,dp)),cos(2._dp*i)]
 end do
 nan=ieee_value(0._dp,ieee_quiet_nan);model%n_components=2;model%max_iter=3
 select case(case_id)
 case(1);model%init_params='invalid'
 case(2);model%n_components=0
 case(3);model%n_components=13
 case(4);model%max_iter=-1
 case(5);model%n_init=0
 case(6);model%tol=-1
 case(7);model%tol=nan
 case(8);model%reg_covar=-1
 case(9);model%reg_covar=nan
 case(10);model%random_state=-1
 case(11);model%weights_init=[1._dp]
 case(12);model%weights_init=[-.1_dp,1.1_dp]
 case(13);model%weights_init=[.2_dp,.2_dp]
 case(14);model%weights_init=[nan,.5_dp]
 case(15);allocate(model%means_init(1,2));model%means_init=0
 case(16);allocate(model%means_init(2,2));model%means_init=nan
 case(17);allocate(model%precisions_init(1,1,2));model%precisions_init=1
 case(18:21,28,32)
  allocate(model%precisions_init(2,2,2));model%precisions_init=0
  model%precisions_init(1,1,:)=1;model%precisions_init(2,2,:)=1
  if(case_id==18)model%precisions_init(1,1,1)=-1
  if(case_id==19)model%precisions_init(1,2,1)=.5_dp
  if(case_id==20)then
   model%covariance_type='diag';model%precisions_init(1,1,1)=0
  end if
  if(case_id==21)model%precisions_init(1,1,1)=nan
  if(case_id==28)model%max_iter=0
  if(case_id==32)then
   model%covariance_type='diag';model%precisions_init(1,2,1)=.5_dp
  end if
 case(22);call model%fit(x(:1,:));stop
 case(23);call model%fit(x(:,:0));stop
 case(24);scores=model%score_samples(x);stop
 case(31);model%tol=ieee_value(0._dp,ieee_positive_inf)
 end select
 call model%fit(x)
 select case(case_id)
 case(25);scores=model%score_samples(x(:,:1))
 case(26);x(1,1)=nan;scores=model%score_samples(x)
 case(27);call model%sample(0,samples,labels)
 case(29);model%warm_start=.true.;model%n_components=3;call model%fit(x)
 case(30);model%warm_start=.true.;model%covariance_type='diag';call model%fit(x)
 end select
end program
