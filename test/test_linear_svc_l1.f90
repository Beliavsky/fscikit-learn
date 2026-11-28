program test_linear_svc_l1
  use sklearn_common,only:dp,require
  use sklearn_linear_svm,only:linear_svc
  use sklearn_sparse,only:csr_matrix,csr_from_arrays,csr_from_dense
  implicit none
  type(linear_svc)::model
  type(csr_matrix)::x,original
  real(dp)::pair(2,1),zero(2,1),three(3,2)
  real(dp),allocatable::coef(:,:),data(:)
  integer,allocatable::ptr(:),indices(:),labels(:),before(:),after(:)
  integer::i,n
  character(32)::mode
  call get_command_argument(1,mode)
  model%penalty='l1'
  if(trim(mode)=='memory')then
    allocate(ptr(129),indices(256),data(256),labels(128))
    do i=1,128
      ptr(i)=2*i-1;indices(2*i-1:2*i)=[1,i+1]
      data(2*i-1:2*i)=[real(mod(i,3)-1,dp),1._dp];labels(i)=mod(i,3)
    end do
    ptr(129)=257;x=csr_from_arrays(128,200000,ptr,indices,data)
    call model%fit(x,labels)
    call require(.not.allocated(model%dual_variables_),'L1 primal alpha remains unallocated')
    call require(all(shape(model%coef_)==[3,200000]).and.model%converged_,'large L1 classifier fit')
    print *,'Large L1 LinearSVC memory case: PASS';stop
  end if
  pair(:,1)=[-1._dp,1._dp];zero=0;x=csr_from_dense(pair)
  select case(trim(mode))
  case('penalty');model%penalty='xx';call model%fit(x,[-3,9])
  case('dual');model%dual='true';call model%fit(x,[-3,9])
  case('loss');model%loss='hinge';call model%fit(x,[-3,9])
  end select
  call require(len_trim(mode)==0,'L1 classifier rejection unexpectedly accepted input')
  call random_seed(size=n);allocate(before(n),after(n));call random_seed(get=before)
  model%fit_intercept=.false.;model%tol=1.e-10_dp;model%store_dual_variables=.true.
  call model%fit(x,[-3,9])
  call require(abs(model%coef_(1,1)-.75_dp)<1.e-12_dp,'analytic L1 squared-hinge coefficient')
  call require(abs(model%objective_(1)-.875_dp)<1.e-12_dp.and.model%gradient_norm_(1)<1.e-12_dp,'L1 objective and subgradient')
  call require(.not.allocated(model%dual_variables_),'L1 ignores opt-in alpha for a primal fit')
  model%c=.1_dp;call model%fit(pair,[-3,9])
  call require(all(model%coef_==0).and.abs(model%objective_(1)-.2_dp)<1.e-12_dp,'L1 exact zero coefficient')
  model%c=1;model%fit_intercept=.true.;model%intercept_scaling=2
  call model%fit(zero,[-3,9],sample_weight=[1._dp,3._dp])
  call require(abs(model%intercept_(1)-7._dp/16)<1.e-12_dp,'scaled intercept has L1 penalty')
  model%fit_intercept=.false.;x=csr_from_arrays(2,1,[1,3,5],[1,1,1,1],[-.25_dp,-.75_dp,.25_dp,.75_dp]);original=x
  call model%fit(x,[-3,9]);coef=model%coef_
  call require(abs(coef(1,1)-.75_dp)<1.e-9_dp,'L1 duplicate column entries')
  call require(all(x%data==original%data).and.all(x%indices==original%indices),'owned L1 transpose preserves input')
  three(:,1)=[-1._dp,0._dp,1._dp];three(:,2)=[1._dp,-1._dp,1._dp]
  call model%fit(three,[-4,7,21]);coef=model%coef_;call model%fit(three,[-4,7,21])
  call require(all(model%coef_==coef),'L1 OVR streams reset on refit')
  model%multi_class='crammer_singer';model%loss='hinge';model%dual='true'
  call model%fit(pair,[-3,9]);call require(abs(model%coef_(1,1)-1._dp)<1.e-10_dp,'joint solver ignores valid penalty loss and dual')
  model%multi_class='ovr';model%loss='squared_hinge';model%dual='auto';call model%fit(pair,[-3,9])
  call require(.not.allocated(model%dual_variables_).and.abs(model%coef_(1,1)-.75_dp)<1.e-12_dp,'L1 refit clears joint dual state')
  model%penalty='l2';call model%fit(pair,[-3,9])
  call require(abs(model%coef_(1,1)-.8_dp)<1.e-12_dp,'refit restores L2 primal objective')
  call random_seed(get=after);call require(all(before==after),'L1 solver preserves intrinsic RNG')
  print *,'L1 LinearSVC analytic optima, sparse storage, refits and RNG: PASS'
end program
