program test_linear_svc_dual
  use sklearn_common,only:dp,require
  use sklearn_linear_svm,only:linear_svc
  use sklearn_sparse,only:csr_matrix,csr_from_arrays,csr_from_dense
  use sklearn_parameter_space,only:parameter_set,parameter_item,param
  use sklearn_search_adapters,only:set_search_params
  use sklearn_base_dispatch,only:seed_base
  implicit none
  type(linear_svc)::model,other
  type(csr_matrix)::x
  type(parameter_set)::parameters
  real(dp)::pair(2,1),data3(3,2)
  real(dp),allocatable::before_coef(:,:),data(:),scores(:,:)
  integer,allocatable::ptr(:),indices(:),labels(:),predicted(:),before(:),after(:)
  integer::i,n
  character(32)::mode
  call get_command_argument(1,mode)
  if(trim(mode)=='memory')then
    allocate(ptr(129),indices(256),data(256),labels(128))
    do i=1,128
      ptr(i)=2*i-1;indices(2*i-1:2*i)=[1,i+1]
      data(2*i-1:2*i)=[real(mod(i,3)-1,dp),1._dp];labels(i)=mod(i,3)
    end do
    ptr(129)=257;x=csr_from_arrays(128,200000,ptr,indices,data)
    model%loss='hinge';other%loss='squared_hinge';other%dual='true'
    call model%fit(x,labels);call other%fit(x,labels)
    call require(model%converged_.and.other%converged_,'large classification dual solvers converge')
    call require(.not.allocated(model%dual_variables_),'dual variable storage is opt-in')
    predicted=model%predict(x);predicted=other%predict(x)
    call require(size(model%coef_,2)==200000.and.size(predicted)==128,'large dual classifier dimensions')
    print *,'Large dual LinearSVC memory case: PASS';stop
  end if
  pair(:,1)=[-1._dp,1._dp];x=csr_from_dense(pair)
  select case(trim(mode))
  case('loss');model%loss='invalid';call model%fit(x,[-3,9])
  case('dual');model%dual='invalid';call model%fit(x,[-3,9])
  case('combination');model%loss='hinge';model%dual='false';call model%fit(x,[-3,9])
  case('seed');model%random_state=-1;call model%fit(x,[-3,9])
  end select
  call require(len_trim(mode)==0,'LinearSVC dual guard unexpectedly accepted input')
  call require(model%loss=='squared_hinge'.and.model%dual=='auto','sklearn classifier defaults')
  call random_seed(size=n);allocate(before(n),after(n));call random_seed(get=before)
  model%loss='hinge';model%dual='true';model%fit_intercept=.false.;model%tol=1.e-10_dp
  model%store_dual_variables=.true.;call model%fit(x,[-3,9])
  call require(abs(model%coef_(1,1)-1)<1.e-9_dp,'analytic hinge classifier optimum')
  call require(abs(model%objective_(1)-.5_dp)<1.e-9_dp.and.abs(model%duality_gap_(1))<1.e-8_dp, &
    'hinge classifier primal dual objectives')
  call require(all(model%dual_variables_>=0).and.all(model%dual_variables_<=model%c),'hinge dual bounds')
  before_coef=model%coef_;call model%fit(pair,[-3,9])
  call require(all(before_coef==model%coef_),'dense and CSR classifier dual fits share the seed')
  model%loss='squared_hinge';model%tol=1.e-8_dp;call model%fit(x,[-3,9])
  call require(abs(model%coef_(1,1)-.8_dp)<1.e-8_dp,'analytic squared-hinge dual optimum')
  call require(abs(model%duality_gap_(1))<1.e-8_dp,'squared-hinge dual certificate')
  model%dual='false';call model%fit(x,[-3,9])
  call require(.not.allocated(model%dual_variables_).and.all(model%duality_gap_==0),'primal refit clears dual state')
  model%loss='hinge';model%dual='auto';model%max_iter=1;call model%fit(x,[-3,9])
  call require(model%n_iter_==1.and..not.model%converged_,'classification dual cap status')
  model%max_iter=1000;x=csr_from_arrays(2,3,[1,1,1],[integer::],[real(dp)::])
  call model%fit(x,[-3,9])
  call require(all(model%coef_==0).and.abs(model%duality_gap_(1))<1.e-12_dp,'zero-norm hinge row endpoints')
  data3(:,1)=[-1._dp,0._dp,1._dp];data3(:,2)=[1._dp,-1._dp,1._dp]
  x=csr_from_dense(data3);model%fit_intercept=.true.;call model%fit(x,[-4,7,21])
  before_coef=model%coef_;call model%fit(x,[-4,7,21])
  call require(all(before_coef==model%coef_).and.size(model%coef_,1)==3,'multiclass streams reset on refit')
  call model%fit(x,[-4,7,21],sample_weight=[0._dp,1._dp,1._dp])
  call require(size(model%classes_)==3.and.size(model%coef_,1)==1,'Source filtered active-class coefficient shape')
  call require(all(model%dual_variables_(1,:)==0),'zero-weight original-row alpha mapping')
  call random_seed(get=after);call require(all(before==after),'classifier dual fits preserve intrinsic RNG')
  parameters%items=[parameter_item('loss',param('hinge')),parameter_item('dual',param(.true.)), &
    parameter_item('random_state',param(42))]
  call set_search_params(other,parameters)
  call require(other%loss=='hinge'.and.other%dual=='true'.and.other%random_state==42,'classifier search controls')
  call seed_base(other,7);call seed_base(model,7)
  call require(other%random_state==model%random_state.and.other%random_state/=42,'ensemble classifier seeding')
  print *,'LinearSVC losses, solver selection, dual certificates and grouped streams: PASS'
end program
