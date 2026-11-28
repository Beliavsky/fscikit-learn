program test_linear_svr_dual
  use sklearn_common,only:dp,require
  use sklearn_linear_svm,only:linear_svr
  use sklearn_sparse,only:csr_matrix,csr_from_arrays,csr_from_dense
  use,intrinsic::ieee_exceptions
  implicit none
  type(linear_svr)::model,repeat
  type(csr_matrix)::x
  real(dp)::pair(2,1),target(2)
  real(dp),allocatable::before_coef(:),targets(:),data(:),values(:)
  integer,allocatable::before(:),after(:),ptr(:),indices(:)
  integer::n,i
  logical::modes(size(ieee_all)),after_modes(size(ieee_all))
  character(32)::mode
  call get_command_argument(1,mode)
  if(trim(mode)=='memory')then
    allocate(ptr(129),indices(256),data(256),targets(128))
    do i=1,128
      ptr(i)=2*i-1;indices(2*i-1:2*i)=[1,i+1]
      data(2*i-1:2*i)=[real(mod(i,3)-1,dp),1._dp];targets(i)=sin(real(i,dp))
    end do
    ptr(129)=257;x=csr_from_arrays(128,200000,ptr,indices,data)
    model%epsilon=.1_dp;repeat%loss='squared_epsilon_insensitive';repeat%dual='true'
    call model%fit(x,targets);call repeat%fit(x,targets)
    call require(model%converged_.and.repeat%converged_,'large dual regressions converge')
    values=model%predict(x);values=repeat%predict(x)
    call require(size(model%coef_)==200000.and.size(values)==128,'large dual regression dimensions')
    call require(allocated(model%dual_variables_).and.allocated(repeat%dual_variables_), &
      'large dual regressions retain dual certificates')
    print *,'Large dual LinearSVR memory case: PASS';stop
  end if
  pair(:,1)=[-1._dp,1._dp];target=[-2._dp,2._dp];x=csr_from_dense(pair)
  select case(trim(mode))
  case('loss');model%loss='invalid';call model%fit(x,target)
  case('dual');model%dual='invalid';call model%fit(x,target)
  case('combination');model%dual='false';call model%fit(x,target)
  case('seed');model%random_state=-1;call model%fit(x,target)
  end select
  call require(len_trim(mode)==0,'dual LinearSVR guard unexpectedly accepted input')
  call require(model%loss=='epsilon_insensitive'.and.model%dual=='auto','sklearn default loss and dual selection')
  call random_seed(size=n);allocate(before(n),after(n));call random_seed(get=before)
  call ieee_get_halting_mode(ieee_all,modes)
  model%fit_intercept=.false.;model%epsilon=.5_dp;model%tol=1.e-12_dp
  call model%fit(x,target)
  call require(abs(model%coef_(1)-1.5_dp)<1.e-12_dp,'analytic unsquared epsilon-insensitive optimum')
  call require(abs(model%objective_-1.125_dp)<1.e-12_dp,'unsquared primal objective')
  call require(abs(model%duality_gap_)<1.e-11_dp.and.model%kkt_violation_<1.e-11_dp,'unsquared dual certificate')
  call require(maxval(abs(model%dual_variables_))<=model%c,'unsquared dual box constraints')
  before_coef=model%coef_;call model%fit(pair,target)
  call require(all(before_coef==model%coef_),'dense and CSR use the same seeded dual solver')
  model%loss='squared_epsilon_insensitive';model%dual='true';call model%fit(x,target)
  call require(abs(model%coef_(1)-1.2_dp)<1.e-11_dp.and.abs(model%objective_-.9_dp)<1.e-11_dp, &
    'squared dual regression optimum')
  call require(abs(model%duality_gap_)<1.e-10_dp,'squared dual certificate')
  model%dual='false';call model%fit(x,target)
  call require(.not.allocated(model%dual_variables_).and.model%duality_gap_==0,'primal refit clears dual diagnostics')
  model%loss='epsilon_insensitive';model%dual='auto';model%max_iter=1;call model%fit(x,target)
  call require(model%n_iter_==1.and..not.model%converged_,'dual iteration cap follows Source warning status')
  model%max_iter=1000;x=csr_from_arrays(2,3,[1,1,1],[integer::],[real(dp)::])
  call model%fit(x,target)
  call require(all(model%coef_==0).and.abs(model%duality_gap_)<1.e-12_dp,'zero-norm rows use finite box endpoints')
  model%fit_intercept=.true.;model%intercept_scaling=2
  call model%fit(x,[2._dp,2._dp],sample_weight=[0._dp,1._dp])
  call require(model%dual_variables_(1)==0.and.model%converged_,'zero-weight removal and original-row dual mapping')
  before_coef=model%coef_;call model%fit(x,[2._dp,2._dp],sample_weight=[0._dp,1._dp])
  call require(all(before_coef==model%coef_),'dual refits reset the local RNG')
  call random_seed(get=after);call require(all(before==after),'dual fits preserve intrinsic RNG')
  call ieee_get_halting_mode(ieee_all,after_modes)
  call require(all(modes.eqv.after_modes),'dual fits restore IEEE halting modes')
  print *,'Dual LinearSVR losses, defaults, certificates, seeded refits and zero rows: PASS'
end program
