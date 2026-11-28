program test_rbm_sparse
  use sklearn_rbm, only: bernoulli_rbm
  use sklearn_sparse, only: csr_matrix,csr_from_dense,csr_from_arrays
  use sklearn_common, only: dp,require
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf,ieee_is_nan,ieee_is_finite
  use, intrinsic :: ieee_exceptions, only: ieee_get_flag,ieee_set_flag,ieee_invalid,ieee_overflow, &
    ieee_get_halting_mode
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  type(bernoulli_rbm) :: model,dense,copy,tail,extreme,recovered
  type(csr_matrix) :: sparse,before,batch,bad
  real(dp) :: x(13,5),huge_x(13,5),one(1,1),signed_query(1,11)
  real(dp), allocatable :: h(:,:),expected(:,:),draws(:,:),scores(:),again(:)
  integer :: i,j,n,p
  logical :: invalid,overflow,halting,prior_halting
  character(32) :: mode
  call get_command_argument(1,mode)
  if(mode=='memory') then
    n=8192; p=100000; sparse%n_rows=n; sparse%n_cols=p
    allocate(sparse%indptr(n+1),sparse%indices(2*n),sparse%data(2*n))
    do i=1,n
      sparse%indptr(i)=2*i-1
      sparse%indices(2*i-1:2*i)=[mod(17*i,p)+1,mod(31*i,p)+1]
      sparse%data(2*i-1:2*i)=[.2_dp,.8_dp]
    end do
    sparse%indptr(n+1)=2*n+1
    model%n_components=3; model%batch_size=2; model%n_iter=0
    call model%fit(sparse); h=model%transform(sparse)
    batch%n_rows=2; batch%n_cols=p; batch%indptr=sparse%indptr(:3)
    batch%indices=sparse%indices(:4); batch%data=sparse%data(:4)
    call model%partial_fit(sparse); scores=model%score_samples(sparse); draws=model%gibbs(batch)
    call require(all(shape(h)==[n,3]).and.all(ieee_is_finite(h)),'large CSR RBM hidden output')
    call require(all(ieee_is_finite(scores)).and.all(scores<=0),'large CSR RBM pseudo-likelihood')
    call require(all(shape(draws)==[2,p]).and.all(draws==0.or.draws==1),'large CSR RBM dense Gibbs samples')
    print *, 'dense_input_bytes',int(n,int64)*p*8
    print *, 'Large sparse RBM memory case: PASS'
    stop
  end if
  do j=1,5
    do i=1,13
      x(i,j)=merge(1._dp,0._dp,mod(i+j,3)==0)
    end do
  end do
  x(1,:)=0; sparse=csr_from_dense(x); before=sparse
  model%n_components=3; model%batch_size=4; model%n_iter=3; model%random_state=7
  select case(mode)
  case('nan'); sparse%data(1)=ieee_value(0._dp,ieee_quiet_nan)
  case('infinity'); sparse%data(1)=ieee_value(0._dp,ieee_positive_inf)
  case('zero_rows'); sparse=csr_from_arrays(0,5,[1],[integer::],[real(dp)::])
  case('zero_cols'); sparse=csr_from_arrays(1,0,[1,1],[integer::],[real(dp)::])
  case('bad_pointer'); sparse%indptr(1)=0
  case('bad_index'); sparse%indices(1)=6
  case('unfitted'); h=model%transform(sparse)
  case('unfitted_score'); scores=model%score_samples(sparse)
  end select
  dense=model; h=model%fit_transform(sparse); expected=dense%fit_transform(x)
  if(mode=='features') then
    bad=csr_from_dense(x(:,:4)); h=model%transform(bad)
  else if(mode=='changed_components') then
    model%n_components=4; call model%partial_fit(sparse)
  end if
  if(len_trim(mode)>0) error stop 'Expected guard did not reject input'
  call require(maxval(abs(h-expected))<1.e-13_dp,'CSR/dense RBM fitting and hidden means')
  call require(maxval(abs(model%components_-dense%components_))<1.e-13_dp,'CSR/dense RBM parameters')
  copy=model; call model%partial_fit(x(:3,:)); bad=csr_from_dense(x(:3,:)); call copy%partial_fit(bad)
  call require(maxval(abs(model%components_-copy%components_))<1.e-13_dp,'mixed CSR/dense RBM continuation')
  copy=model; scores=model%score_samples(sparse); again=model%score_samples(sparse)
  call require(all(scores==again),'CSR RBM scalar-seed deterministic scores')
  draws=model%gibbs(sparse); expected=copy%gibbs(sparse)
  call require(all(draws==expected),'CSR scores preserve sampling stream')
  call require(all(sparse%data==before%data).and.all(sparse%indices==before%indices).and. &
    all(sparse%indptr==before%indptr),'CSR RBM preserves input arrays')
  model%components_=0; model%intercept_hidden_=0; model%intercept_visible_=0
  scores=model%score_samples(sparse)
  call require(maxval(abs(scores+5*log(2._dp)))<1.e-14_dp,'CSR RBM uniform pseudo-likelihood')
  one=1; tail%n_components=1; tail%n_iter=0; call tail%fit(one); tail%components_=-710
  call ieee_set_flag(ieee_invalid,.false.); call ieee_set_flag(ieee_overflow,.false.)
  call ieee_get_halting_mode(ieee_overflow,prior_halting)
  h=tail%transform(csr_from_dense(one)); expected=tail%transform(one)
  call ieee_get_flag(ieee_invalid,invalid); call ieee_get_flag(ieee_overflow,overflow)
  call ieee_get_halting_mode(ieee_overflow,halting)
  call require(all(h==0).and.all(expected==0),'RBM scipy expit negative-tail underflow')
  call require(.not.invalid.and..not.overflow.and.(halting.eqv.prior_halting),'RBM expit restores IEEE state')
  recovered%n_components=1; recovered%n_iter=0; recovered%random_state=42
  signed_query=0; call recovered%fit(signed_query); recovered%components_=0
  recovered%intercept_visible_=-1.e307_dp*[6._dp,4._dp,7._dp,8._dp,7._dp,4._dp,2._dp,5._dp,6._dp,5._dp,3._dp]
  signed_query(1,:)=[-1._dp,-1._dp,-1._dp,-1._dp,1._dp,1._dp,1._dp,1._dp,1._dp,-1._dp,1._dp]
  scores=recovered%score_samples(signed_query)
  call require(.not.ieee_is_nan(scores(1)).and..not.ieee_is_finite(scores(1)).and.scores(1)<0, &
    'RBM recovers finite dot cancellation before pseudo-likelihood saturation')
  extreme%n_components=3; extreme%batch_size=4; extreme%n_iter=3; extreme%learning_rate=1.e300_dp
  huge_x=x*1.e100_dp; call extreme%fit(csr_from_dense(huge_x))
  call require(any(ieee_is_nan(extreme%h_samples_)),'RBM persistent-chain floor retains NaNs')
  draws=extreme%gibbs(sparse)
  call require(all(draws==0),'RBM Boolean Gibbs sampling treats NaN probabilities as false')
  call ieee_get_flag(ieee_invalid,invalid); call ieee_get_flag(ieee_overflow,overflow)
  call require(.not.invalid.and..not.overflow,'RBM nonfinite training preserves IEEE flags')
  print *, 'Sparse RBM tests: PASS'
end program
