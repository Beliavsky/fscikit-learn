program test_svc_precomputed
  use sklearn_svm,only:svc
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  use sklearn_common,only:dp,require
  use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan
  implicit none
  type(svc)::model,ordinary
  type(csr_matrix)::sparse
  real(dp)::pair(2,1),gram(2,2),x(4,1),full(4,4),q(3,1),cross(3,4),asym(2,2)
  real(dp),allocatable::scores(:,:),expected(:,:)
  integer,allocatable::labels(:),before(:),after(:)
  integer::n
  character(32)::mode
  pair(:,1)=[-1._dp,1._dp];gram=matmul(pair,transpose(pair));model%kernel='precomputed';model%tol=1.e-9_dp
  call get_command_argument(1,mode)
  select case(trim(mode))
  case('square');call model%fit(reshape([1._dp,0._dp,0._dp,1._dp,2._dp,3._dp],[2,3]),[-3,9])
  case('sparse');sparse=csr_from_dense(gram);call model%fit(sparse,[-3,9])
  case('target_count');call model%fit(gram,[-3])
  case('query_width');call model%fit(gram,[-3,9]);labels=model%predict(pair)
  case('query_nan');call model%fit(gram,[-3,9]);gram(1,1)=ieee_value(0._dp,ieee_quiet_nan);labels=model%predict(gram)
  case('fit_nan');gram(1,1)=ieee_value(0._dp,ieee_quiet_nan);call model%fit(gram,[-3,9])
  case('csr_query');call model%fit(gram,[-3,9]);sparse=csr_from_dense(gram);labels=model%predict(sparse)
  end select
  call require(len_trim(mode)==0,'precomputed SVC rejection unexpectedly accepted input')
  call random_seed(size=n);allocate(before(n),after(n));call random_seed(get=before)
  ordinary%kernel='linear';ordinary%tol=1.e-9_dp;call ordinary%fit(pair,[-3,9]);call model%fit(gram,[-3,9])
  expected=ordinary%decision_function(pair);scores=model%decision_function(gram)
  call require(maxval(abs(expected-scores))<1.e-12_dp,'precomputed and ordinary kernels agree without zero weights')
  call require(model%gamma_==0.and.model%n_features_in_==2,'precomputed metadata and unused gamma')
  call require(.not.allocated(model%x_fit_).and.size(model%support_vectors_)==0,'precomputed model retains no Gram or feature design')
  call require(.not.allocated(model%coef_).and..not.allocated(model%coef_csr_),'precomputed has no primal feature coefficients')
  x(:,1)=[-4._dp,-1._dp,1._dp,3._dp];q(:,1)=[-2._dp,0._dp,2._dp]
  full=matmul(x,transpose(x));cross=matmul(q,transpose(x));call model%fit(full,[-3,-3,9,9],[0._dp,1._dp,1._dp,1._dp])
  call require(all(model%support_==[1,2]).and.all(model%support_original_==[2,3]),'precomputed compressed and original support maps')
  scores=model%decision_function(cross)
  call require(maxval(abs(scores(:,1)-[-3._dp,0._dp,3._dp]))<1.e-12_dp,'reference zero-weight compressed query-column convention')
  asym=reshape([1._dp,-3._dp,5._dp,1._dp],[2,2]);call model%fit(asym,[-3,9]);scores=model%decision_function(asym)
  call require(all(abs(scores-8._dp)<1.e-12_dp),'asymmetric supplied-kernel row orientation')
  model%kernel='linear';call model%fit(pair,[-3,9]);call require(allocated(model%coef_).and.allocated(model%x_fit_),'ordinary refit restores feature storage')
  model%kernel='precomputed';call model%fit(gram,[-3,9]);call require(.not.allocated(model%coef_).and..not.allocated(model%x_fit_),'precomputed refit clears ordinary storage')
  call random_seed(get=after);call require(all(before==after),'precomputed SVC preserves intrinsic RNG')
  print *,'Precomputed SVC shape, storage, support indexing and supplied kernels: PASS'
end program
