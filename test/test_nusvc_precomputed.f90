program test_nusvc_precomputed
  use sklearn_nu_classification,only:nu_svc
  use sklearn_common,only:dp,require
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan
  implicit none
  type(nu_svc)::model
  type(csr_matrix)::sparse
  real(dp)::asym(2,2),raw(4,1),queries(3,1),bad(2,2)
  real(dp),allocatable::scores(:,:),gram(:,:),query_gram(:,:)
  integer,allocatable::predicted(:)
  character(32)::mode
  asym(1,:)=[2._dp,5._dp];asym(2,:)=[-1._dp,3._dp]
  model%kernel='precomputed';call get_command_argument(1,mode)
  select case(trim(mode))
  case('square');call model%fit(reshape([1._dp,0._dp,0._dp,1._dp,1._dp,0._dp],[3,2]),[-3,9,9])
  case('fit_nan');bad=asym;bad(1,1)=ieee_value(0._dp,ieee_quiet_nan);call model%fit(bad,[-3,9])
  case('query_width');call model%fit(asym,[-3,9]);scores=model%decision_function(asym(:,:1))
  case('query_nan');call model%fit(asym,[-3,9]);bad=asym;bad(1,1)=ieee_value(0._dp,ieee_quiet_nan);scores=model%decision_function(bad)
  case('csr_fit');sparse=csr_from_dense(asym);call model%fit(sparse,[-3,9])
  case('csr_query');call model%fit(asym,[-3,9]);sparse=csr_from_dense(asym);predicted=model%predict(sparse)
  end select
  call require(len_trim(mode)==0,'precomputed NuSVC rejection unexpectedly accepted input')
  call model%fit(asym,[-3,9]);scores=model%decision_function(asym)
  call require(all(abs(model%dual_coef_(1,:)-[-2._dp,2._dp])<1.e-12_dp),'asymmetric NuSVC Source duals')
  call require(abs(model%intercept_(1)-5._dp)<1.e-12_dp.and.all(abs(scores(:,1)-[11._dp,13._dp])<1.e-12_dp), &
    'asymmetric NuSVC Source row-kernel normalization and queries')
  call require(model%fit_status_==0.and.all(model%n_iter_==0),'asymmetric NuSVC Source stationary cap status')
  call require(all(shape(model%support_vectors_)==[0,0]).and..not.allocated(model%coef_), &
    'precomputed NuSVC empty support-vector storage')
  call require(model%gamma_==0.and.model%n_features_in_==2,'precomputed NuSVC metadata')
  raw(:,1)=[-4._dp,-1._dp,1._dp,3._dp];queries(:,1)=[-2._dp,0._dp,2._dp]
  gram=matmul(raw,transpose(raw));query_gram=matmul(queries,transpose(raw))
  call model%fit(gram,[-3,-3,9,9],[0._dp,1._dp,1._dp,1._dp]);scores=model%decision_function(query_gram)
  call require(all(abs(scores(:,1)-queries(:,1))<1.e-10_dp),'precomputed NuSVC original query columns after zero weights')
  call require(all(model%support_==[2,3]),'precomputed NuSVC original support mapping')
  model%kernel='linear';call model%fit(raw,[-3,-3,9,9]);call require(allocated(model%coef_),'dense linear refit')
  model%kernel='precomputed';call model%fit(asym,[-3,9]);scores=model%decision_function(asym)
  call require(.not.allocated(model%coef_).and.all(shape(model%support_vectors_)==[0,0]),'precomputed refit clears feature storage')
  model%kernel='linear';sparse=csr_from_dense(raw);call model%fit(sparse,[-3,-3,9,9])
  model%kernel='precomputed';call model%fit(asym,[-3,9]);scores=model%decision_function(asym)
  call require(.not.model%sparse_fit_.and..not.allocated(model%x_fit_csr_).and..not.allocated(model%coef_csr_), &
    'precomputed refit clears sparse fit storage')
  call require(all(abs(scores(:,1)-[11._dp,13._dp])<1.e-12_dp),'precomputed queries after sparse refit')
  print *,'Precomputed NuSVC asymmetry, normalization, original supports and refits: PASS'
end program
