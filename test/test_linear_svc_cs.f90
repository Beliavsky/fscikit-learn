program test_linear_svc_cs
  use sklearn_common,only:dp,require
  use sklearn_linear_svm,only:linear_svc
  use sklearn_sparse,only:csr_matrix,csr_from_arrays,csr_from_dense
  implicit none
  type(linear_svc)::model,repeat
  type(csr_matrix)::x
  real(dp)::pair(2,1),three(3,2)
  real(dp),allocatable::coef(:,:)
  integer,allocatable::ptr(:),indices(:),labels(:),before(:),after(:)
  real(dp),allocatable::data(:)
  integer::i,n,iterations
  character(32)::mode
  call get_command_argument(1,mode)
  model%multi_class='crammer_singer'
  if(trim(mode)=='memory')then
    allocate(ptr(129),indices(256),data(256),labels(128))
    do i=1,128
      ptr(i)=2*i-1;indices(2*i-1:2*i)=[1,i+1]
      data(2*i-1:2*i)=[real(mod(i,3)-1,dp),1._dp];labels(i)=mod(i,3)
    end do
    ptr(129)=257;x=csr_from_arrays(128,200000,ptr,indices,data)
    call model%fit(x,labels)
    call require(.not.allocated(model%dual_variables_),'joint alpha diagnostics default off')
    call require(all(shape(model%coef_)==[3,200000]).and.model%converged_,'large joint classifier fit')
    print *,'Large Crammer-Singer memory case: PASS';stop
  end if
  pair(:,1)=[-1._dp,1._dp];x=csr_from_dense(pair)
  select case(trim(mode))
  case('multi_class');model%multi_class='invalid';call model%fit(x,[-3,9])
  case('inactive_binary');call model%fit(x,[-3,9],sample_weight=[0._dp,1._dp])
  end select
  call require(len_trim(mode)==0,'joint classifier rejection unexpectedly accepted input')
  call random_seed(size=n);allocate(before(n),after(n));call random_seed(get=before)
  call require(model%loss=='squared_hinge'.and.model%dual=='auto','joint selection keeps default loss controls')
  model%fit_intercept=.false.;model%tol=1.e-10_dp;model%store_dual_variables=.true.
  call model%fit(x,[-3,9])
  call require(abs(model%coef_(1,1)-1._dp)<1.e-12_dp,'binary joint coefficient difference')
  call require(abs(model%objective_(1)-.25_dp)<1.e-12_dp.and.abs(model%duality_gap_(1))<1.e-12_dp, &
    'joint binary objective uses both uncollapsed class vectors')
  call require(all(shape(model%dual_variables_)==[2,2]),'binary alpha retains both class coordinates')
  call require(maxval(abs(sum(model%dual_variables_,dim=2)))<1.e-12_dp,'joint alpha row constraints')
  coef=model%coef_;iterations=model%n_iter_;model%loss='hinge';model%dual='false';model%max_iter=1
  call model%fit(pair,[-3,9])
  call require(all(coef==model%coef_).and.iterations==model%n_iter_,'joint solver ignores loss dual and requested cap')
  call require(.not.model%converged_,'joint warning status uses requested cap')
  model%max_iter=1000;three(:,1)=[-1._dp,0._dp,1._dp];three(:,2)=[1._dp,-1._dp,1._dp]
  call model%fit(three,[-4,7,21]);coef=model%coef_
  call require(size(model%coef_,1)==3.and.size(model%objective_)==1,'multiclass vectors with a joint objective')
  call model%fit(three,[-4,7,21]);call require(all(model%coef_==coef),'joint seeded refit')
  x=csr_from_arrays(2,3,[1,1,1],[integer::],[real(dp)::]);call model%fit(x,[-3,9])
  call require(model%n_iter_==1.and.all(model%coef_==0),'reference skips zero-norm joint rows')
  call require(model%objective_(1)==2.and.model%duality_gap_(1)==2,'zero-norm residual duality gap is retained')
  model%multi_class='ovr';model%loss='squared_hinge';call model%fit(pair,[-3,9])
  call require(abs(model%coef_(1,1)-.8_dp)<1.e-12_dp,'refit from joint solver to primal OVR')
  call require(.not.allocated(model%dual_variables_),'primal refit clears joint alpha state')
  call random_seed(get=after);call require(all(before==after),'joint solver preserves intrinsic RNG')
  print *,'Crammer-Singer objectives, binary collapse, refits and control semantics: PASS'
end program
