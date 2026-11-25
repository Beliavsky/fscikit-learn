program test_kernel_ridge_extended
  use sklearn_common,only:dp,require
  use sklearn_kernel_ridge,only:kernel_ridge
  use sklearn_sparse,only:csr_matrix,csr_from_dense,csr_from_arrays
  use sklearn_base_dispatch,only:train_regressor,base_weighted
  use sklearn_voting,only:voting_regressor,voting_estimator
  use,intrinsic::ieee_exceptions
  implicit none
  type(kernel_ridge)::model,saved
  type(voting_regressor)::vote
  type(voting_estimator)::bases(2)
  type(csr_matrix)::sparse,query
  real(dp)::x(3,2),y(3),w(3),targets(3,2),k(2,2),rhs(2)
  real(dp),allocatable::pred(:,:),vector(:),large_y(:)
  integer,allocatable::ptr(:),cols(:)
  real(dp),allocatable::values(:)
  logical::flags(size(ieee_all)),modes(size(ieee_all)),after(size(ieee_all)),after_modes(size(ieee_all))
  integer::i,n,p
  x(:,1)=[1._dp,0._dp,1._dp];x(:,2)=[0._dp,1._dp,1._dp]
  y=[1._dp,2._dp,3._dp];w=[1._dp,0._dp,4._dp]
  call model%fit(x,y,w)
  call require(maxval(abs(model%dual_coef_-[-3._dp/14,0._dp,10._dp/7]))<1.e-13_dp,'weighted dual coefficients')
  vector=model%predict(x)
  call require(maxval(abs(vector-[17._dp/14,10._dp/7,37._dp/14]))<1.e-13_dp,'weighted kernel predictions')
  saved=model;targets(:,1)=y;targets(:,2)=2*y;model%alpha_values=[1._dp,2._dp]
  call model%fit(x,targets,w);pred=model%predict_multi(x)
  call require(model%n_targets_==2.and.all(shape(model%dual_coef_multi_)==[3,2]),'multi-target fitted shapes')
  call require(maxval(abs(model%dual_coef_multi_(:,2)-[-2._dp/13,0._dp,32._dp/13]))<1.e-13_dp, &
    'per-target penalties')
  call require(maxval(abs(saved%predict(x)-vector))==0,'model copies own coefficients')
  sparse=csr_from_dense(x);query=sparse
  call model%fit(sparse,targets,w)
  call require(model%sparse_fit_.and..not.allocated(model%x_fit_),'CSR fit keeps sparse training storage')
  call require(maxval(abs(model%predict_multi(query)-pred))<1.e-13_dp,'CSR train/query weighted prediction')
  call require(maxval(abs(model%predict_multi(x)-pred))<1.e-13_dp,'mixed sparse training/dense query')
  call model%fit(x,targets,w)
  call require(maxval(abs(model%predict_multi(query)-pred))<1.e-13_dp,'mixed dense training/sparse query')
  deallocate(model%alpha_values);model%kernel='precomputed';model%alpha=1
  k(1,:)=[2._dp,1._dp];k(2,:)=[99._dp,3._dp];rhs=[1._dp,2._dp]
  call model%fit(k,rhs)
  call require(maxval(abs(model%dual_coef_-[2._dp/11,5._dp/11]))<1.e-13_dp,'upper-triangle positive solve')
  call require(k(2,1)==99.and.all(model%x_fit_==k),'precomputed caller matrix preserved')
  call require(maxval(abs(model%predict(k)-[9._dp/11,213._dp/11]))<1.e-12_dp,'prediction uses full precomputed matrix')
  model%alpha=0;call model%fit(spread([1._dp,1._dp,1._dp],2,3),y)
  call require(model%used_lstsq_,'shared singular penalty selects least squares')
  call require(maxval(abs(model%dual_coef_-2._dp/3))<1.e-12_dp,'minimum-norm constant-kernel solution')
  model%kernel='linear';model%alpha=1
  call model%fit(x,y,0._dp)
  call require(all(model%dual_coef_==0),'scalar zero weight')
  call model%fit(x,y,1.75_dp);saved=model
  call model%fit(x,y,spread(1.75_dp,1,3))
  call require(maxval(abs(saved%dual_coef_-model%dual_coef_))==0,'scalar/vector weight broadcasting')
  call ieee_set_flag(ieee_all,.false.);call ieee_set_flag(ieee_underflow,.true.)
  call ieee_get_flag(ieee_all,flags);call ieee_get_halting_mode(ieee_all,modes)
  call model%fit(x,y,w);vector=model%predict(x)
  call ieee_get_flag(ieee_all,after);call ieee_get_halting_mode(ieee_all,after_modes)
  call require(all(flags.eqv.after).and.all(modes.eqv.after_modes),'kernel ridge IEEE state preservation')
  call require(base_weighted(model),'kernel ridge ensemble weight capability')
  call train_regressor(model,x,y,w)
  call require(maxval(abs(model%dual_coef_-[-3._dp/14,0._dp,10._dp/7]))<1.e-13_dp,'weighted ensemble dispatch')
  allocate(bases(1)%estimator,source=model)
  model%alpha=2
  allocate(bases(2)%estimator,source=model)
  vote%weights=[1._dp,2._dp]
  call vote%fit(x,y,bases,sample_weight=w)
  call model%fit(x,y,w)
  call require(maxval(abs(vote%predict(x)-(vector+2*model%predict(x))/3))<1.e-13_dp, &
    'voting forwards sample weights to both kernel ridge members')
  n=8;p=2097152;allocate(ptr(n+1),cols(n),values(n),large_y(n))
  ptr=[(i,i=1,n+1)];cols=[(i*10000,i=1,n)];values=1;large_y=[(real(i,dp),i=1,n)]
  sparse=csr_from_arrays(n,p,ptr,cols,values);model%alpha=.5_dp
  call model%fit(sparse,large_y)
  call require(maxval(abs(model%dual_coef_-large_y/1.5_dp))<1.e-13_dp,'wide one-hot sparse analytic solution')
  call require(model%n_features_in_==p.and..not.allocated(model%x_fit_),'wide CSR avoids full dense feature storage')
  print *,'Extended KernelRidge passed; avoided a 128 MiB dense feature matrix'
end program
