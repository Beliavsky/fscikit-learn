program test_quantile_csr
  use sklearn_quantile,only:quantile_regressor
  use sklearn_common,only:dp,require
  use sklearn_sparse,only:csr_matrix,csr_from_arrays,csr_from_dense
  implicit none
  type(quantile_regressor)::model,dense
  type(csr_matrix)::s,query,before
  real(dp)::x(9,2),y(9),weights(9)
  real(dp),allocatable::prediction(:)
  integer::i
  do i=1,9
    x(i,:)=[real(i-5,dp),sin(real(i,dp))]
  end do
  y=2+3*x(:,1)-.5_dp*x(:,2);weights=1;weights(1)=0
  s=csr_from_dense(x,store_zeros=.true.);before=s;model%alpha=.01_dp;dense%alpha=.01_dp
  call model%fit(s,reshape(y,[9,1]),weights);call dense%fit(x,y,weights)
  call require(maxval(abs(model%coef_-dense%coef_))<1.e-10_dp,'CSR quantile coefficients')
  call require(abs(model%score(s,reshape(y,[9,1]),weights)-dense%score(x,y,weights))<1.e-12_dp,'CSR weighted score')
  call require(all(s%data==before%data).and.all(s%indices==before%indices),'CSR quantile input preserved')
  call model%fit(s,y,0._dp)
  call require(all(model%coef_==0).and.model%intercept_==0.and.model%n_iter_==0,'zero-weight quantile fit')
  call require(model%converged_.and.all(model%dual_coef_==0),'zero-weight quantile certificate')
  call model%fit(x,reshape(y,[9,1]),2._dp);call dense%fit(x,y)
  call require(maxval(abs(model%predict(s)-dense%predict(x)))<1.e-10_dp,'scalar column weights')
  ! The design has 4,000 columns; the query's dense representation is 1.28 GB.
  ! Only the required simplex tableau is dense during fitting.
  s=csr_from_arrays(9,4000,[(2*i+1,i=0,9)],[(i,i=1,18)],spread(0._dp,1,18))
  do i=1,9
    s%indices(2*i-1:2*i)=[4000,1];s%data(2*i-1:2*i)=x(i,[2,1])
  end do
  call model%fit(s,y)
  call require(maxval(abs(model%predict(s)-dense%predict(x)))<1.e-8_dp,'wide sparse quantile fit')
  query%n_rows=40000;query%n_cols=4000
  query%indptr=[(2*i+1,i=0,40000)];allocate(query%indices(80000),query%data(80000))
  do i=1,40000
    query%indices(2*i-1:2*i)=[4000,1];query%data(2*i-1:2*i)=[.2_dp,-.3_dp]
  end do
  prediction=model%predict(query)
  call require(maxval(abs(prediction-(model%intercept_+.2_dp*model%coef_(4000)-.3_dp*model%coef_(1))))<1.e-10_dp, &
    'wide sparse query values')
  print *,'Quantile CSR 40000 by 4000 memory: PASS'
end program
