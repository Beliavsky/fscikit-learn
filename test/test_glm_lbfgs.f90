program test_glm_lbfgs
  use sklearn
  use sklearn_common,only:require
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  use sklearn_search_adapters,only:reset_search_clone,set_search_params
  use sklearn_parameter_space,only:parameter_set,param
  use,intrinsic::ieee_arithmetic,only:ieee_is_finite
  implicit none
  type(tweedie_regressor)::dense,copy,wide
  type(poisson_regressor)::poisson
  type(gamma_regressor)::gamma
  type(csr_matrix)::sparse,huge_design
  type(parameter_set)::parameters
  type(pipeline)::pipe
  real(dp)::x(27,3),y(27),weight(27),before(27,3)
  real(dp),allocatable::stored(:),prediction(:)
  integer::i
  do i=1,27
    x(i,:)=[real(i-14,dp)/27,sin(real(i,dp))*.1_dp,real(mod(i,4),dp)*.1_dp]
  end do
  y=exp(.3_dp+.2_dp*x(:,1)-.4_dp*x(:,2)+.1_dp*x(:,3));weight=1;weight(1:3)=0
  before=x;sparse=csr_from_dense(x,store_zeros=.true.);stored=sparse%data
  dense%power=1.5_dp;dense%alpha=.2_dp;dense%tol=1.e-10_dp
  call require(dense%solver=='lbfgs','GLM default solver follows sklearn')
  call dense%fit(x,y,weight);copy=dense;call reset_search_clone(copy)
  call require(.not.allocated(copy%coef_).and.copy%solver=='lbfgs','GLM clone retains solver and clears fit')
  call copy%fit(sparse,reshape(y,[27,1]),weight)
  call require(maxval(abs(dense%predict(x)-copy%predict(sparse)))<1.e-7_dp,'GLM weighted dense/CSR L-BFGS')
  call require(abs(dense%score(x,y,weight)-copy%score(sparse,y,weight))<1.e-8_dp,'GLM weighted sparse D2')
  call require(all(sparse%data==stored).and.all(x==before),'GLM leaves dense/CSR inputs unchanged')
  copy%warm_start=.true.;copy%alpha=.1_dp;call copy%fit(sparse,y,weight)
  call poisson%fit(sparse,y,2._dp);call gamma%fit(sparse,reshape(y,[27,1]),2._dp)
  call require(poisson%power_==1.and.gamma%power_==2,'GLM sparse derived-type fixed losses')
  allocate(pipe%steps(1));pipe%steps(1)%name='glm';allocate(pipe%steps(1)%model,source=dense)
  call pipe%fit(x,y,weight)
  call require(maxval(abs(pipe%predict(x)-dense%predict(x)))<1.e-9_dp,'GLM L-BFGS Pipeline dispatch')
  allocate(parameters%items(1));parameters%items(1)%name='solver';parameters%items(1)%value=param('native-newton')
  call set_search_params(copy,parameters);call require(copy%solver=='native-newton','GLM solver search parameter')
  ! This fit would require over 12 GiB for a 40001-square Hessian. The L-BFGS
  ! sparse path retains only O(nnz+n+p) data plus its fixed-memory workspace.
  huge_design%n_rows=8;huge_design%n_cols=40000
  huge_design%indptr=[(i,i=1,9)];huge_design%indices=[1,10000,40000,1,10000,40000,1,10000]
  huge_design%data=[.1_dp,.2_dp,.3_dp,.4_dp,.5_dp,.6_dp,.7_dp,.8_dp]
  wide%power=1;wide%max_iter=2;call wide%fit(huge_design,y(:8))
  prediction=wide%predict(huge_design)
  call require(size(wide%coef_)==40000.and.wide%n_iter_<=2,'GLM wide sparse L-BFGS fit')
  call require(all(ieee_is_finite(prediction)).and.all(prediction>0),'GLM wide sparse predictions')
  print *,'GLM L-BFGS dense/CSR, warm starts, clones and wide sparse memory: PASS'
end program
