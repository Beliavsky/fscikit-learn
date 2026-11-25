program test_kernel_dispatch
 use sklearn,only:dp,pairwise_kernels
 use sklearn_common,only:require
 use sklearn_sparse,only:csr_matrix,csr_from_dense
 use sklearn_gp_kernels,only:gp_kernel,white_covariance,rbf_covariance,operator(+)
 use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan,ieee_positive_inf,ieee_is_nan
 implicit none
 real(dp),allocatable::x(:,:),y(:,:),k(:,:),direct(:,:)
 type(csr_matrix)::sx,sy
 type(gp_kernel)::white,combo
 integer::counter
 real(dp)::nan,inf
 nan=ieee_value(0._dp,ieee_quiet_nan);inf=ieee_value(0._dp,ieee_positive_inf)
 x=reshape([0._dp,1._dp,0._dp,1._dp],[2,2]);y=reshape([1._dp,1._dp,0._dp,1._dp],[2,2])
 k=pairwise_kernels(x,y)
 call require(all(k(1,:)==0).and.all(k(2,:)==[1._dp,2._dp]),'default linear kernel')
 k=pairwise_kernels(x,y,metric='linear',gamma=nan,degree=-3._dp,coef0=inf,filter_params=.true.)
 call require(all(k(2,:)==[1._dp,2._dp]),'filter removes inapplicable parameters without validating them')
 k=pairwise_kernels(x,y,metric='rbf',gamma=-.5_dp)
 call require(abs(k(1,1)-exp(.5_dp))<1.e-14_dp,'dispatcher retains negative gamma bypass behavior')
 k=pairwise_kernels(x,y,metric='poly',degree=2)
 call require(all(k(1,:)==1).and.all(k(2,:)==[2.25_dp,4._dp]),'integer polynomial degree overload')
 k=pairwise_kernels(x,y,metric='poly',gamma=nan,degree=0._dp)
 call require(all(k==1),'zero exponent of NaN bases is one')
 k=pairwise_kernels(x,y,metric='poly',gamma=1._dp,degree=.5_dp)
 call require(abs(k(2,1)-sqrt(2._dp))<1.e-15_dp,'fractional polynomial degree')
 sx=csr_from_dense(x);sy=csr_from_dense(y)
 k=pairwise_kernels(sx,sy,metric='linear')
 call require(all(k(2,:)==[1._dp,2._dp]),'sparse linear kernel')
 k=pairwise_kernels(sx,y,metric='rbf')
 call require(k(2,2)==1,'mixed sparse RBF')
 x=reshape([1.e-16_dp,0._dp],[1,2]);y=reshape([1._dp,0._dp],[1,2]);sx=csr_from_dense(x)
 k=pairwise_kernels(x,y,metric='cosine');call require(k(1,1)==1.e-16_dp,'dense tiny-row normalization cutoff')
 k=pairwise_kernels(sx,y,metric='cosine');call require(k(1,1)==1,'sparse tiny row is normalized')
 x=reshape([0._dp,1._dp],[1,2]);y=reshape([0._dp,3._dp],[1,2])
 k=pairwise_kernels(x,y,metric='additive_chi2');call require(k(1,1)==-1,'additive chi-squared')
 k=pairwise_kernels(x,y,metric='chi2',gamma=-1._dp)
 call require(abs(k(1,1)-exp(1._dp))<1.e-14_dp,'negative chi-squared gamma is retained by dispatcher')
 x=reshape([1._dp,-2._dp,3._dp,4._dp,5._dp,6._dp],[2,3]);y=reshape([1._dp,2._dp,3._dp],[3,1])
 k=pairwise_kernels(x,y,metric='precomputed',gamma=nan)
 call require(all(shape(k)==[2,3]).and.all(k==x),'precomputed reference shape and ignored parameters')
 k(1,1)=99;call require(x(1,1)==1,'precomputed returns an owned dense result')
 sx=csr_from_dense(x);k=pairwise_kernels(sx,y,metric='precomputed')
 call require(all(k==x),'precomputed CSR becomes owned dense storage')
 x=reshape([0._dp,1._dp,0._dp,1._dp],[2,2]);counter=0
 k=pairwise_kernels(x,metric=counting)
 call require(counter==3.and.all(k(1,:)==[2._dp,1._dp]).and.all(k(2,:)==[1._dp,3._dp]), &
  'self callbacks evaluate upper triangle then diagonal once')
 white=white_covariance(.3_dp);combo=rbf_covariance(1.2_dp)+white
 k=pairwise_kernels(x,metric=white);direct=white%matrix(x)
 call require(all(k==0).and.direct(1,1)==.3_dp,'dispatcher supplies Y=X to white GP kernels')
 k=pairwise_kernels(x,metric=combo)
 call require(k(1,1)==1.and.k(2,2)==1,'GP composite forwarding')
 x=reshape([1.e200_dp,1.e200_dp,1.e200_dp],[1,3]);y=x;y(1,2)=-y(1,2)
 k=pairwise_kernels(x,y,metric='linear')
 call require(.not.ieee_is_nan(k(1,1)),'fused dense dot preserves the pinned overflow path')
 print *,'Pairwise named/callback/GP/precomputed kernel dispatch: PASS'
contains
 real(dp)function counting(a,b)result(value)
  real(dp),intent(in)::a(:),b(:)
  counter=counter+1;value=real(counter,dp)
 end function
end program
