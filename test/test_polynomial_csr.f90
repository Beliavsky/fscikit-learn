program test_polynomial_csr
 use sklearn_common,only:dp,require
 use sklearn_count_sketch_fft,only:count_sketch_fft_plan
 use sklearn_sparse,only:csr_matrix,csr_from_arrays,csr_from_dense
 use sklearn_polynomial_count_sketch,only:polynomial_count_sketch
 use,intrinsic::ieee_arithmetic,only:ieee_is_nan,ieee_is_finite
 use,intrinsic::ieee_exceptions,only:ieee_get_halting_mode,ieee_get_flag, &
   ieee_invalid,ieee_divide_by_zero,ieee_overflow
 implicit none
 type(polynomial_count_sketch)::model,dense_model
 type(count_sketch_fft_plan)::plan
 complex(dp),allocatable::v(:),expected(:),input(:)
 integer::n,i,j,t
 integer,parameter::primes(*)=[3,5,7,11,13,17,19,23,29,31]
 real(dp)::angle,pi
 type(csr_matrix)::x,original,big,q
 real(dp)::a(2,2)
 real(dp),allocatable::z(:,:),dense(:,:)
 logical::before(6),after(6)
 pi=acos(-1._dp)
 do t=1,size(primes)
  n=primes(t);allocate(v(n),expected(n),input(n))
  input=[(cmplx(sin(real(i,dp)),cos(real(i,dp)),dp),i=1,n)];v=input
  call plan%initialize(n);call plan%forward(v);expected=0
  do i=0,n-1
   do j=0,n-1
    angle=-2*pi*real(i*j,dp)/n
    expected(i+1)=expected(i+1)+input(j+1)*cmplx(cos(angle),sin(angle),dp)
   end do
  end do
  call require(maxval(abs(v-expected))<1.e-11_dp,'prime butterfly direct DFT')
  call plan%inverse(v);call require(maxval(abs(v-input))<1.e-11_dp,'prime butterfly inverse')
  deallocate(v,expected,input)
 end do
 x=csr_from_arrays(2,2,[1,5,5],[2,1,1,2],[3._dp,1._dp,1._dp,0._dp]);original=x
 a=x%to_dense();model%n_components=5;model%degree=2;call model%fit(x)
 model%index_hash_(1,:)=[0,1];model%index_hash_(2,:)=[1,2]
 model%bit_hash_(1,:)=[1,-1];model%bit_hash_(2,:)=[-1,1]
 z=model%transform(x)
 call require(maxval(abs(z(1,:)-[0._dp,-4._dp,12._dp,-9._dp,0._dp]))<1.e-12_dp,'CSR tensor enumeration')
 call require(all(z(2,:)==0),'CSR implicit zero row')
 dense=model%transform(a);call require(maxval(abs(z-dense))<1.e-12_dp,'mixed TensorSketch query')
 model%gamma=2.3_dp;model%coef0=.7_dp;model%n_components=17
 z=model%fit_transform(x);dense_model=model;dense=dense_model%fit_transform(a)
 call require(all(model%index_hash_==dense_model%index_hash_),'dense/CSR source hash identity')
 call require(maxval(abs(z-dense))<1.e-9_dp,'scaled duplicate CSR sketch')
 call require(all(x%indptr==original%indptr).and.all(x%data==original%data),'CSR input preservation')
 call environment(before);model%coef0=-1;z=model%fit_transform(x);call environment(after)
 call require(all(ieee_is_nan(z)).and.all(before.eqv.after),'negative offset NaNs preserve IEEE modes')
 a=1.e308_dp;model%coef0=0;model%gamma=1;model%degree=2;model%n_components=8
 call environment(before);z=model%fit_transform(a);call environment(after)
 call require(all(ieee_is_nan(z)).and.all(before.eqv.after),'FFT overflow returns NaNs and restores environment')
 ! A dense copy of this fit input would require 80 GB.
 big%n_rows=100000;big%n_cols=100000
 allocate(big%indptr(100001),big%indices(1),big%data(1))
 big%indptr=2;big%indptr(1)=1;big%indices=100000;big%data=2
 model%degree=2;model%n_components=8;call model%fit(big)
 q=csr_from_arrays(2,100000,[1,2,2],[100000],[2._dp]);z=model%transform(q)
 call require(all(shape(z)==[2,8]).and.all(ieee_is_finite(z)),'large sparse TensorSketch fit/query')
 call model%fit(a)
 call require(model%n_features_in_==2.and.all(shape(model%index_hash_)==[2,2]),'refit replaces sparse hash dimensions')
 print *, 'PolynomialCountSketch CSR duplicates, mixed queries, IEEE environment and sparse shapes: PASS'
contains
 subroutine environment(values)
  logical,intent(out)::values(6)
  call ieee_get_halting_mode(ieee_invalid,values(1));call ieee_get_halting_mode(ieee_divide_by_zero,values(2))
  call ieee_get_halting_mode(ieee_overflow,values(3))
  call ieee_get_flag(ieee_invalid,values(4));call ieee_get_flag(ieee_divide_by_zero,values(5))
  call ieee_get_flag(ieee_overflow,values(6))
 end subroutine
end program
