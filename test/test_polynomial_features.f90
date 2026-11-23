program test_polynomial_features
  use sklearn_polynomial, only: polynomial_features
  use sklearn_feature_extraction, only: feature_string
  use sklearn_sparse, only: csr_matrix,csr_from_arrays
  use sklearn_common, only: dp,require
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
  implicit none
  type(polynomial_features) :: model,high
  type(csr_matrix) :: sparse,expanded
  type(feature_string), allocatable :: names(:)
  type(feature_string) :: input_names(3)
  integer(int64), allocatable :: powers(:,:)
  real(dp) :: x(2,3)
  real(dp), allocatable :: z(:,:),legacy(:,:),dense(:,:)
  integer :: j
  x(1,:)=[2._dp,3._dp,-1._dp];x(2,:)=[0._dp,4._dp,2._dp]
  z=model%fit_transform(x)
  call require(model%n_output_features_==10.and.model%n_features_in_==3,'default fitted dimensions')
  call require(all(z(1,:)==[1._dp,2._dp,3._dp,-1._dp,4._dp,6._dp,-2._dp,9._dp,-3._dp,1._dp]), &
    'degree-two monomial ordering and values')
  legacy=polynomial_features(x,2)
  call require(all(legacy==z),'array API retained alongside type constructor')
  model=polynomial_features(degree=3_int64,min_degree=2_int64,include_bias=.false.)
  z=model%fit_transform(x);powers=model%powers_()
  call require(size(z,2)==16.and.all(sum(powers,dim=2)>=2).and.all(sum(powers,dim=2)<=3),'degree range and powers')
  input_names(1)%text='a';input_names(2)%text='b';input_names(3)%text='c'
  call model%fit(x,input_names);names=model%get_feature_names_out()
  call require(names(1)%text=='a^2'.and.names(2)%text=='a b'.and.names(16)%text=='c^3','feature name ordering')
  call model%fit(x);names=model%get_feature_names_out()
  call require(.not.allocated(model%feature_names_in_).and.names(1)%text=='x0^2','unnamed refit clears names')
  model=polynomial_features(degree=5_int64,min_degree=3_int64,include_bias=.false.)
  input_names(1)%text='';input_names(2)%text='heat flux';input_names(3)%text=achar(206)//achar(178)
  call model%fit(x,input_names);names=model%get_feature_names_out()
  call require(names(1)%text=='^3'.and.names(2)%text=='^2 heat flux'.and. &
    names(size(names))%text==achar(206)//achar(178)//'^5','empty and UTF-8 polynomial feature names')
  names=model%get_feature_names_out(input_names)
  call model%fit(x);names=model%get_feature_names_out()
  call require(names(1)%text=='x0^3'.and.names(size(names))%text=='x2^5','high-degree name refit')
  model=polynomial_features(degree=4_int64,min_degree=3_int64,include_bias=.false.,interaction_only=.true.)
  z=model%fit_transform(x(:,:2));powers=model%get_powers();names=model%get_feature_names_out()
  call require(size(z,2)==0.and.size(powers,1)==0.and.size(names)==0,'empty interaction feature set')
  model=polynomial_features()
  sparse=csr_from_arrays(1,2,[1,4],[2,1,1],[1._dp,2._dp,3._dp])
  expanded=model%fit_transform(sparse);dense=expanded%to_dense()
  call require(all(dense(1,:)==[1._dp,5._dp,1._dp,25._dp,5._dp,1._dp]),'duplicate CSR columns canonicalized')
  call require(all(sparse%indices==[2,1,1]).and.all(sparse%data==[1._dp,2._dp,3._dp]),'raw CSR input preserved')
  model=polynomial_features(degree=3_int64)
  x(1,:)=[1.e200_dp,1.e200_dp,0._dp];z=model%fit_transform(x(:1,:));powers=model%get_powers()
  sparse=csr_from_arrays(1,3,[1,4],[1,2,3],x(1,:));expanded=model%transform(sparse);dense=expanded%to_dense()
  do j=1,size(powers,1)
    if(all(powers(j,:)==1)) then
      call require(z(1,j)==0.and.ieee_is_nan(dense(1,j)),'dense suffix versus CSR left-associated products')
    end if
  end do
  model=polynomial_features(degree=4_int64)
  sparse=csr_from_arrays(1,3,[1,3],[1,2],[1.e200_dp,1.e200_dp])
  expanded=model%fit_transform(sparse);dense=expanded%to_dense();powers=model%get_powers()
  do j=1,size(powers,1)
    if(all(powers(j,:)==1)) call require(ieee_is_nan(dense(1,j)),'CSC overflow times implicit zero retains NaN')
  end do
  high=polynomial_features(degree=2147483647_int64,min_degree=2147483647_int64,include_bias=.false.)
  call high%fit(reshape([1._dp],[1,1]));powers=high%get_powers();names=high%get_feature_names_out()
  call require(high%n_output_features_==1.and.powers(1,1)==2147483647_int64,'large degree fit/powers without tuple allocation')
  call require(names(1)%text=='x0^2147483647','large degree feature name')
  sparse=csr_from_arrays(1,100000,[1,1],[integer::],[real(dp)::])
  high=polynomial_features(degree=3_int64);call high%fit(sparse)
  call require(high%n_output_features_==166676666850001_int64,'int64 combinatorial fit without expansion allocation')
  legacy=polynomial_features(reshape([.5_dp],[1,1]),13)
  call require(size(legacy,2)==14.and.legacy(1,14)==.5_dp**13,'legacy degree cap removed')
  print *, 'PolynomialFeatures estimator and array API: PASS'
end program
