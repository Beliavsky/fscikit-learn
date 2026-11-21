program test_kernel_centering
  use sklearn_kernel_centering
  use sklearn_common, only: dp,require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan,ieee_is_finite
  implicit none
  type(kernel_centerer) :: model,copy
  real(dp) :: kernel(4,4),original(4,4),h(4,4),query(2,4),large(2,2)
  real(dp), allocatable :: centered(:,:),expected(:,:),result(:,:)
  character(32), allocatable :: names(:)
  integer :: i
  kernel=reshape([(real(i*i,dp),i=1,16)],[4,4]); original=kernel
  centered=model%fit_transform(kernel); h=-.25_dp
  do i=1,4
    h(i,i)=h(i,i)+1
  end do
  expected=matmul(matmul(h,kernel),h)
  call require(maxval(abs(centered-expected))<1.e-12_dp,'kernel centering equals H K H for nonsymmetric input')
  call require(maxval(abs(sum(centered,dim=1)))<1.e-12_dp,'centered kernel column sums vanish')
  call require(maxval(abs(sum(centered,dim=2)))<1.e-12_dp,'centered kernel row sums vanish')
  call require(all(kernel==original),'kernel centering preserves caller storage')
  query=kernel(:2,:)+3
  result=model%transform(query)
  call require(maxval(abs(result-centered(:2,:)))<1.e-12_dp,'kernel transform uses training column moments')
  copy=model; names=model%get_feature_names_out()
  call require(trim(names(1))=='kernelcenterer0'.and.size(names)==4,'kernel centerer feature names')
  large=1.e308_dp; result=model%fit_transform(large)
  call require(all(ieee_is_nan(result)),'kernel overflow retains source quiet NaN results under FPE traps')
  call model%fit(reshape([-1.e308_dp],[1,1])); result=model%transform(reshape([1.e308_dp],[1,1]))
  call require(.not.ieee_is_finite(result(1,1)).and.result(1,1)>0,'kernel subtraction overflow retains positive infinity')
  result=copy%transform(kernel)
  call require(maxval(abs(result-centered))<1.e-12_dp,'kernel centerer copy survives refit and width changes')
  print *, 'KernelCenterer: PASS'
end program
