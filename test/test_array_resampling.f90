program test_array_resampling
  use sklearn_resampling, only: resample,shuffle
  use sklearn_sparse, only: csr_matrix,csr_from_arrays
  use sklearn_common, only: dp,require
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf,ieee_is_nan,ieee_is_negative
  implicit none
  integer, allocatable :: ids(:),other(:)
  integer :: labels(9)=[0,0,1,1,1,1,1,1,1],joint(4,2)
  real(dp), allocatable :: values(:)
  integer(int64), allocatable :: wide(:)
  character(:), allocatable :: text(:)
  type(csr_matrix) :: sparse,sampled
  ids=resample(3,random_state=0)
  call require(all(ids==[1,2,1]),'Source bootstrap example uses one-based indices')
  values=resample([1._dp,2._dp,0._dp],random_state=0)
  call require(all(values==[1._dp,2._dp,1._dp]),'Dense real data sampling')
  text=resample(['one','two','nil'],random_state=0)
  call require(all(text==['one','two','one']),'Character data sampling')
  wide=resample([huge(1_int64),-huge(1_int64),0_int64],random_state=0)
  call require(all(wide==[huge(1_int64),-huge(1_int64),huge(1_int64)]),'Int64 data are copied exactly')
  ids=resample(9,n_samples=5,replace=.false.,stratify=labels,random_state=0)
  call require(all(ids==[9,4,7,2,5]),'Stratified source no-replacement example')
  joint(:,1)=[2,10,2,10];joint(:,2)=0
  ids=resample(4,stratify=joint,random_state=0)
  call require(all(ids==[2,3,4,1]),'Joint integer labels use source lexical row order')
  ids=shuffle(9,random_state=7);other=resample(9,replace=.false.,random_state=7)
  call require(all(ids==other),'Shuffle is no-replacement resampling')
  ids=resample(0);call require(size(ids)==0,'Default empty bootstrap returns empty indices')
  sparse=csr_from_arrays(3,2,[1,3,4,4],[2,1,2], &
    [ieee_value(0._dp,ieee_quiet_nan),-0._dp,ieee_value(0._dp,ieee_positive_inf)])
  sampled=resample(sparse,random_state=0)
  call require(all(sampled%indptr==[1,3,4,6]).and.all(sampled%indices==[2,1,2,2,1]),'CSR row storage is preserved')
  call require(ieee_is_nan(sampled%data(1)).and.ieee_is_nan(sampled%data(4)), 'CSR nonfinite values are copied')
  call require(ieee_is_negative(sampled%data(2)).and.ieee_is_negative(sampled%data(5)),'CSR signed zeros are copied')
  print *, 'Array bootstrap/shuffle, stratification, joint labels and CSR row copies: PASS'
end program
