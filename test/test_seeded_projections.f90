program test_seeded_projections
  use sklearn_common, only: dp,require
  use sklearn_random_projection
  use sklearn_sparse, only: csr_matrix,csr_from_arrays
  implicit none
  type(gaussian_random_projection) :: gaussian
  type(sparse_random_projection) :: sparse
  type(csr_matrix) :: wide
  real(dp) :: x(2,3)
  real(dp),allocatable :: z(:,:),back(:,:)
  integer,allocatable :: before(:),after(:)
  integer :: n
  integer :: i,j,moment,total_moment
  integer, parameter :: row_counts(8)=[4,3,7,11,2,5,3,9]
  integer, parameter :: row_moments(8)=[16998,-42307,-71161,-70182,33874,-84220,-17903,36986]
  integer, parameter :: tracking_counts(3)=[4932,5095,5032]
  integer, parameter :: tracking_moments(3)=[-965384,33941593,136982876]
  x=reshape([1._dp,2._dp,3._dp,4._dp,5._dp,6._dp],[2,3])
  call random_seed(size=n);allocate(before(n),after(n));call random_seed(get=before)
  gaussian%n_components=2;gaussian%random_state=0;z=gaussian%fit_transform(x)
  call require(maxval(abs(gaussian%components_(1,:)-[1.2473733762017727_dp,.28295387557714186_dp, &
    .6920722655660196_dp]))<1.e-15_dp,'NumPy row-major Gaussian component stream')
  call require(gaussian%n_features_in_==3,'projection fitted feature width')
  back=gaussian%inverse_transform(z)
  call require(maxval(abs(gaussian%transform(back)-z))<1.e-12_dp,'seeded Gaussian inverse coordinates')
  sparse%n_components=2;sparse%random_state=0;sparse%density=1;call sparse%fit(x)
  call require(all(sparse%components_==reshape([1._dp,1._dp,1._dp,-1._dp,1._dp,1._dp],[2,3])/sqrt(2._dp)), &
    'NumPy dense binomial signs')
  sparse%density=.2_dp;sparse%compute_inverse_components=.true.;call sparse%fit(x)
  sparse%compute_inverse_components=.false.;call sparse%fit(x)
  call require(.not.allocated(sparse%inverse_components_),'refit clears inverse cache')
  ! sklearn 1.6.1: 64 by 50000 components, density .0001, seed 42.
  ! In particular, signs must be drawn after ALL row counts and columns.
  wide=csr_from_arrays(2,50000,[1,1,1],[integer ::],[real(dp) ::])
  sparse%n_components=64;sparse%density=.0001_dp;sparse%random_state=42
  call sparse%fit(wide)
  call require(count(sparse%components_/=0)==333,'wide sparse projection sampled count')
  total_moment=0
  do i=1,64
    moment=0
    do j=1,50000
      if(sparse%components_(i,j)/=0) moment=moment+j*int(sign(1._dp,sparse%components_(i,j)))
    end do
    total_moment=total_moment+moment
    if(i<=8) then
      call require(count(sparse%components_(i,:)/=0)==row_counts(i),'wide sparse row counts')
      call require(moment==row_moments(i),'wide sparse signed column moments')
    end if
  end do
  call require(total_moment==-102015,'wide sparse complete signed column moment')
  ! An empty selection must neither consume sign draws nor retain old storage.
  sparse%density=1.e-310_dp;call sparse%fit(wide)
  call require(all(sparse%components_==0),'wide sparse empty selections on refit')
  wide=csr_from_arrays(2,1000000,[1,1,1],[integer ::],[real(dp) ::])
  sparse%n_components=3;sparse%density=.005_dp;call sparse%fit(wide)
  do i=1,3
    call require(count(sparse%components_(i,:)/=0)==tracking_counts(i),'tracking-selection row count')
    moment=0
    do j=1,1000000
      if(sparse%components_(i,j)/=0) moment=moment+j*int(sign(1._dp,sparse%components_(i,j)))
    end do
    call require(moment==tracking_moments(i),'tracking-selection signed column moment')
  end do
  call random_seed(get=after);call require(all(before==after),'projection fits preserve intrinsic RNG')
  print *, 'Seeded Gaussian and sparse projections, inverse cache and RNG isolation: PASS'
end program
