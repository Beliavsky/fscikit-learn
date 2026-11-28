program test_sparse_projection_components
  use sklearn_common, only: dp,require
  use sklearn_random_projection, only: sparse_random_projection
  use sklearn_sparse, only: csr_matrix,csr_from_arrays,csr_from_dense
  implicit none
  type(sparse_random_projection) :: model,dense
  type(csr_matrix) :: x,projected,large,query
  real(dp) :: a(4,9),densities(5)=[.001_dp,.2_dp,.99_dp,1._dp,-1._dp]
  real(dp), allocatable :: z(:,:),expected(:,:),back(:,:),components(:,:)
  integer, allocatable :: before(:),after(:),ptr(:)
  integer :: i,j,n
  character(32) :: mode
  call get_command_argument(1,mode)
  model%store_sparse_components=.true.
  if(trim(mode)=='memory') then
    allocate(ptr(4097));ptr=1
    large=csr_from_arrays(4096,2000000,ptr,[integer ::],[real(dp) ::])
    model%n_components=1024;model%density=1.e-6_dp;model%random_state=42
    projected=model%fit_transform_sparse(large)
    call require(.not.allocated(model%components_),'memory fit never allocates dense components')
    call require(.not.allocated(model%inverse_components_),'memory fit never allocates inverse')
    call model%components_csr_%validate();call projected%validate()
    call require(projected%n_rows==4096.and.projected%n_cols==1024.and.projected%nnz()==0, &
      'large sparse projection output')
    call require(model%components_csr_%nnz()>0,'large sparse projection sampled entries')
    query=csr_from_arrays(2,2000000,[1,2,2],[model%components_csr_%indices(1)],[1._dp])
    z=model%transform(query);projected=model%transform_sparse(query);expected=projected%to_dense()
    call require(all(z==expected),'large CSR components dense and sparse output agree')
    call require(any(z(1,:)/=0).and.all(z(2,:)==0),'large sparse query selects stored components')
    print *, 'Large sparse projection components memory case: PASS'
    stop
  end if
  call random_seed(size=n);allocate(before(n),after(n));call random_seed(get=before)
  a=reshape([(sin(real(i,dp)),i=1,size(a))],shape(a))
  x=csr_from_arrays(4,9,[1,4,4,6,7],[9,1,9,2,4,1],[2._dp,1._dp,-1._dp,3._dp,0._dp,-2._dp])
  model%n_components=5;dense%n_components=5
  model%compute_inverse_components=.true.;dense%compute_inverse_components=.true.
  do j=1,5
    model%density=densities(j);dense%density=densities(j)
    model%random_state=42;dense%random_state=42
    z=model%fit_transform(a);expected=dense%fit_transform(a)
    call require(.not.allocated(model%components_),'CSR storage omits dense components')
    call model%components_csr_%validate();components=model%components_csr_%to_dense()
    call require(all(components==dense%components_),'CSR storage preserves seeded components')
    call require(maxval(abs(z-expected))<1.e-13_dp,'dense input with CSR components')
    back=model%inverse_transform(z);expected=dense%inverse_transform(z)
    call require(all(back==expected),'cached dense inverse with CSR components')
    z=model%transform(x);expected=dense%transform(x)
    call require(all(z==expected),'noncanonical CSR input with CSR components')
    projected=model%transform_sparse(x);expected=projected%to_dense()
    call require(all(z==expected),'CSR components sparse and dense output agree')
    projected=model%fit_transform_sparse(x)
    call require(.not.allocated(model%components_),'CSR fit_transform retains sparse component storage')
    query=csr_from_dense(z);back=model%inverse_transform(query);expected=dense%inverse_transform(query)
    call require(all(back==expected),'cached CSR inverse with CSR components')
  end do
  model%compute_inverse_components=.false.;call model%fit(x)
  call require(.not.allocated(model%inverse_components_),'CSR component refit clears inverse')
  z=model%transform(x);back=model%inverse_transform(z);expected=dense%inverse_transform(z)
  call require(all(back==expected),'uncached dense inverse with CSR components')
  query=csr_from_dense(z);back=model%inverse_transform(query);expected=dense%inverse_transform(query)
  call require(all(back==expected),'uncached CSR inverse with CSR components')
  model%store_sparse_components=.false.;call model%fit(a)
  call require(allocated(model%components_),'refit can restore dense storage')
  call require(.not.allocated(model%components_csr_%indptr),'dense refit clears CSR component storage')
  model%store_sparse_components=.true.;model%density=1.e-310_dp;call model%fit(a)
  call require(model%components_csr_%nnz()==0,'empty CSR component selections')
  z=model%transform(a);call require(all(z==0),'empty CSR components transform to zero')
  z=model%transform(x);call require(all(z==0),'empty CSR components with CSR input')
  call random_seed(get=after);call require(all(before==after),'CSR component fits preserve intrinsic RNG')
  print *, 'Sparse random projection component storage and all transform paths: PASS'
end program
