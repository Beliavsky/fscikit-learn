program test_structured_datasets
  use sklearn_common, only: dp,require
  use sklearn_datasets, only: make_sparse_coded_signal,make_sparse_spd_matrix
  use sklearn_sparse, only: csr_matrix
  implicit none
  real(dp), allocatable :: x(:,:),d(:,:),code(:,:),a(:,:),b(:,:),factor(:,:)
  type(csr_matrix) :: csr
  integer, allocatable :: before(:),after(:)
  integer :: n,i,j,k
  real(dp) :: pivot
  call make_sparse_coded_signal(20,12,7,3,x,d,code,42)
  call require(all(shape(x)==[20,7]).and.all(shape(d)==[12,7]),'coded output orientation')
  call require(all(count(code/=0,dim=2)==3),'exact code support cardinality')
  call require(maxval(abs(sum(d*d,dim=2)-1))<1.e-14_dp,'unit dictionary atoms')
  call require(maxval(abs(x-matmul(code,d)))<1.e-14_dp,'signal reconstruction')
  call make_sparse_coded_signal(1,1,1,1,x,d,code,42)
  call require(all(shape(x)==[1,1]).and.abs(abs(d(1,1))-1)<1.e-14_dp,'singleton keeps rank two')
  call make_sparse_spd_matrix(a,n_dim=25,alpha=.2_dp,random_state=42)
  call make_sparse_spd_matrix(csr,n_dim=25,alpha=.2_dp,random_state=42)
  b=csr%to_dense();call require(all(a==b),'dense/CSR equality')
  call require(maxval(abs(a-transpose(a)))<1.e-14_dp,'SPD symmetry')
  factor=a
  do j=1,25
    pivot=factor(j,j)-sum(factor(j,:j-1)**2)
    call require(pivot>0,'positive Cholesky pivots');factor(j,j)=sqrt(pivot)
    do i=j+1,25
      factor(i,j)=(factor(i,j)-sum(factor(i,:j-1)*factor(j,:j-1)))/factor(j,j)
    end do
  end do
  call make_sparse_spd_matrix(a,n_dim=25,alpha=.2_dp,norm_diag=.true.,random_state=42)
  call require(maxval(abs([(a(k,k)-1,k=1,25)]))<1.e-14_dp,'normalized SPD diagonal')
  call make_sparse_spd_matrix(a,n_dim=7,alpha=1._dp,random_state=42)
  call require(sum(abs(a))==7._dp,'zero density identity')
  call make_sparse_spd_matrix(a);call require(all(shape(a)==[1,1]).and.a(1,1)==1,'default singleton')
  call random_seed(size=n);allocate(before(n),after(n));call random_seed(get=before)
  call make_sparse_coded_signal(4,8,2,5,x,d,code)
  call make_sparse_spd_matrix(a,n_dim=5)
  call random_seed(get=after);call require(all(before==after),'isolated RNG')
  print *, 'Sparse coded signals and dense/CSR positive definite samples: PASS'
end program
