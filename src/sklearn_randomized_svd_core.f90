! sklearn 1.6.1 utils/extmath.py randomized_svd/range_finder defaults (BSD).
! Native pivoted LU, Householder QR and existing Jacobi SVD; no SciPy runtime.
module sklearn_randomized_svd_core
  use sklearn_common, only: dp,require,dense_svd,check_x
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_sparse, only: csr_matrix
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: randomized_svd
  interface randomized_svd
    module procedure randomized_svd_dense,randomized_svd_sparse
  end interface
contains
  function lu_basis(a) result(basis)
    real(dp), intent(in) :: a(:,:)
    real(dp), allocatable :: basis(:,:),work(:,:),row(:)
    integer :: permutation(size(a,1)),m,n,r,j,i,p,temp
    m=size(a,1); n=size(a,2); r=min(m,n); work=a; permutation=[(i,i=1,m)]
    do j=1,r
      p=j-1+maxloc(abs(work(j:,j)),dim=1)
      if(p/=j) then
        row=work(j,:); work(j,:)=work(p,:); work(p,:)=row
        temp=permutation(j); permutation(j)=permutation(p); permutation(p)=temp
      end if
      if(work(j,j)==0) cycle
      do i=j+1,m
        work(i,j)=work(i,j)/work(j,j)
        work(i,j+1:)=work(i,j+1:)-work(i,j)*work(j,j+1:)
      end do
    end do
    allocate(basis(m,r)); basis=0
    do j=1,r
      basis(permutation(j),j)=1
      do i=j+1,m
        basis(permutation(i),j)=work(i,j)
      end do
    end do
  end function
  function qr_basis(a) result(q)
    real(dp), intent(in) :: a(:,:)
    real(dp), allocatable :: q(:,:),work(:,:),vectors(:,:),v(:),product(:),tau(:)
    real(dp) :: norm,beta,first
    integer :: m,n,r,j,i
    m=size(a,1); n=size(a,2); r=min(m,n); work=a
    allocate(vectors(m,r),tau(r),q(m,r)); vectors=0; tau=0; q=0
    do j=1,r
      v=work(j:,j); norm=sqrt(sum(v*v)); first=v(1)
      if(norm==0) cycle
      beta=-sign(norm,first); v(1)=first-beta; v=v/v(1); tau(j)=(beta-first)/beta
      vectors(j:,j)=v
      product=matmul(v,work(j:,j:)); work(j:,j:)=work(j:,j:)-tau(j)*spread(v,2,n-j+1)*spread(product,1,m-j+1)
    end do
    do j=1,r
      q(j,j)=1
    end do
    do j=r,1,-1
      if(tau(j)==0) cycle
      product=matmul(vectors(j:,j),q(j:,:))
      q(j:,:)=q(j:,:)-tau(j)*spread(vectors(j:,j),2,r)*spread(product,1,m-j+1)
    end do
  end function
  subroutine randomized_svd_dense(x,k,rng,s,u,vt,n_iter,n_oversamples,power_iteration_normalizer,flip_sign)
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: k
    integer, optional, intent(in) :: n_iter,n_oversamples
    character(*), optional, intent(in) :: power_iteration_normalizer
    logical, optional, intent(in) :: flip_sign
    type(legacy_random_state), intent(inout) :: rng
    real(dp), allocatable, intent(out) :: s(:),u(:,:),vt(:,:)
    call check_x(x)
    call randomized_svd_body(size(x,1),size(x,2),k,rng,s,u,vt,n_iter,n_oversamples, &
      power_iteration_normalizer,flip_sign,x=x)
  end subroutine
  subroutine randomized_svd_sparse(x,k,rng,s,u,vt,n_iter,n_oversamples,power_iteration_normalizer,flip_sign)
    type(csr_matrix), intent(in) :: x
    integer, intent(in) :: k
    integer, optional, intent(in) :: n_iter,n_oversamples
    character(*), optional, intent(in) :: power_iteration_normalizer
    logical, optional, intent(in) :: flip_sign
    type(legacy_random_state), intent(inout) :: rng
    real(dp), allocatable, intent(out) :: s(:),u(:,:),vt(:,:)
    call x%validate()
    call require(x%n_rows>0.and.x%n_cols>0,'randomized SVD CSR shape must be nonempty')
    call require(all(ieee_is_finite(x%data)),'randomized SVD CSR data must be finite')
    call randomized_svd_body(x%n_rows,x%n_cols,k,rng,s,u,vt,n_iter,n_oversamples, &
      power_iteration_normalizer,flip_sign,sparse_x=x)
  end subroutine
  function csr_product(x,right,transposed) result(product)
    type(csr_matrix), intent(in) :: x
    real(dp), intent(in) :: right(:,:)
    logical, intent(in) :: transposed
    real(dp), allocatable :: product(:,:)
    integer :: i,j,e
    if(.not.transposed) then
      product=x%matmul(right)
    else
      call require(size(right,1)==x%n_rows,'CSR transpose product dimension mismatch')
      allocate(product(x%n_cols,size(right,2)));product=0
      do j=1,size(right,2)
        do i=1,x%n_rows
          do e=x%indptr(i),x%indptr(i+1)-1
            product(x%indices(e),j)=product(x%indices(e),j)+x%data(e)*right(i,j)
          end do
        end do
      end do
    end if
  end function
  subroutine randomized_svd_body(n_rows,n_cols,k,rng,s,u,vt,n_iter,n_oversamples,power_iteration_normalizer,flip_sign, &
      x,sparse_x)
    real(dp), optional, intent(in) :: x(:,:)
    type(csr_matrix), optional, intent(in) :: sparse_x
    integer, intent(in) :: n_rows,n_cols
    integer, intent(in) :: k
    integer, optional, intent(in) :: n_iter,n_oversamples
    character(*), optional, intent(in) :: power_iteration_normalizer
    logical, optional, intent(in) :: flip_sign
    type(legacy_random_state), intent(inout) :: rng
    real(dp), allocatable, intent(out) :: s(:),u(:,:),vt(:,:)
    real(dp), allocatable :: a(:,:),q(:,:),b(:,:),left(:,:),right(:,:),values(:),work_u(:,:),work_v(:,:),next(:,:)
    integer :: i,j,iterations,m,p,r,keep,index,extra
    character(8) :: normalizer
    logical :: orient
    logical :: transposed
    extra=10;if(present(n_oversamples)) extra=n_oversamples
    call require(extra>=0,'randomized SVD oversampling must be nonnegative')
    call require(k>=1.and.k<=huge(k)-extra,'invalid randomized SVD component count')
    if(present(n_iter)) call require(n_iter>=0,'randomized SVD iteration count must be nonnegative')
    normalizer='auto';if(present(power_iteration_normalizer)) normalizer=power_iteration_normalizer
    call require(any(normalizer==[character(8)::'auto','none','LU','QR']),'unsupported randomized SVD normalizer')
    orient=.true.;if(present(flip_sign)) orient=flip_sign
    transposed=n_rows<n_cols
    if(present(x)) then
      if(transposed) then
        a=transpose(x)
      else
        a=x
      end if
    end if
    m=max(n_rows,n_cols);p=min(n_rows,n_cols);allocate(q(p,k+extra))
    do i=1,p
      do j=1,k+extra
        q(i,j)=rng%normal()
      end do
    end do
    iterations=4
    if(k<.1_dp*min(n_rows,n_cols)) iterations=7
    if(present(n_iter)) iterations=n_iter
    if(normalizer=='auto') then
      normalizer='LU';if(iterations<=2) normalizer='none'
    end if
    do i=1,iterations
      if(present(sparse_x)) then
        next=csr_product(sparse_x,q,transposed)
      else
        next=matmul(a,q)
      end if
      select case(normalizer)
      case('none');q=next
      case('LU');q=lu_basis(next)
      case('QR');q=qr_basis(next)
      end select
      if(present(sparse_x)) then
        next=csr_product(sparse_x,q,.not.transposed)
      else
        next=matmul(transpose(a),q)
      end if
      select case(normalizer)
      case('none');q=next
      case('LU');q=lu_basis(next)
      case('QR');q=qr_basis(next)
      end select
    end do
    if(present(sparse_x)) then
      next=csr_product(sparse_x,q,transposed);q=qr_basis(next)
      b=transpose(csr_product(sparse_x,q,.not.transposed))
    else
      q=qr_basis(matmul(a,q));b=matmul(transpose(q),a)
    end if
    r=size(b,1)
    ! SVD of B.T keeps the Jacobi system at the sampled range dimension.
    allocate(values(r),work_u(p,r),work_v(r,r))
    call dense_svd(transpose(b),values,work_u,work_v)
    left=matmul(q,work_v); right=work_u; keep=min(k,r,p)
    do j=1,keep
      if(.not.orient) cycle
      if(transposed) then
        index=maxloc(abs(right(:,j)),dim=1)
        if(right(index,j)>=0) cycle
      else
        index=maxloc(abs(left(:,j)),dim=1)
        if(left(index,j)>=0) cycle
      end if
      left(:,j)=-left(:,j); right(:,j)=-right(:,j)
    end do
    s=values(:keep)
    if(transposed) then
      u=right(:,:keep); vt=transpose(left(:,:keep))
    else
      u=left(:,:keep); vt=transpose(right(:,:keep))
    end if
  end subroutine
end module
