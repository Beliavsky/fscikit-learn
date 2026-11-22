! sklearn 1.6.1 utils/extmath.py randomized_svd/range_finder defaults (BSD).
! Native pivoted LU, Householder QR and existing Jacobi SVD; no SciPy runtime.
module sklearn_randomized_svd_core
  use sklearn_common, only: dp,require,dense_svd,check_x
  use sklearn_legacy_rng, only: legacy_random_state
  implicit none
  private
  public :: randomized_svd
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
  subroutine randomized_svd(x,k,rng,s,u,vt,n_iter)
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: k
    integer, optional, intent(in) :: n_iter
    type(legacy_random_state), intent(inout) :: rng
    real(dp), allocatable, intent(out) :: s(:),u(:,:),vt(:,:)
    real(dp), allocatable :: a(:,:),q(:,:),b(:,:),left(:,:),right(:,:),values(:),work_u(:,:),work_v(:,:)
    integer :: i,j,iterations,m,p,r,keep,index
    logical :: transposed
    call check_x(x); call require(k>=1.and.k<=huge(k)-10,'invalid randomized SVD component count')
    transposed=size(x,1)<size(x,2)
    if(transposed) then
      a=transpose(x)
    else
      a=x
    end if
    m=size(a,1); p=size(a,2); allocate(q(p,k+10))
    do i=1,p
      do j=1,k+10
        q(i,j)=rng%normal()
      end do
    end do
    iterations=4
    if(k<.1_dp*min(size(x,1),size(x,2))) iterations=7
    if(present(n_iter)) then
      call require(n_iter>=0,'randomized SVD iteration count must be nonnegative');iterations=n_iter
    end if
    do i=1,iterations
      if(iterations<=2) then
        q=matmul(a,q);q=matmul(transpose(a),q)
      else
        q=lu_basis(matmul(a,q));q=lu_basis(matmul(transpose(a),q))
      end if
    end do
    q=qr_basis(matmul(a,q)); b=matmul(transpose(q),a); r=size(b,1)
    ! SVD of B.T keeps the Jacobi system at the sampled range dimension.
    allocate(values(r),work_u(p,r),work_v(r,r))
    call dense_svd(transpose(b),values,work_u,work_v)
    left=matmul(q,work_v); right=work_u; keep=min(k,r)
    do j=1,keep
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
