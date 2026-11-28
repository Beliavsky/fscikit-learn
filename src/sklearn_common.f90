module sklearn_common
  use, intrinsic :: iso_fortran_env, only: real64, int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  integer, parameter, public :: dp = real64
  public :: require, check_x, check_xy, unique_labels, label_index, argsort, seed_rng
  public :: solve_system, symmetric_eigen, logsumexp, sigmoid
  public :: cholesky_factor,solve_lower,solve_upper,normal_random
  public :: dense_svd,dense_svd_status
contains
  subroutine require(condition, message)
    logical, intent(in) :: condition
    character(*), intent(in) :: message
    if (.not. condition) then
      print '(a)', 'fscikit-learn: '//message
      error stop 1
    end if
  end subroutine

  subroutine check_x(x)
    real(dp), intent(in) :: x(:,:)
    call require(size(x,1)>0 .and. size(x,2)>0, 'X must be nonempty')
    call require(all(ieee_is_finite(x)), 'X contains NaN or infinity')
  end subroutine

  subroutine check_xy(x, n)
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: n
    call check_x(x)
    call require(size(x,1)==n, 'X and y sample counts differ')
  end subroutine

  function unique_labels(y) result(classes)
    integer, intent(in) :: y(:)
    integer, allocatable :: classes(:)
    integer :: tmp(size(y)), i,j,k,v
    call require(size(y)>0, 'labels must be nonempty')
    tmp=y
    do i=2,size(y)
      v=tmp(i); j=i-1
      do while(j>=1)
        if(tmp(j)<=v) exit
        tmp(j+1)=tmp(j); j=j-1
      end do
      tmp(j+1)=v
    end do
    k=1
    do i=2,size(y)
      if(tmp(i)/=tmp(k)) then
        k=k+1; tmp(k)=tmp(i)
      end if
    end do
    classes=tmp(:k)
  end function

  integer function label_index(classes, value) result(idx)
    integer, intent(in) :: classes(:), value
    integer :: i
    idx=0
    do i=1,size(classes)
      if(classes(i)==value) then
        idx=i; return
      end if
    end do
  end function

  function argsort(x) result(order)
    real(dp), intent(in) :: x(:)
    integer :: order(size(x)), i,j,v
    order=[(i,i=1,size(x))]
    do i=2,size(x)
      v=order(i); j=i-1
      do while(j>=1)
        if(x(order(j))<=x(v)) exit
        order(j+1)=order(j); j=j-1
      end do
      order(j+1)=v
    end do
  end function

  subroutine seed_rng(seed)
    integer, intent(in) :: seed
    integer, allocatable :: seeds(:)
    integer :: n,i
    integer(int64) :: state
    call random_seed(size=n)
    allocate(seeds(n)); state=modulo(int(seed,int64),2147483646_int64)+1
    do i=1,n
      state=modulo(16807_int64*state,2147483647_int64)
      seeds(i)=int(state)
    end do
    call random_seed(put=seeds)
  end subroutine

  function solve_system(a,b) result(x)
    ! Gaussian elimination with partial pivoting. Singular systems are rejected.
    real(dp), intent(in) :: a(:,:), b(:)
    real(dp) :: x(size(b)), work(size(b),size(b)), rhs(size(b)), row(size(b)), t, scale
    integer :: n,i,j,k,p
    n=size(b)
    call require(size(a,1)==n.and.size(a,2)==n, 'linear solve shape mismatch')
    work=a; rhs=b; scale=maxval(abs(a))
    do k=1,n
      p=k-1+maxloc(abs(work(k:n,k)),dim=1)
      call require(abs(work(p,k))>epsilon(1.0_dp)*max(scale,tiny(1.0_dp)), 'singular linear system')
      row=work(k,:); work(k,:)=work(p,:); work(p,:)=row
      t=rhs(k); rhs(k)=rhs(p); rhs(p)=t
      do i=k+1,n
        t=work(i,k)/work(k,k)
        work(i,k:n)=work(i,k:n)-t*work(k,k:n); rhs(i)=rhs(i)-t*rhs(k)
      end do
    end do
    do i=n,1,-1
      x(i)=(rhs(i)-dot_product(work(i,i+1:n),x(i+1:n)))/work(i,i)
    end do
  end function

  subroutine symmetric_eigen(a, values, vectors)
    ! Cyclic maximum-pivot Jacobi eigensolver; descending eigenvalues.
    real(dp), intent(in) :: a(:,:)
    real(dp), intent(out) :: values(:),vectors(:,:)
    real(dp) :: w(size(a,1),size(a,1)), c,s,t,tau,ap,aq,off,tol
    integer :: n,i,j,p,q,it,loc(2),order(size(a,1))
    n=size(a,1); call require(size(a,2)==n, 'eigen matrix must be square')
    call require(size(values)==n.and.all(shape(vectors)==[n,n]),'eigen output shape mismatch')
    w=a; vectors=0
    do i=1,n
      vectors(i,i)=1
    end do
    tol=epsilon(1.0_dp)*max(maxval(abs(a)),tiny(1.0_dp))*n
    do it=1,max(1,100*n*n)
      off=0; p=1; q=1
      do j=2,n
        do i=1,j-1
          if(abs(w(i,j))>off) then
            off=abs(w(i,j)); p=i; q=j
          end if
        end do
      end do
      if(off<=tol) exit
      tau=(w(q,q)-w(p,p))/(2*w(p,q))
      t=sign(1.0_dp,tau)/(abs(tau)+sqrt(1+tau*tau)); c=1/sqrt(1+t*t); s=t*c
      ap=w(p,p); aq=w(q,q)
      w(p,p)=ap-t*w(p,q); w(q,q)=aq+t*w(p,q); w(p,q)=0; w(q,p)=0
      do j=1,n
        if(j/=p.and.j/=q) then
          ap=w(j,p); aq=w(j,q)
          w(j,p)=c*ap-s*aq; w(p,j)=w(j,p)
          w(j,q)=s*ap+c*aq; w(q,j)=w(j,q)
        end if
        ap=vectors(j,p); aq=vectors(j,q)
        vectors(j,p)=c*ap-s*aq; vectors(j,q)=s*ap+c*aq
      end do
    end do
    call require(it<=max(1,100*n*n), 'Jacobi eigensolver did not converge')
    do i=1,n
      values(i)=w(i,i)
    end do
    order=argsort(-values); values=values(order); vectors=vectors(:,order)
  end subroutine

  real(dp) function logsumexp(x) result(v)
    real(dp), intent(in) :: x(:)
    real(dp) :: m
    m=maxval(x); v=m+log(sum(exp(x-m)))
  end function

  elemental real(dp) function sigmoid(x) result(v)
    real(dp), intent(in) :: x
    if(x>=0) then
      v=1/(1+exp(-x))
    else
      v=exp(x)/(1+exp(x))
    end if
  end function
  function cholesky_factor(a) result(l)
    real(dp), intent(in) :: a(:,:)
    real(dp) :: l(size(a,1),size(a,2)),v
    integer :: i,j,n
    n=size(a,1); call require(size(a,2)==n,'Cholesky requires square matrix'); l=0
    do i=1,n
      do j=1,i
        v=a(i,j)-dot_product(l(i,:j-1),l(j,:j-1))
        if(i==j) then
          call require(v>0,'matrix is not positive definite'); l(i,j)=sqrt(v)
        else
          l(i,j)=v/l(j,j)
        end if
      end do
    end do
  end function
  function solve_lower(l,b) result(x)
    real(dp), intent(in) :: l(:,:),b(:)
    real(dp) :: x(size(b))
    integer :: i,n
    n=size(b); call require(all(shape(l)==[n,n]),'lower solve shape mismatch')
    do i=1,n
      x(i)=(b(i)-dot_product(l(i,:i-1),x(:i-1)))/l(i,i)
    end do
  end function
  function solve_upper(u,b) result(x)
    real(dp), intent(in) :: u(:,:),b(:)
    real(dp) :: x(size(b))
    integer :: i,n
    n=size(b); call require(all(shape(u)==[n,n]),'upper solve shape mismatch')
    do i=n,1,-1
      x(i)=(b(i)-dot_product(u(i,i+1:),x(i+1:)))/u(i,i)
    end do
  end function
  function normal_random(n,p) result(x)
    integer, intent(in) :: n,p
    real(dp) :: x(n,p),u(n,p),v(n,p)
    call random_number(u); call random_number(v)
    x=sqrt(-2*log(max(u,tiny(1._dp))))*cos(2*acos(-1._dp)*v)
  end function
  subroutine dense_svd(a,s,u,v)
    real(dp),intent(in)::a(:,:)
    real(dp),intent(out)::s(:),u(:,:),v(:,:)
    logical::converged
    call dense_svd_status(a,s,u,v,converged)
    call require(converged,'Jacobi SVD did not converge')
  end subroutine

  subroutine dense_svd_status(a,s,u,v,converged)
    ! One-sided Jacobi SVD avoids squaring the condition number in A^T A.
    ! U has shape (m,n), V (n,n); zero singular columns of U are zero.
    real(dp), intent(in) :: a(:,:)
    real(dp), intent(out) :: s(:),u(:,:),v(:,:)
    logical,intent(out)::converged
    real(dp) :: work(size(a,1),size(a,2)),column(size(a,1)),right(size(a,2))
    real(dp) :: alpha,beta,gamma,tau,t,c,sine,cutoff,scale,threshold
    integer :: n,m,p,q,i,sweep,order(size(a,2))
    logical :: changed
    m=size(a,1); n=size(a,2)
    call require(size(s)==n.and.all(shape(u)==[m,n]).and.all(shape(v)==[n,n]),'SVD output shapes')
    work=a; v=0
    do i=1,n
      v(i,i)=1
    end do
    scale=maxval(sum(work**2,dim=1)); threshold=(epsilon(1._dp)*max(m,n))**2*scale
    do sweep=1,100
      changed=.false.
      do q=2,n
        do p=1,q-1
          alpha=sum(work(:,p)**2); beta=sum(work(:,q)**2)
          if(min(alpha,beta)<=max(threshold,tiny(1._dp))) cycle
          gamma=dot_product(work(:,p),work(:,q))
          if(abs(gamma)<=epsilon(1._dp)*sqrt(alpha)*sqrt(beta)*max(m,n)) cycle
          tau=(beta-alpha)/(2*gamma)
          t=sign(1._dp,tau)/(abs(tau)+sqrt(1+tau*tau)); c=1/sqrt(1+t*t); sine=t*c
          column=work(:,p); work(:,p)=c*column-sine*work(:,q); work(:,q)=sine*column+c*work(:,q)
          right=v(:,p); v(:,p)=c*right-sine*v(:,q); v(:,q)=sine*right+c*v(:,q)
          changed=.true.
        end do
      end do
      if(.not.changed) exit
    end do
    converged=.not.changed
    if(changed)return
    s=sqrt(sum(work**2,dim=1)); order=argsort(-s); s=s(order); work=work(:,order); v=v(:,order)
    u=0; cutoff=epsilon(1._dp)*max(m,n)*maxval(s)
    do i=1,n
      if(s(i)>cutoff) u(:,i)=work(:,i)/s(i)
      p=maxloc(abs(v(:,i)),dim=1)
      if(v(p,i)<0) then
        v(:,i)=-v(:,i); u(:,i)=-u(:,i)
      end if
    end do
  end subroutine
end module
