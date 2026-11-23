! Dense Lawson-Hanson NNLS, following SciPy 1.15.3 optimize/_cython_nnls.pyx.
! BSD provenance and LAPACK reflector/rotation conventions: THIRD_PARTY_NOTICES.md.
module sklearn_nnls
  use sklearn_common, only: dp,require,check_xy,check_x
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: nnls_solve,positive_least_squares
contains
  real(dp) function quotient(a,b) result(c)
    real(dp), intent(in) :: a,b
    call require(b/=0,'NNLS zero triangular diagonal')
    if(abs(b)<1) call require(abs(a)<=huge(1._dp)*abs(b)/16,'NNLS coefficient exceeds working range')
    c=a/b
  end function
  subroutine reflector(v,beta,tau)
    real(dp), intent(inout) :: v(:)
    real(dp), intent(out) :: beta,tau
    real(dp) :: scale,head,tail,diagonal,denominator
    scale=maxval(abs(v));beta=0;tau=0
    if(scale==0) then
      v=0;v(1)=1;return
    end if
    v=v/scale;head=v(1);tail=norm2(v(2:));beta=head*scale
    if(tail<=epsilon(1._dp)*abs(head)) then
      if(head<0) then
        tau=2;v(2:)=0;beta=-beta
      end if
    else
      diagonal=sign(norm2(v),head);denominator=head+diagonal
      if(diagonal<0) then
        beta=-diagonal*scale;tau=-denominator/(-diagonal)
      else
        beta=diagonal*scale;denominator=tail*(tail/denominator)
        tau=denominator/diagonal;denominator=-denominator
      end if
      v(2:)=v(2:)/denominator
    end if
    v(1)=1
  end subroutine
  subroutine reflect(v,tau,b)
    real(dp), intent(in) :: v(:),tau
    real(dp), intent(inout) :: b(:)
    if(tau/=0) b=b-(tau*dot_product(v,b))*v
  end subroutine
  subroutine triangular(a,b,indices,count,z)
    real(dp), intent(in) :: a(:,:),b(:)
    integer, intent(in) :: indices(:),count
    real(dp), intent(out) :: z(:)
    integer :: k,ip,jj
    z=b;jj=0
    do k=0,count-1
      ip=count-k
      if(k/=0) then
        call require(all(abs(a(:ip,jj))<=huge(1._dp)/max(1._dp,abs(z(ip+1)))/32), &
          'NNLS back substitution exceeds working range')
        z(:ip)=z(:ip)-a(:ip,jj)*z(ip+1)
      end if
      jj=indices(ip);z(ip)=quotient(z(ip),a(ip,jj))
    end do
  end subroutine
  subroutine nnls_solve(design,target,coef,residual,iterations,converged,max_iter)
    real(dp), intent(in) :: design(:,:),target(:)
    real(dp), allocatable, intent(out) :: coef(:)
    real(dp), intent(out) :: residual
    integer, intent(out) :: iterations
    logical, intent(out) :: converged
    integer, optional, intent(in) :: max_iter
    real(dp), allocatable :: a(:,:),b(:),dual(:),work(:),z(:)
    integer, allocatable :: indices(:)
    integer :: m,n,nset,first,row,col,j,k,ip,iz,jj,ii,i,limit
    real(dp) :: tau,beta,unorm,wmax,test,alpha,t,cc,ss,r,temp,range
    logical :: skip,found
    call check_xy(design,size(target));call check_x(reshape(target,[size(target),1]))
    m=size(design,1);n=size(design,2)
    call require(n<=huge(1)/3,'NNLS feature count exceeds iteration range')
    range=sqrt(huge(1._dp))/(128*real(max(m,n),dp))
    call require(maxval(abs(design))<=range.and.maxval(abs(target))<=range,'NNLS data exceeds working range')
    limit=3*n;if(present(max_iter)) limit=max_iter
    call require(limit>0,'NNLS iteration limit must be positive')
    a=design;b=target;allocate(coef(n),dual(n),work(m),z(m));coef=0;dual=0
    indices=[(j,j=1,n)];nset=0;first=1;row=1;iterations=0;skip=.false.;converged=.false.
    do while(first<=n.and.nset<m)
      if(skip) then
        skip=.false.
      else
        do col=first,n
          j=indices(col);dual(j)=dot_product(b(row:),a(row:,j))
        end do
      end if
      wmax=0;iz=first
      do col=first,n
        j=indices(col)
        if(dual(j)>wmax) then
          wmax=dual(j);iz=col
        end if
      end do
      if(wmax<=0) exit
      j=indices(iz);work(row:)=a(row:,j);call reflector(work(row:),beta,tau)
      unorm=norm2(a(:nset,j))
      if((unorm+abs(beta)*.01_dp)-unorm<=0) then
        dual(j)=0;skip=.true.;cycle
      end if
      z=b;call reflect(work(row:),tau,z(row:));test=quotient(z(row),beta)
      if(test<=0) then
        dual(j)=0;skip=.true.;cycle
      end if
      a(row,j)=beta;b=z;indices(iz)=indices(first);indices(first)=j
      first=first+1;nset=nset+1
      do col=first,n
        k=indices(col);call reflect(work(row:),tau,a(row:,k))
      end do
      row=row+1
      ! All entries below the accepted pivot belong to the zero triangular
      ! tail, including the single remaining row. SciPy 1.15.3 leaves that
      ! last entry stale; later column removal can then violate NNLS KKT.
      if(nset<m) a(row:,j)=0
      dual(j)=0;call triangular(a,b,indices,nset,z)
      do
        iterations=iterations+1
        if(iterations==limit) then
          residual=0;return
        end if
        alpha=2;jj=1
        do ip=1,nset
          k=indices(ip)
          if(z(ip)<=0) then
            t=quotient(-coef(k),z(ip)-coef(k))
            if(alpha>t) then
              alpha=t;jj=ip
            end if
          end if
        end do
        if(alpha==2) exit
        coef(indices(:nset))=(1-alpha)*coef(indices(:nset))+alpha*z(:nset)
        i=indices(jj)
        do
          coef(i)=0
          if(jj/=nset+1) then
            jj=jj+1
            do j=jj,nset
              ii=indices(j);indices(j-1)=ii
              temp=a(j-1,ii);r=norm2([temp,a(j,ii)])
              cc=1;ss=0
              if(r>0) then
                cc=temp/r;ss=a(j,ii)/r
              end if
              a(j-1,ii)=r;a(j,ii)=0
              do col=1,n
                if(col==ii) cycle
                temp=a(j-1,col);a(j-1,col)=cc*temp+ss*a(j,col);a(j,col)=-ss*temp+cc*a(j,col)
              end do
              temp=b(j-1);b(j-1)=cc*temp+ss*b(j);b(j)=-ss*temp+cc*b(j)
            end do
          end if
          row=row-1;nset=nset-1;first=first-1;indices(first)=i;found=.false.
          do jj=1,nset
            i=indices(jj)
            if(coef(i)<=0) then
              found=.true.;exit
            end if
          end do
          if(.not.found) exit
        end do
        call triangular(a,b,indices,nset,z)
      end do
      coef(indices(:nset))=z(:nset)
    end do
    residual=norm2(b(row:));converged=.true.
    call require(all(ieee_is_finite(coef)),'NNLS coefficients must be finite')
  end subroutine
  subroutine positive_least_squares(a,b,alpha,coef)
    real(dp), intent(in) :: a(:,:),b(:),alpha
    real(dp), allocatable, intent(out) :: coef(:)
    real(dp), allocatable :: design(:,:),target(:)
    real(dp) :: residual
    integer :: p,j,iterations
    logical :: converged
    call require(ieee_is_finite(alpha),'positive linear alpha must be finite')
    call require(alpha>=0,'positive linear alpha must be nonnegative')
    p=size(a,2)
    if(alpha>0) then
      call require(size(a,1)<=huge(1)-p,'positive linear augmented design exceeds dimension range')
      allocate(design(size(a,1)+p,p),target(size(b)+p));design=0;target=0
      design(:size(a,1),:)=a;target(:size(b))=b
      do j=1,p
        design(size(a,1)+j,j)=sqrt(alpha)
      end do
      call nnls_solve(design,target,coef,residual,iterations,converged)
    else
      call nnls_solve(a,b,coef,residual,iterations,converged)
    end if
    call require(converged,'positive linear NNLS reached its iteration limit')
  end subroutine
end module
