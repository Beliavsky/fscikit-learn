! Hessian eigenmapping and modified LLE from sklearn 1.6.1
! manifold/_locally_linear.py (BSD-3-Clause). Internal dense alignment backend.
! Complete attribution and research references: THIRD_PARTY_NOTICES.md.
module sklearn_lle_variants
  use sklearn_common, only: dp,require,check_x,dense_svd,symmetric_eigen,argsort
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: lle_hessian_alignment,lle_modified_alignment,lle_local_spectra
contains
  real(dp) function norm2_scaled(x) result(value)
    real(dp), intent(in) :: x(:)
    value=maxval(abs(x))
    if(value>0) value=value*sqrt(sum((x/value)**2))
  end function
  subroutine check_alignment_input(x,neighbors,d)
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: neighbors(:,:),d
    call check_x(x)
    call require(size(neighbors,1)==size(x,1),'LLE alignment neighbor row count mismatch')
    call require(size(neighbors,2)>0,'LLE alignment neighborhoods must be nonempty')
    call require(all(neighbors>=1).and.all(neighbors<=size(x,1)),'LLE alignment neighbor index out of range')
    call require(d>=1.and.d<=size(x,2),'invalid LLE alignment dimension')
  end subroutine
  function full_qr(a) result(q)
    real(dp), intent(in) :: a(:,:)
    real(dp), allocatable :: q(:,:),work(:,:),reflectors(:,:),v(:),product(:),tau(:)
    real(dp) :: length,beta,first
    integer :: m,n,r,i,j
    m=size(a,1); n=size(a,2); r=min(m,n); work=a
    allocate(reflectors(m,r),tau(r),q(m,m)); reflectors=0; tau=0; q=0
    do j=1,r
      v=work(j:,j); length=norm2_scaled(v); first=v(1)
      if(length==0) cycle
      beta=-sign(length,first); v(1)=first-beta; v=v/v(1); tau(j)=(beta-first)/beta
      reflectors(j:,j)=v; product=matmul(v,work(j:,j:))
      work(j:,j:)=work(j:,j:)-tau(j)*spread(v,2,n-j+1)*spread(product,1,m-j+1)
    end do
    do i=1,m
      q(i,i)=1
    end do
    do j=r,1,-1
      if(tau(j)==0) cycle
      product=matmul(reflectors(j:,j),q(j:,:))
      q(j:,:)=q(j:,:)-tau(j)*spread(reflectors(j:,j),2,m)*spread(product,1,m-j+1)
    end do
  end function
  subroutine local_basis(local,full,values)
    real(dp), intent(in) :: local(:,:)
    real(dp), allocatable, intent(out) :: full(:,:),values(:)
    real(dp), allocatable :: thin(:,:),right(:,:),singular(:),candidate(:),best(:),all_values(:)
    real(dp) :: length,best_length,coefficient
    integer :: k,p,i,j,c,pass
    k=size(local,1); p=size(local,2)
    allocate(full(k,k),values(min(k,p))); full=0
    if(k<=p) then
      allocate(all_values(k)); call symmetric_eigen(matmul(local,transpose(local)),all_values,full)
      values=all_values; return
    end if
    allocate(thin(k,p),right(p,p),singular(p))
    call dense_svd(local,singular,thin,right); values=singular**2
    ! Complete the thin Jacobi left singular vectors to an orthonormal basis.
    ! Null-space frames are not unique; the normal modified-LLE construction
    ! depends on their projector rather than their orientation.
    allocate(candidate(k),best(k))
    do j=1,k
      length=0
      if(j<=p) then
        candidate=thin(:,j); length=norm2_scaled(candidate)
      end if
      if(length==0) then
        best_length=-1
        do c=1,k
          candidate=0; candidate(c)=1
          do pass=1,2
            do i=1,j-1
              coefficient=dot_product(full(:,i),candidate); candidate=candidate-coefficient*full(:,i)
            end do
          end do
          length=norm2_scaled(candidate)
          if(length>best_length) then
            best_length=length; best=candidate
          end if
        end do
        candidate=best
      else
        do pass=1,2
          do i=1,j-1
            coefficient=dot_product(full(:,i),candidate); candidate=candidate-coefficient*full(:,i)
          end do
        end do
      end if
      length=norm2_scaled(candidate)
      call require(length>epsilon(1._dp),'LLE local orthogonal completion failed')
      full(:,j)=candidate/length
    end do
  end subroutine
  subroutine lle_local_spectra(x,neighbors,v,values)
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: neighbors(:,:)
    real(dp), allocatable, intent(out) :: v(:,:,:),values(:,:)
    real(dp), allocatable :: local(:,:),basis(:,:),spectrum(:)
    integer :: n,k,nev,i
    call check_alignment_input(x,neighbors,1)
    n=size(x,1); k=size(neighbors,2); nev=min(k,size(x,2)); allocate(v(n,k,k),values(n,nev))
    do i=1,n
      local=x(neighbors(i,:),:)-spread(x(i,:),1,k)
      call local_basis(local,basis,spectrum); v(i,:,:)=basis; values(i,:)=spectrum
    end do
  end subroutine
  function lle_hessian_alignment(x,neighbors,d,hessian_tol) result(alignment)
    real(dp), intent(in) :: x(:,:),hessian_tol
    integer, intent(in) :: neighbors(:,:),d
    real(dp), allocatable :: alignment(:,:),local(:,:),basis(:,:),values(:),yi(:,:),q(:,:),w(:,:),block(:,:)
    real(dp) :: mass
    integer(int64) :: quadratic
    integer :: n,k,i,j,a,b,column
    call check_alignment_input(x,neighbors,d); n=size(x,1); k=size(neighbors,2)
    call require(ieee_is_finite(hessian_tol),'LLE hessian_tol must be finite')
    call require(hessian_tol>=0,'LLE hessian_tol must be nonnegative')
    quadratic=int(d,int64)*(int(d,int64)+1)/2
    call require(int(k,int64)>int(d,int64)+quadratic,'Hessian LLE needs n_neighbors > d*(d+3)/2')
    allocate(alignment(n,n),yi(k,1+d+int(quadratic))); alignment=0; yi(:,1)=1
    do i=1,n
      local=x(neighbors(i,:),:); local=local-spread(sum(local,dim=1)/k,1,k)
      call local_basis(local,basis,values); yi(:,2:1+d)=basis(:,:d); column=1+d
      do a=1,d
        do b=a,d
          column=column+1; yi(:,column)=basis(:,a)*basis(:,b)
        end do
      end do
      q=full_qr(yi); w=q(:,d+2:)
      do j=1,size(w,2)
        mass=sum(w(:,j))
        if(abs(mass)<hessian_tol) mass=1
        call require(mass/=0,'Hessian LLE normalization sum is zero')
        w(:,j)=w(:,j)/mass
      end do
      block=matmul(w,transpose(w))
      do a=1,k
        do b=1,k
          alignment(neighbors(i,a),neighbors(i,b))=alignment(neighbors(i,a),neighbors(i,b))+block(a,b)
        end do
      end do
    end do
  end function
  function lle_modified_alignment(x,neighbors,d,modified_tol,nullity,weights,eta_out) result(alignment)
    real(dp), intent(in) :: x(:,:),modified_tol
    integer, intent(in) :: neighbors(:,:),d
    integer, allocatable, intent(out), optional :: nullity(:)
    real(dp), allocatable, intent(out), optional :: weights(:,:)
    real(dp), intent(out), optional :: eta_out
    real(dp), allocatable :: alignment(:,:),v(:,:,:),values(:,:),regularized(:,:),rho(:),ratios(:),vi(:,:)
    real(dp), allocatable :: projection(:),w(:,:),h(:),block(:,:),row_sum(:),median_values(:)
    integer, allocatable :: sizes(:),order(:)
    real(dp) :: regularization,denominator,cumulative,total,eta,alpha,length
    integer :: n,k,nev,i,j,a,b,s,left,right,middle
    call check_alignment_input(x,neighbors,d); n=size(x,1); k=size(neighbors,2)
    call require(k>=d,'modified LLE needs n_neighbors >= n_components')
    call require(ieee_is_finite(modified_tol),'LLE modified_tol must be finite')
    call require(modified_tol>=0,'LLE modified_tol must be nonnegative')
    call lle_local_spectra(x,neighbors,v,values); nev=size(values,2)
    allocate(regularized(n,k),rho(n),sizes(n),ratios(max(0,nev-1)),alignment(n,n)); alignment=0
    do i=1,n
      regularization=.001_dp*sum(values(i,:))
      call require(regularization>0,'modified LLE local spectrum must have positive trace')
      projection=matmul(transpose(v(i,:,:)),spread(1._dp,1,k))
      projection(:nev)=projection(:nev)/(values(i,:)+regularization)
      projection(nev+1:)=projection(nev+1:)/regularization
      regularized(i,:)=matmul(v(i,:,:),projection); denominator=sum(regularized(i,:))
      call require(abs(denominator)>tiny(1._dp),'modified LLE regularized weight sum is zero')
      regularized(i,:)=regularized(i,:)/denominator; total=sum(values(i,:d))
      call require(total>0,'modified LLE leading spectrum must be positive')
      rho(i)=sum(values(i,d+1:))/total
    end do
    order=argsort(rho); median_values=rho(order)
    if(mod(n,2)==1) then
      eta=median_values((n+1)/2)
    else
      eta=(median_values(n/2)+median_values(n/2+1))/2
    end if
    do i=1,n
      total=0
      do j=1,nev
        total=total+values(i,j)
      end do
      cumulative=0
      do j=1,nev-1
        cumulative=cumulative+values(i,j)
        call require(cumulative>0,'modified LLE cumulative spectrum must be positive')
        ratios(nev-j)=total/cumulative-1
      end do
      left=1; right=nev
      do while(left<right)
        middle=left+(right-left)/2
        if(ratios(middle)<eta) then
          left=middle+1
        else
          right=middle
        end if
      end do
      s=k-nev+left-1
      call require(s>0,'modified LLE selected an empty local nullspace')
      sizes(i)=s; vi=v(i,:,k-s+1:); projection=sum(vi,dim=1)
      alpha=norm2_scaled(projection)/sqrt(real(s,dp)); h=alpha-projection; length=norm2_scaled(h)
      if(length<modified_tol) then
        h=0
      else
        call require(length>0,'modified LLE Householder norm is zero')
        h=h/length
      end if
      w=vi-2*spread(matmul(vi,h),2,s)*spread(h,1,k)+(1-alpha)*spread(regularized(i,:),2,s)
      block=matmul(w,transpose(w)); row_sum=sum(w,dim=2)
      do a=1,k
        do b=1,k
          alignment(neighbors(i,a),neighbors(i,b))=alignment(neighbors(i,a),neighbors(i,b))+block(a,b)
        end do
        alignment(i,neighbors(i,a))=alignment(i,neighbors(i,a))-row_sum(a)
        alignment(neighbors(i,a),i)=alignment(neighbors(i,a),i)-row_sum(a)
      end do
      alignment(i,i)=alignment(i,i)+s
    end do
    if(present(nullity)) nullity=sizes
    if(present(weights)) weights=regularized
    if(present(eta_out)) eta_out=eta
  end function
end module
