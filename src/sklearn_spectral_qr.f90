! sklearn.cluster._spectral.cluster_qr; pivoting follows LAPACK DLAQP2/DLARFG.
! Retained sklearn/LAPACK BSD notices: LICENSES. Typed integration is MIT.
module sklearn_spectral_qr
  use sklearn_common,only:dp,require,check_x,dense_svd
  use sklearn_kernel_dispatch,only:kernel_blas_dot
  implicit none
  private
  public::spectral_cluster_qr,spectral_qr_pivots
contains
  function product(a,b)result(c)
    real(dp),intent(in)::a(:,:),b(:,:)
    real(dp)::c(size(a,1),size(b,2))
    integer::i,j
    do j=1,size(b,2)
      do i=1,size(a,1)
        c(i,j)=kernel_blas_dot(a(i,:),b(:,j),size(a,1),size(b,2),i,j)
      end do
    end do
  end function
  real(dp)function vector_norm(x)result(value)
    real(dp),intent(in)::x(:)
    real(dp)::scale
    value=0
    if(size(x)==0)return
    scale=maxval(abs(x))
    if(scale==0)return
    if(scale<sqrt(tiny(1._dp)/epsilon(1._dp)).or.scale>sqrt(huge(1._dp)*epsilon(1._dp)))then
      value=scale*sqrt(sum((x/scale)**2))
    else
      value=sqrt(sum(x**2))
    end if
  end function

  subroutine reflector(x,tau)
    real(dp),intent(inout)::x(:)
    real(dp),intent(out)::tau
    real(dp)::norm,beta,alpha,safe,inverse
    integer::scaled
    tau=0
    if(size(x)<=1)return
    norm=vector_norm(x(2:));if(norm==0)return
    alpha=x(1);beta=-sign(norm2([alpha,norm]),alpha)
    safe=tiny(1._dp)/(epsilon(1._dp)*.5_dp);scaled=0
    if(abs(beta)<safe)then
      inverse=1/safe
      do
        scaled=scaled+1;x(2:)=x(2:)*inverse;alpha=alpha*inverse;beta=beta*inverse
        if(abs(beta)>=safe.or.scaled>=20)exit
      end do
      norm=vector_norm(x(2:));beta=-sign(norm2([alpha,norm]),alpha)
    end if
    tau=(beta-alpha)/beta;x(2:)=x(2:)*(1/(alpha-beta))
    do while(scaled>0)
      beta=beta*safe;scaled=scaled-1
    end do
    x(1)=beta
  end subroutine

  function spectral_qr_pivots(vectors)result(pivots)
    real(dp),intent(in)::vectors(:,:)
    integer,allocatable::pivots(:)
    real(dp),allocatable::a(:,:),vn1(:),vn2(:),column(:),v(:),work(:)
    real(dp)::tau,diagonal,temp,reliability,tolerance
    integer::m,n,i,j,p,index,lastv,lastc
    call check_x(vectors);m=size(vectors,2);n=size(vectors,1)
    a=transpose(vectors);allocate(vn1(n),vn2(n),column(m),pivots(n),work(n))
    do j=1,n
      pivots(j)=j;vn1(j)=vector_norm(a(:,j))
    end do
    vn2=vn1;tolerance=sqrt(epsilon(1._dp)*.5_dp)
    do i=1,min(m,n)
      p=i-1+maxloc(vn1(i:),dim=1)
      if(p/=i)then
        column=a(:,p);a(:,p)=a(:,i);a(:,i)=column
        index=pivots(p);pivots(p)=pivots(i);pivots(i)=index
        vn1(p)=vn1(i);vn2(p)=vn2(i)
      end if
      call reflector(a(i:,i),tau)
      if(i<n.and.tau/=0)then
        diagonal=a(i,i);a(i,i)=1;v=a(i:,i);lastv=size(v)
        do while(lastv>0)
          if(v(lastv)/=0)exit
          lastv=lastv-1
        end do
        lastc=n
        do while(lastc>i)
          if(any(a(i:i+lastv-1,lastc)/=0))exit
          lastc=lastc-1
        end do
        do j=i+1,lastc
          work(j)=kernel_blas_dot(a(i:i+lastv-1,j),v(:lastv),1,lastc-i,1,j-i)
        end do
        do j=i+1,lastc
          temp=-tau*work(j)
          a(i:i+lastv-1,j)=a(i:i+lastv-1,j)+v(:lastv)*temp
        end do
        a(i,i)=diagonal
      end if
      do j=i+1,n
        if(vn1(j)==0)cycle
        temp=max(0._dp,1-(abs(a(i,j))/vn1(j))**2)
        reliability=temp*(vn1(j)/vn2(j))**2
        if(reliability<=tolerance)then
          vn1(j)=vector_norm(a(i+1:,j));vn2(j)=vn1(j)
        else
          vn1(j)=vn1(j)*sqrt(temp)
        end if
      end do
    end do
  end function

  function spectral_cluster_qr(vectors,pivots)result(labels)
    real(dp),intent(in)::vectors(:,:)
    integer,allocatable,optional,intent(out)::pivots(:)
    integer::labels(size(vectors,1))
    integer,allocatable::order(:)
    real(dp),allocatable::s(:),u(:,:),v(:,:),selected(:,:),rotation(:,:),projected(:,:)
    integer::m,k
    call check_x(vectors);m=size(vectors,2);k=minval(shape(vectors))
    call require(m<=size(vectors,1),'cluster_qr requires no more components than samples')
    order=spectral_qr_pivots(vectors)
    selected=transpose(vectors(order(:k),:));allocate(s(k),u(m,k),v(k,k))
    call dense_svd(selected,s,u,v)
    rotation=product(u,transpose(v));projected=abs(product(vectors,rotation))
    labels=maxloc(projected,dim=2)-1
    if(present(pivots))pivots=order
  end function
end module
