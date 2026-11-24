! sklearn 1.6.1 datasets/_samples_generator.py bicluster contracts (BSD).
! NumPy 2.2.6 binomial inversion/BTPE kernels (BSD; retained distribution license).
module sklearn_bicluster_datasets
  use sklearn_common, only: dp,require
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_ieee_math, only: ieee_product,ieee_difference
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  private
  public :: make_biclusters,make_checkerboard
  interface make_checkerboard
    module procedure checker_scalar,checker_pair
  end interface
contains
  integer function inversion(n,p,rng) result(y)
    integer, intent(in) :: n
    real(dp), intent(in) :: p
    type(legacy_random_state), intent(inout) :: rng
    real(dp) :: q,qn,np,px,u
    integer :: bound
    q=1-p;qn=exp(real(n,dp)*log(q));np=n*p
    bound=int(min(real(n,dp),np+10*sqrt(np*q+1)));y=0;px=qn;u=rng%uniform()
    do while(u>px)
      y=y+1
      if(y>bound) then
        y=0;px=qn;u=rng%uniform()
      else
        u=u-px;px=((n-y+1)*p*px)/(y*q)
      end if
    end do
  end function
  real(dp) function stirling(v) result(value)
    real(dp), intent(in) :: v
    real(dp) :: square
    square=v*v
    value=(13680._dp-(462._dp-(132._dp-(99._dp-140._dp/square)/square)/square)/square)/v/166320._dp
  end function
  integer function btpe(n,r,rng) result(y)
    integer, intent(in) :: n
    real(dp), intent(in) :: r
    type(legacy_random_state), intent(inout) :: rng
    real(dp) :: q,fm,p1,xm,xl,xr,c,laml,lamr,p2,p3,p4,a,u,v,s,f,rho,t,aval,nrq,x1,f1,z,w,x
    integer :: m,k,i
    q=1-r;fm=n*r+r;m=floor(fm);p1=floor(2.195_dp*sqrt(n*r*q)-4.6_dp*q)+.5_dp
    xm=m+.5_dp;xl=xm-p1;xr=xm+p1;c=.134_dp+20.5_dp/(15.3_dp+m)
    a=(fm-xl)/(fm-xl*r);laml=a*(1+a/2)
    a=(xr-fm)/(xr*q);lamr=a*(1+a/2)
    p2=p1*(1+2*c);p3=p2+c/laml;p4=p3+c/lamr
10  nrq=n*r*q;u=rng%uniform()*p4;v=rng%uniform()
    if(u>p1) goto 20
    y=floor(xm-p1*v+u);return
20  if(u>p2) goto 30
    x=xl+(u-p1)/c;v=v*c+1-abs(m-x+.5_dp)/p1
    if(v>1) goto 10
    y=floor(x);goto 50
30  if(u>p3) goto 40
    if(v==0) goto 10
    y=floor(xl+log(v)/laml)
    if(y<0) goto 10
    v=v*(u-p2)*laml;goto 50
40  if(v==0) goto 10
    y=floor(xr-log(v)/lamr)
    if(y>n) goto 10
    v=v*(u-p3)*lamr
50  k=abs(y-m)
    if(k>20.and.real(k,dp)<nrq/2-1) goto 52
    s=r/q;a=s*(real(n,dp)+1);f=1
    if(m<y) then
      do i=m+1,y
        f=f*(a/i-s)
      end do
    else if(m>y) then
      do i=y+1,m
        f=f/(a/i-s)
      end do
    end if
    if(v>f) goto 10
    return
52  rho=(k/nrq)*((k*(k/3._dp+.625_dp)+.16666666666666666_dp)/nrq+.5_dp)
    t=-real(k,dp)*k/(2*nrq)
    if(v==0) return
    aval=log(v)
    if(aval<t-rho) return
    if(aval>t+rho) goto 10
    x1=real(y,dp)+1;f1=real(m,dp)+1;z=real(n,dp)+1-m;w=real(n,dp)-y+1
    if(aval>xm*log(f1/x1)+(real(n-m,dp)+.5_dp)*log(z/w)+ &
        (y-m)*log(w*r/(x1*q))+stirling(f1)+stirling(z)+stirling(x1)+stirling(w)) goto 10
  end function
  integer function binomial(n,p,rng) result(y)
    integer, intent(in) :: n
    real(dp), intent(in) :: p
    type(legacy_random_state), intent(inout) :: rng
    real(dp) :: r
    y=0;if(n==0.or.p==0) return
    r=min(p,1-p)
    if(r*n<=30) then
      y=inversion(n,r,rng)
    else
      y=btpe(n,r,rng)
    end if
    if(p>.5_dp) y=n-y
  end function
  function partition(n,k,rng) result(labels)
    integer, intent(in) :: n,k
    type(legacy_random_state), intent(inout) :: rng
    integer, allocatable :: labels(:),counts(:)
    real(dp) :: probability,remaining_p
    integer :: remaining,i,first
    allocate(labels(n),counts(k));counts=0;remaining=n;remaining_p=1;probability=1._dp/k
    do i=1,k-1
      counts(i)=binomial(remaining,probability/remaining_p,rng);remaining=remaining-counts(i)
      if(remaining<=0) exit
      remaining_p=remaining_p-probability
    end do
    if(remaining>0) counts(k)=remaining
    first=1
    do i=1,k
      labels(first:first+counts(i)-1)=i-1;first=first+counts(i)
    end do
  end function
  function permutation(n,rng) result(order)
    integer, intent(in) :: n
    type(legacy_random_state), intent(inout) :: rng
    integer :: order(n),i,j,temp
    order=[(i,i=1,n)]
    do i=n,2,-1
      j=rng%integer_index(i);temp=order(i);order(i)=order(j);order(j)=temp
    end do
  end function
  subroutine make_biclusters(shape,n_clusters,data,rows,cols,noise,minval,maxval,shuffle,random_state)
    integer, intent(in) :: shape(:),n_clusters
    real(dp), allocatable, intent(out) :: data(:,:)
    logical, allocatable, intent(out) :: rows(:,:),cols(:,:)
    real(dp), optional, intent(in) :: noise,minval,maxval
    logical, optional, intent(in) :: shuffle
    integer, optional, intent(in) :: random_state
    call generate(shape,n_clusters,n_clusters,.true.,data,rows,cols,noise,minval,maxval,shuffle,random_state)
  end subroutine
  subroutine checker_scalar(shape,n_clusters,data,rows,cols,noise,minval,maxval,shuffle,random_state)
    integer, intent(in) :: shape(:),n_clusters
    real(dp), allocatable, intent(out) :: data(:,:)
    logical, allocatable, intent(out) :: rows(:,:),cols(:,:)
    real(dp), optional, intent(in) :: noise,minval,maxval
    logical, optional, intent(in) :: shuffle
    integer, optional, intent(in) :: random_state
    call generate(shape,n_clusters,n_clusters,.false.,data,rows,cols,noise,minval,maxval,shuffle,random_state)
  end subroutine
  subroutine checker_pair(shape,n_clusters,data,rows,cols,noise,minval,maxval,shuffle,random_state)
    integer, intent(in) :: shape(:),n_clusters(:)
    real(dp), allocatable, intent(out) :: data(:,:)
    logical, allocatable, intent(out) :: rows(:,:),cols(:,:)
    real(dp), optional, intent(in) :: noise,minval,maxval
    logical, optional, intent(in) :: shuffle
    integer, optional, intent(in) :: random_state
    call require(size(n_clusters)==2,'checkerboard cluster counts must have two entries')
    call generate(shape,n_clusters(1),n_clusters(2),.false.,data,rows,cols,noise,minval,maxval,shuffle,random_state)
  end subroutine
  subroutine generate(shape,kr,kc,diagonal,data,rows,cols,noise,minval,maxval,shuffle,random_state)
    integer, intent(in) :: shape(:),kr,kc
    logical, intent(in) :: diagonal
    real(dp), allocatable, intent(out) :: data(:,:)
    logical, allocatable, intent(out) :: rows(:,:),cols(:,:)
    real(dp), optional, intent(in) :: noise,minval,maxval
    logical, optional, intent(in) :: shuffle
    integer, optional, intent(in) :: random_state
    type(legacy_random_state) :: rng,shuffle_rng
    real(dp), allocatable :: constants(:)
    integer, allocatable :: row_labels(:),col_labels(:),ri(:),ci(:)
    real(dp) :: std,low,high,width,value
    logical :: shuf
    integer :: n,p,seed,i,j,r,c,blocks,index
    call require(size(shape)==2,'bicluster shape must have two entries')
    call require(all(shape>=0),'bicluster dimensions must be nonnegative')
    call require(kr>=1.and.kc>=1,'bicluster cluster counts must be positive')
    n=shape(1);p=shape(2);std=0;low=10;high=100;seed=0;shuf=.true.
    if(present(noise)) std=noise
    if(present(minval)) low=minval
    if(present(maxval)) high=maxval
    if(present(random_state)) seed=random_state
    if(present(shuffle)) shuf=shuffle
    call require(ieee_is_finite(std),'bicluster noise must be finite')
    call require(std>=0,'bicluster noise must be nonnegative')
    call require(ieee_is_finite(low).and.ieee_is_finite(high),'bicluster bounds must be finite')
    width=ieee_difference(high,low)
    call require(ieee_is_finite(width),'bicluster value range must be finite')
    call require(width>=0,'bicluster maxval must be at least minval')
    call rng%seed(seed)
    if(diagonal) then
      blocks=kr;allocate(constants(kr))
      do r=1,kr
        constants(r)=ieee_difference(low,-ieee_product(width,rng%uniform()))
      end do
    else
      call require(int(kr,int64)*kc<=int(huge(blocks),int64),'checkerboard block count exceeds index range')
      blocks=kr*kc
    end if
    row_labels=partition(n,kr,rng);col_labels=partition(p,kc,rng);allocate(data(n,p));data=0
    if(diagonal) then
      do i=1,n
        do j=1,p
          if(row_labels(i)==col_labels(j)) data(i,j)=constants(row_labels(i)+1)
        end do
      end do
    else
      do r=0,kr-1
        do c=0,kc-1
          value=ieee_difference(low,-ieee_product(width,rng%uniform()))
          do i=1,n
            if(row_labels(i)/=r) cycle
            do j=1,p
              if(col_labels(j)==c) data(i,j)=value
            end do
          end do
        end do
      end do
    end if
    if(std>0) then
      do i=1,n
        do j=1,p
          data(i,j)=ieee_difference(data(i,j),-ieee_product(std,rng%normal()))
        end do
      end do
    end if
    if(shuf) then
      ! Integer-seeded sklearn _shuffle starts its own stream at the original seed.
      call shuffle_rng%seed(seed);ri=permutation(n,shuffle_rng);ci=permutation(p,shuffle_rng)
      data=data(ri,ci);row_labels=row_labels(ri);col_labels=col_labels(ci)
    end if
    allocate(rows(blocks,n),cols(blocks,p))
    if(diagonal) then
      do r=0,kr-1
        rows(r+1,:)=row_labels==r;cols(r+1,:)=col_labels==r
      end do
    else
      index=0
      do r=0,kr-1
        do c=0,kc-1
          index=index+1;rows(index,:)=row_labels==r;cols(index,:)=col_labels==c
        end do
      end do
    end if
  end subroutine
end module
