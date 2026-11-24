! NumPy 2.2.6 legacy binomial inversion/BTPE kernels (BSD).
! Shared by random projections and bicluster generators; see THIRD_PARTY_NOTICES.md.
module sklearn_random_binomial
  use sklearn_common, only: dp
  use sklearn_legacy_rng, only: legacy_random_state
  implicit none
  private
  public :: legacy_binomial
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
  integer function legacy_binomial(n,p,rng) result(y)
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
end module
