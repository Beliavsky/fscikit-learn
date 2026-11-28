! Translated from sklearn 1.6.1 bundled LIBSVM Solver (BSD notices retained).
! Internal dense float-rounded Q adapter; shared by native kernel solvers.
module sklearn_libsvm_solver
  use sklearn_common,only:dp,require
  implicit none
  private
  public::libsvm_solve,libsvm_solve_nu
contains
  subroutine libsvm_solve(q,qd,p,labels,c,solution,tol,max_iter,shrinking,rho,iterations,converged)
    real(dp),intent(in)::q(:,:),qd(:),p(:),labels(:),c(:),tol
    real(dp),intent(inout)::solution(:)
    real(dp),intent(out)::rho
    integer,intent(in)::max_iter
    logical,intent(in)::shrinking
    integer,intent(out)::iterations
    logical,intent(out)::converged
    real(dp)::alpha(size(labels)),g(size(labels)),y(size(labels))
    real(dp)::gmax,gmax2,best,diff,quad,obj,oldi,oldj,delta,total,ci,cj,di,dj,ub,lb,yg
    integer::n,i,j,k,t,first,last,free_count,active,counter,order(size(labels))
    real(dp)::gbar(size(labels)),boxes(size(labels)),restored(size(labels))
    logical::unshrink,ui,uj
    ! LIBSVM C-SVC second-order selection, active-set shrinking and box updates.
    ! Its float Q cache is distinct from the double diagonal and query kernels.
    n=size(labels);y=labels;alpha=solution;g=p;gbar=0;boxes=c
    order=[(i,i=1,n)];active=n;counter=min(n,1000)+1;unshrink=.false.
    ! Source initialization accumulates float Q columns in original order.
    do i=1,n
      if(alpha(i)<=0)cycle
      do j=1,n
        g(j)=g(j)+alpha(i)*q(j,i)
        if(alpha(i)>=c(i))gbar(j)=gbar(j)+c(i)*q(j,i)
      end do
    end do
    iterations=0;converged=.false.
    do while(iterations<max_iter)
      counter=counter-1
      if(counter==0)then
        counter=min(n,1000)
        if(shrinking)call shrink()
      end if
      block
        logical::optimal
        call select_pair(optimal)
        if(optimal)then
          call reconstruct();active=n
          call select_pair(optimal)
          if(optimal)then
            converged=.true.;exit
          end if
          counter=1
        end if
      end block
      j=last;iterations=iterations+1;ci=boxes(i);cj=boxes(j);oldi=alpha(i);oldj=alpha(j)
      if(y(i)/=y(j))then
        quad=qd(order(i))+qd(order(j))+2*q(order(j),order(i));if(quad<=0)quad=1.e-12_dp
        delta=(-g(i)-g(j))/quad;diff=alpha(i)-alpha(j);alpha(i)=alpha(i)+delta;alpha(j)=alpha(j)+delta
        if(diff>0)then
          if(alpha(j)<0)then
            alpha(j)=0;alpha(i)=diff
          end if
        else
          if(alpha(i)<0)then
            alpha(i)=0;alpha(j)=-diff
          end if
        end if
        if(diff>ci-cj)then
          if(alpha(i)>ci)then
            alpha(i)=ci;alpha(j)=ci-diff
          end if
        else
          if(alpha(j)>cj)then
            alpha(j)=cj;alpha(i)=cj+diff
          end if
        end if
      else
        quad=qd(order(i))+qd(order(j))-2*q(order(j),order(i));if(quad<=0)quad=1.e-12_dp
        delta=(g(i)-g(j))/quad;total=alpha(i)+alpha(j);alpha(i)=alpha(i)-delta;alpha(j)=alpha(j)+delta
        if(total>ci)then
          if(alpha(i)>ci)then
            alpha(i)=ci;alpha(j)=total-ci
          end if
        else
          if(alpha(j)<0)then
            alpha(j)=0;alpha(i)=total
          end if
        end if
        if(total>cj)then
          if(alpha(j)>cj)then
            alpha(j)=cj;alpha(i)=total-cj
          end if
        else
          if(alpha(i)<0)then
            alpha(i)=0;alpha(j)=total
          end if
        end if
      end if
      di=alpha(i)-oldi;dj=alpha(j)-oldj
      do k=1,active
        ! Match C++ G[k] += (Q_i[k]*di + Q_j[k]*dj), preserving rounding.
        g(k)=g(k)+(q(order(k),order(i))*di+q(order(k),order(j))*dj)
      end do
      ui=oldi>=ci;uj=oldj>=cj
      if(ui.neqv.(alpha(i)>=ci))then
        do k=1,n
          if(ui)then
            gbar(k)=gbar(k)-ci*q(order(k),order(i))
          else
            gbar(k)=gbar(k)+ci*q(order(k),order(i))
          end if
        end do
      end if
      if(uj.neqv.(alpha(j)>=cj))then
        do k=1,n
          if(uj)then
            gbar(k)=gbar(k)-cj*q(order(k),order(j))
          else
            gbar(k)=gbar(k)+cj*q(order(k),order(j))
          end if
        end do
      end if
    end do
    ub=huge(1._dp);lb=-huge(1._dp);total=0;free_count=0
    do i=1,active
      yg=y(i)*g(i)
      if(alpha(i)>=boxes(i))then
        if(y(i)<0)then
          ub=min(ub,yg)
        else
          lb=max(lb,yg)
        end if
      else if(alpha(i)<=0)then
        if(y(i)>0)then
          ub=min(ub,yg)
        else
          lb=max(lb,yg)
        end if
      else
        free_count=free_count+1;total=total+yg
      end if
    end do
    if(free_count>0)then
      rho=total/free_count
    else
      call require(ub/=huge(1._dp).and.lb/=-huge(1._dp),'LIBSVM unbounded threshold')
      rho=(ub+lb)/2
    end if
    restored(order)=alpha;solution=restored
  contains
    subroutine select_pair(optimal)
      logical,intent(out)::optimal
      gmax=-huge(1._dp);gmax2=-huge(1._dp);first=0;last=0;best=huge(1._dp)
      do t=1,active
        if(y(t)>0)then
          if(alpha(t)<boxes(t).and.-g(t)>=gmax)then
            gmax=-g(t);first=t
          end if
        else
          if(alpha(t)>0.and.g(t)>=gmax)then
            gmax=g(t);first=t
          end if
        end if
      end do
      if(first==0)then
        optimal=.true.;return
      end if
      i=first
      do j=1,active
        if(y(j)>0)then
          if(alpha(j)<=0)cycle
          diff=gmax+g(j);gmax2=max(gmax2,g(j));quad=qd(order(i))+qd(order(j))-2*y(i)*q(order(j),order(i))
        else
          if(alpha(j)>=boxes(j))cycle
          diff=gmax-g(j);gmax2=max(gmax2,-g(j));quad=qd(order(i))+qd(order(j))+2*y(i)*q(order(j),order(i))
        end if
        if(diff>0)then
          if(quad<=0)quad=1.e-12_dp
          obj=-(diff*diff)/quad
          if(obj<=best)then
            best=obj;last=j
          end if
        end if
      end do
      optimal=gmax+gmax2<tol.or.last==0
    end subroutine
    subroutine reconstruct()
      integer::a,b,nfree
      if(active==n)return
      g(active+1:)=gbar(active+1:)+p(order(active+1:))
      nfree=count(alpha(:active)>0.and.alpha(:active)<boxes(:active))
      if(real(nfree,dp)*n>2._dp*active*(n-active))then
        do a=active+1,n
          do b=1,active
            if(alpha(b)>0.and.alpha(b)<boxes(b)) &
              g(a)=g(a)+alpha(b)*q(order(b),order(a))
          end do
        end do
      else
        do a=1,active
          if(alpha(a)<=0.or.alpha(a)>=boxes(a))cycle
          do b=active+1,n
            g(b)=g(b)+alpha(a)*q(order(b),order(a))
          end do
        end do
      end if
    end subroutine
    logical function be_shrunk(a,g1,g2)result(v)
      integer,intent(in)::a
      real(dp),intent(in)::g1,g2
      v=.false.
      if(alpha(a)>=boxes(a))then
        if(y(a)>0)then
          v=-g(a)>g1
        else
          v=-g(a)>g2
        end if
      else if(alpha(a)<=0)then
        if(y(a)>0)then
          v=g(a)>g2
        else
          v=g(a)>g1
        end if
      end if
    end function
    subroutine shrink()
      integer::a
      real(dp)::g1,g2
      g1=-huge(1._dp);g2=-huge(1._dp)
      do a=1,active
        if(y(a)>0)then
          if(alpha(a)<boxes(a))g1=max(g1,-g(a))
          if(alpha(a)>0)g2=max(g2,g(a))
        else
          if(alpha(a)<boxes(a))g2=max(g2,-g(a))
          if(alpha(a)>0)g1=max(g1,g(a))
        end if
      end do
      if(.not.unshrink.and.g1+g2<=tol*10)then
        unshrink=.true.;call reconstruct();active=n
      end if
      a=1
      do while(a<=active)
        if(be_shrunk(a,g1,g2))then
          active=active-1
          do while(active>=a)
            if(.not.be_shrunk(active+1,g1,g2))then
              call swap_index(a,active+1);exit
            end if
            active=active-1
          end do
        end if
        a=a+1
      end do
    end subroutine
    subroutine swap_index(a,b)
      integer,intent(in)::a,b
      integer::it
      real(dp)::v
      it=order(a);order(a)=order(b);order(b)=it
      v=y(a);y(a)=y(b);y(b)=v
      v=g(a);g(a)=g(b);g(b)=v
      v=gbar(a);gbar(a)=gbar(b);gbar(b)=v
      v=alpha(a);alpha(a)=alpha(b);alpha(b)=v
      v=boxes(a);boxes(a)=boxes(b);boxes(b)=v
    end subroutine
  end subroutine
  subroutine libsvm_solve_nu(q,qd,p,labels,c,solution,tol,max_iter,shrinking,rho,r,iterations,converged)
    real(dp),intent(in)::q(:,:),qd(:),p(:),labels(:),c(:),tol
    real(dp),intent(inout)::solution(:)
    integer,intent(in)::max_iter
    logical,intent(in)::shrinking
    real(dp),intent(out)::rho,r
    integer,intent(out)::iterations
    logical,intent(out)::converged
    real(dp)::alpha(size(labels)),g(size(labels)),y(size(labels))
    real(dp)::gmax,gmax2,best,diff,quad,obj,oldi,oldj,delta,total,ci,cj,di,dj,ub,lb
    integer::n,i,j,k,t,last,free_count,active,counter,order(size(labels))
    real(dp)::gbar(size(labels)),boxes(size(labels)),restored(size(labels))
    logical::unshrink,ui,uj
    real(dp)::levels(2),maxima(4),hi,lo
    integer::h,ip,ineg
    real(dp)::gmaxn,gmaxn2
    ! LIBSVM Solver_NU two-group second-order selection, active-set shrinking and box updates.
    ! Its float Q cache is distinct from the double diagonal and query kernels.
    n=size(labels);y=labels;alpha=solution;g=p;gbar=0;boxes=c
    order=[(i,i=1,n)];active=n;counter=min(n,1000)+1;unshrink=.false.
    do i=1,n
      if(alpha(i)<=0)cycle
      do j=1,n
        g(j)=g(j)+alpha(i)*q(j,i)
        if(alpha(i)>=c(i))gbar(j)=gbar(j)+c(i)*q(j,i)
      end do
    end do
    iterations=0;converged=.false.
    do while(iterations<max_iter)
      counter=counter-1
      if(counter==0)then
        counter=min(n,1000)
        if(shrinking)call shrink()
      end if
      block
        logical::optimal
        call select_pair(optimal)
        if(optimal)then
          call reconstruct();active=n
          call select_pair(optimal)
          if(optimal)then
            converged=.true.;exit
          end if
          counter=1
        end if
      end block
      j=last;iterations=iterations+1;ci=boxes(i);cj=boxes(j);oldi=alpha(i);oldj=alpha(j)
      quad=qd(order(i))+qd(order(j))-2*q(order(j),order(i));if(quad<=0)quad=1.e-12_dp
      delta=(g(i)-g(j))/quad;total=alpha(i)+alpha(j);alpha(i)=alpha(i)-delta;alpha(j)=alpha(j)+delta
      if(total>ci)then
        if(alpha(i)>ci)then
          alpha(i)=ci;alpha(j)=total-ci
        end if
      else
        if(alpha(j)<0)then
          alpha(j)=0;alpha(i)=total
        end if
      end if
      if(total>cj)then
        if(alpha(j)>cj)then
          alpha(j)=cj;alpha(i)=total-cj
        end if
      else
        if(alpha(i)<0)then
          alpha(i)=0;alpha(j)=total
        end if
      end if
      di=alpha(i)-oldi;dj=alpha(j)-oldj
      do k=1,active
        ! Match C++ G[k] += (Q_i[k]*di + Q_j[k]*dj), preserving rounding.
        g(k)=g(k)+(q(order(k),order(i))*di+q(order(k),order(j))*dj)
      end do
      ui=oldi>=ci;uj=oldj>=cj
      if(ui.neqv.(alpha(i)>=ci))then
        do k=1,n
          if(ui)then
            gbar(k)=gbar(k)-ci*q(order(k),order(i))
          else
            gbar(k)=gbar(k)+ci*q(order(k),order(i))
          end if
        end do
      end if
      if(uj.neqv.(alpha(j)>=cj))then
        do k=1,n
          if(uj)then
            gbar(k)=gbar(k)-cj*q(order(k),order(j))
          else
            gbar(k)=gbar(k)+cj*q(order(k),order(j))
          end if
        end do
      end if
    end do
    do h=1,2
      ub=huge(1._dp);lb=-huge(1._dp);total=0;free_count=0
      do i=1,active
        if((y(i)>0).neqv.(h==1))cycle
        if(alpha(i)>=boxes(i))then
          lb=max(lb,g(i))
        else if(alpha(i)<=0)then
          ub=min(ub,g(i))
        else
          free_count=free_count+1;total=total+g(i)
        end if
      end do
      if(free_count>0)then
        levels(h)=total/free_count
      else
        call require(ub/=huge(1._dp).and.lb/=-huge(1._dp),'LIBSVM nu solver unbounded threshold')
        levels(h)=(ub+lb)/2
      end if
    end do
    r=(levels(1)+levels(2))/2;rho=(levels(1)-levels(2))/2
    restored(order)=alpha;solution=restored
  contains
    subroutine select_pair(optimal)
      logical,intent(out)::optimal
      gmax=-huge(1._dp);gmax2=-huge(1._dp);gmaxn=-huge(1._dp);gmaxn2=-huge(1._dp)
      ip=0;ineg=0;last=0;best=huge(1._dp)
      do t=1,active
        if(y(t)>0)then
          if(alpha(t)<boxes(t).and.-g(t)>=gmax)then
            gmax=-g(t);ip=t
          end if
        else
          if(alpha(t)>0.and.g(t)>=gmaxn)then
            gmaxn=g(t);ineg=t
          end if
        end if
      end do
      do j=1,active
        if(y(j)>0)then
          if(alpha(j)<=0)cycle
          diff=gmax+g(j);gmax2=max(gmax2,g(j));i=ip
        else
          if(alpha(j)>=boxes(j))cycle
          diff=gmaxn-g(j);gmaxn2=max(gmaxn2,-g(j));i=ineg
        end if
        if(diff>0.and.i>0)then
          quad=qd(order(i))+qd(order(j))-2*q(order(j),order(i))
          if(quad<=0)quad=1.e-12_dp
          obj=-(diff*diff)/quad
          if(obj<=best)then
            best=obj;last=j
          end if
        end if
      end do
      ! Use guarded sums in place of C++ -INF for a fully bound group.
      hi=-huge(1._dp);lo=-huge(1._dp)
      if(ip>0.and.gmax2/=-huge(1._dp))hi=gmax+gmax2
      if(ineg>0.and.gmaxn2/=-huge(1._dp))lo=gmaxn+gmaxn2
      optimal=max(hi,lo)<tol.or.last==0
      if(.not.optimal)then
        i=ineg
        if(y(last)>0)i=ip
      end if
    end subroutine
    subroutine reconstruct()
      integer::a,b,nfree
      if(active==n)return
      g(active+1:)=gbar(active+1:)+p(order(active+1:))
      nfree=count(alpha(:active)>0.and.alpha(:active)<boxes(:active))
      if(real(nfree,dp)*n>2._dp*active*(n-active))then
        do a=active+1,n
          do b=1,active
            if(alpha(b)>0.and.alpha(b)<boxes(b)) &
              g(a)=g(a)+alpha(b)*q(order(b),order(a))
          end do
        end do
      else
        do a=1,active
          if(alpha(a)<=0.or.alpha(a)>=boxes(a))cycle
          do b=active+1,n
            g(b)=g(b)+alpha(a)*q(order(b),order(a))
          end do
        end do
      end if
    end subroutine
    logical function be_shrunk(a)result(v)
      integer,intent(in)::a
      v=.false.
      if(alpha(a)>=boxes(a))then
        if(y(a)>0)then
          v=-g(a)>maxima(1)
        else
          v=-g(a)>maxima(4)
        end if
      else if(alpha(a)<=0)then
        if(y(a)>0)then
          v=g(a)>maxima(2)
        else
          v=g(a)>maxima(3)
        end if
      end if
    end function
    subroutine shrink()
      integer::a
      real(dp)::v1,v2
      maxima=-huge(1._dp)
      do a=1,active
        if(y(a)>0)then
          if(alpha(a)<boxes(a))maxima(1)=max(maxima(1),-g(a))
          if(alpha(a)>0)maxima(2)=max(maxima(2),g(a))
        else
          if(alpha(a)<boxes(a))maxima(4)=max(maxima(4),-g(a))
          if(alpha(a)>0)maxima(3)=max(maxima(3),g(a))
        end if
      end do
      v1=-huge(1._dp);v2=v1
      if(all(maxima(:2)/=-huge(1._dp)))v1=maxima(1)+maxima(2)
      if(all(maxima(3:)/=-huge(1._dp)))v2=maxima(3)+maxima(4)
      if(.not.unshrink.and.max(v1,v2)<=tol*10)then
        unshrink=.true.;call reconstruct();active=n
      end if
      a=1
      do while(a<=active)
        if(be_shrunk(a))then
          active=active-1
          do while(active>=a)
            if(.not.be_shrunk(active+1))then
              call swap_index(a,active+1);exit
            end if
            active=active-1
          end do
        end if
        a=a+1
      end do
    end subroutine
    subroutine swap_index(a,b)
      integer,intent(in)::a,b
      integer::it
      real(dp)::v
      it=order(a);order(a)=order(b);order(b)=it
      v=y(a);y(a)=y(b);y(b)=v
      v=g(a);g(a)=g(b);g(b)=v
      v=gbar(a);gbar(a)=gbar(b);gbar(b)=v
      v=alpha(a);alpha(a)=alpha(b);alpha(b)=v
      v=boxes(a);boxes(a)=boxes(b);boxes(b)=v
    end subroutine
  end subroutine
end module
