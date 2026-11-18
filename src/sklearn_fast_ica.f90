! FastICA fixed-point solvers adapted from sklearn/decomposition/_fastica.py.
! See THIRD_PARTY_NOTICES.md for the retained BSD-3-Clause attribution.
module sklearn_fast_ica
  use sklearn_common
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: fast_ica
  type :: fast_ica
    integer :: n_components=0,max_iter=200,random_state=0,n_iter_=0
    character(24) :: algorithm='parallel',whiten='unit-variance',fun='logcosh'
    real(dp) :: tol=1.e-4_dp,alpha=1._dp
    logical :: converged_=.false.
    real(dp), allocatable :: w_init(:,:),components_(:,:),mixing_(:,:),whitening_(:,:),mean_(:)
  contains
    procedure :: fit=>ica_fit
    procedure :: fit_transform=>ica_fit_transform
    procedure :: transform=>ica_transform
    procedure :: inverse_transform=>ica_inverse
  end type
contains
  function decorrelate(w) result(out)
    real(dp), intent(in) :: w(:,:)
    real(dp) :: out(size(w,1),size(w,2)),d(size(w,1)),v(size(w,1),size(w,1))
    call symmetric_eigen(matmul(w,transpose(w)),d,v)
    call require(minval(d)>epsilon(1._dp)*maxval(d), 'FastICA singular fixed-point matrix')
    out=matmul(v*spread(1/sqrt(d),1,size(d)),matmul(transpose(v),w))
  end function
  subroutine contrast(self,z,g,gprime)
    class(fast_ica), intent(in) :: self
    real(dp), intent(in) :: z(:,:)
    real(dp), intent(out) :: g(:,:),gprime(:)
    select case(trim(self%fun))
    case('logcosh')
      g=tanh(self%alpha*z); gprime=sum(self%alpha*(1-g*g),dim=2)/size(z,2)
    case('exp')
      g=exp(-z*z/2); gprime=sum((1-z*z)*g,dim=2)/size(z,2); g=z*g
    case('cube')
      g=z**3; gprime=sum(3*z*z,dim=2)/size(z,2)
    end select
  end subroutine
  subroutine ica_fit(self,x)
    class(fast_ica), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: xc(:,:),s(:),u(:,:),v(:,:),white(:,:),w(:,:),next(:,:),g(:,:),gp(:)
    real(dp), allocatable :: z(:,:),sources(:,:),a(:,:),row(:,:),newrow(:,:)
    real(dp) :: lim,norm,scale
    integer :: n,p,k,i,j,it,l
    logical :: whitening,done
    call check_x(x); n=size(x,1); p=size(x,2)
    call require(n>=2,'FastICA requires at least two samples')
    call require(self%n_components>=0,'invalid FastICA n_components')
    call require(self%max_iter>0,'invalid FastICA max_iter')
    call require(ieee_is_finite(self%tol).and.self%tol>=0,'invalid FastICA tolerance')
    call require(ieee_is_finite(self%alpha).and.self%alpha>=1.and.self%alpha<=2,'invalid FastICA alpha')
    call require(self%algorithm=='parallel'.or.self%algorithm=='deflation','invalid FastICA algorithm')
    call require(self%fun=='logcosh'.or.self%fun=='exp'.or.self%fun=='cube','invalid FastICA contrast')
    call require(self%whiten=='unit-variance'.or.self%whiten=='arbitrary-variance'.or. &
      self%whiten=='none','invalid FastICA whitening')
    whitening=self%whiten/='none'; k=self%n_components
    if(k==0) k=min(n,p)
    k=min(k,min(n,p))
    if(.not.whitening) then
      call require(n>=p,'FastICA unwhitened input requires samples >= features')
      k=p
    end if
    self%mean_=spread(0._dp,1,p); xc=x
    if(allocated(self%whitening_)) deallocate(self%whitening_)
    if(whitening) then
      self%mean_=sum(x,dim=1)/n; xc=x-spread(self%mean_,1,n)
      allocate(s(p),u(n,p),v(p,p)); call dense_svd(xc,s,u,v)
      call require(s(k)>epsilon(1._dp)*max(n,p)*s(1),'FastICA whitening requires retained full rank')
      do j=1,k
        if(v(1,j)<0) v(:,j)=-v(:,j)
      end do
      self%whitening_=transpose(v(:,:k))/spread(s(:k),2,p)
      white=matmul(self%whitening_,transpose(xc))*sqrt(real(n,dp))
    else
      white=transpose(x)
    end if
    if(allocated(self%w_init)) then
      call require(all(shape(self%w_init)==[k,k]),'FastICA w_init shape mismatch')
      call require(all(ieee_is_finite(self%w_init)),'FastICA w_init must be finite')
      w=self%w_init
    else
      call seed_rng(self%random_state); w=normal_random(k,k)
    end if
    allocate(next(k,k),g(k,n),gp(k)); self%converged_=.false.; self%n_iter_=0
    if(self%algorithm=='parallel') then
      w=decorrelate(w)
      do it=1,self%max_iter
        z=matmul(w,white); call contrast(self,z,g,gp)
        next=decorrelate(matmul(g,transpose(white))/n-spread(gp,2,k)*w)
        lim=maxval(abs(abs(sum(next*w,dim=2))-1)); w=next
        self%n_iter_=it
        if(lim<self%tol) then
          self%converged_=.true.; exit
        end if
      end do
    else
      next=0; done=.true.; allocate(row(1,k),newrow(1,k))
      do j=1,k
        row(1,:)=w(j,:); norm=sqrt(sum(row**2))
        call require(norm>tiny(1._dp),'FastICA zero initial vector'); row=row/norm
        do it=1,self%max_iter
          z=matmul(row,white); call contrast(self,z,g(:1,:),gp(:1))
          newrow=matmul(g(:1,:),transpose(white))/n-gp(1)*row
          do l=1,j-1
            scale=sum(newrow(1,:)*next(l,:)); newrow(1,:)=newrow(1,:)-scale*next(l,:)
          end do
          norm=sqrt(sum(newrow**2))
          call require(norm>tiny(1._dp),'FastICA degenerate fixed-point vector')
          newrow=newrow/norm; lim=abs(abs(sum(newrow*row))-1); row=newrow
          if(lim<self%tol) exit
        end do
        self%n_iter_=max(self%n_iter_,min(it,self%max_iter)); done=done.and.lim<self%tol
        next(j,:)=row(1,:)
      end do
      w=next; self%converged_=done
    end if
    if(whitening) then
      self%components_=matmul(w,self%whitening_)
      if(self%whiten=='unit-variance') then
        sources=matmul(xc,transpose(self%components_))
        do j=1,k
          scale=sqrt(sum((sources(:,j)-sum(sources(:,j))/n)**2)/n)
          call require(scale>tiny(1._dp),'FastICA zero source variance')
          self%components_(j,:)=self%components_(j,:)/scale
        end do
      end if
    else
      self%components_=w
    end if
    if(allocated(s)) deallocate(s,u,v)
    allocate(s(p),u(k,p),v(p,p)); call dense_svd(self%components_,s,u,v)
    allocate(a(p,k)); a=0
    do i=1,k
      if(s(i)>epsilon(1._dp)*max(p,k)*s(1)) a=a+ &
        spread(v(:,i)/s(i),2,k)*spread(u(:,i),1,p)
    end do
    self%mixing_=a
  end subroutine
  function ica_transform(self,x) result(z)
    class(fast_ica), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call require(allocated(self%components_),'FastICA is not fitted')
    call check_xy(transpose(x),size(self%components_,2))
    z=matmul(x-spread(self%mean_,1,size(x,1)),transpose(self%components_))
  end function
  function ica_inverse(self,z) result(x)
    class(fast_ica), intent(in) :: self
    real(dp), intent(in) :: z(:,:)
    real(dp), allocatable :: x(:,:)
    call require(allocated(self%components_),'FastICA is not fitted')
    call check_xy(transpose(z),size(self%components_,1))
    x=matmul(z,transpose(self%mixing_))+spread(self%mean_,1,size(z,1))
  end function
  function ica_fit_transform(self,x) result(z)
    class(fast_ica), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x); z=self%transform(x)
  end function
end module
