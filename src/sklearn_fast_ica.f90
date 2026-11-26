! FastICA fixed-point solvers adapted from sklearn/decomposition/_fastica.py.
! See THIRD_PARTY_NOTICES.md for the retained BSD-3-Clause attribution.
module sklearn_fast_ica
  use sklearn_common
  use sklearn_legacy_rng,only:legacy_random_state
  use sklearn_kernel_dispatch,only:kernel_blas_dot
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use,intrinsic::ieee_exceptions
  implicit none
  private
  public :: fast_ica,fastica,fastica_result,fastica_contrast
  abstract interface
    subroutine fastica_contrast(x,g,gprime)
      import dp
      real(dp),intent(in)::x(:,:)
      real(dp),intent(out)::g(:,:),gprime(:)
    end subroutine
  end interface
  type::fastica_result
    real(dp),allocatable::whitening(:,:),unmixing(:,:),sources(:,:),mean(:)
    integer::n_iter=0
    logical::converged=.false.
  end type
  type :: fast_ica
    integer :: n_components=0,max_iter=200,random_state=0,n_iter_=0
    character(24) :: algorithm='parallel',whiten='unit-variance',fun='logcosh'
    character(8)::whiten_solver='svd'
    real(dp) :: tol=1.e-4_dp,alpha=1._dp
    logical :: converged_=.false.
    real(dp), allocatable :: w_init(:,:),components_(:,:),mixing_(:,:),whitening_(:,:),mean_(:)
    real(dp),allocatable::unmixing_(:,:)
    procedure(fastica_contrast),pointer,nopass::contrast_proc=>null()
  contains
    procedure :: fit=>ica_fit
    procedure :: fit_transform=>ica_fit_transform
    procedure :: transform=>ica_transform
    procedure :: inverse_transform=>ica_inverse
  end type
contains
  recursive real(dp)function numpy_sum(a)result(v)
    ! NumPy 2.2.6 pairwise reduction, retained NumPy BSD notice.
    real(dp),intent(in)::a(:)
    real(dp)::lane(8)
    integer::n,i,j,whole,split
    n=size(a);v=-0._dp
    if(n<8)then
      do i=1,n
        v=v+a(i)
      end do
    else if(n<=128)then
      lane=a(:8);whole=n-mod(n,8)
      do i=9,whole,8
        do j=1,8
          lane(j)=lane(j)+a(i+j-1)
        end do
      end do
      v=((lane(1)+lane(2))+(lane(3)+lane(4)))+((lane(5)+lane(6))+(lane(7)+lane(8)))
      do i=whole+1,n
        v=v+a(i)
      end do
    else
      split=n/2;split=split-mod(split,8);v=numpy_sum(a(:split))+numpy_sum(a(split+1:))
    end if
  end function
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
  real(dp)function row_inner(a,b)result(value)
    ! NumPy einsum two-lane real64 sum-of-products kernel (retained NumPy notice).
    real(dp),intent(in)::a(:),b(:)
    real(dp)::left,right
    integer::i
    left=0;right=0
    do i=1,size(a)-1,2
      left=left+a(i)*b(i);right=right+a(i+1)*b(i+1)
    end do
    if(mod(size(a),2)==1)left=left+a(size(a))*b(size(a))
    value=left+right
  end function
  function fastica(x,n_components,algorithm,whiten,fun,alpha,max_iter,tol,w_init,whiten_solver, &
      random_state,compute_sources,contrast_proc)result(fit)
    real(dp),intent(in)::x(:,:)
    integer,optional,intent(in)::n_components,max_iter,random_state
    real(dp),optional,intent(in)::alpha,tol,w_init(:,:)
    character(*),optional,intent(in)::algorithm,whiten,fun,whiten_solver
    logical,optional,intent(in)::compute_sources
    procedure(fastica_contrast),optional::contrast_proc
    type(fastica_result)::fit
    type(fast_ica)::model
    logical::compute
    logical::modes(size(ieee_all)),flags(size(ieee_all))
    call ieee_get_halting_mode(ieee_all,modes);call ieee_get_flag(ieee_all,flags)
    call ieee_set_halting_mode(ieee_all,.false.)
    if(present(n_components))model%n_components=n_components
    if(present(max_iter))model%max_iter=max_iter
    if(present(random_state))model%random_state=random_state
    if(present(alpha))model%alpha=alpha
    if(present(tol))model%tol=tol
    if(present(w_init))model%w_init=w_init
    if(present(algorithm))model%algorithm=algorithm
    if(present(whiten))model%whiten=whiten
    if(present(fun))model%fun=fun
    if(present(whiten_solver))model%whiten_solver=whiten_solver
    if(present(contrast_proc))model%contrast_proc=>contrast_proc
    compute=.true.;if(present(compute_sources))compute=compute_sources
    call model%fit(x)
    fit%unmixing=model%unmixing_;fit%n_iter=model%n_iter_;fit%converged=model%converged_
    if(allocated(model%whitening_))then
      fit%whitening=model%whitening_;fit%mean=model%mean_
    end if
    ! sklearn 1.6.1 retains unit-variance sources even when compute_sources=False.
    if(compute.or.model%whiten=='unit-variance')fit%sources=model%transform(x)
    call ieee_set_flag(ieee_all,flags);call ieee_set_halting_mode(ieee_all,modes)
  end function
  function decorrelate(w) result(out)
    real(dp), intent(in) :: w(:,:)
    real(dp) :: out(size(w,1),size(w,2)),d(size(w,1)),v(size(w,1),size(w,1))
    call require(all(ieee_is_finite(w)),'FastICA fixed-point matrix must be finite')
    out=product(w,transpose(w))
    call require(all(ieee_is_finite(out)),'FastICA decorrelation Gram matrix must be finite')
    call symmetric_eigen(out,d,v)
    ! scipy.linalg.eigh supplies ascending eigenpairs to the decorrelation products.
    d=d(size(d):1:-1);v=v(:,size(d):1:-1)
    d=max(d,tiny(1._dp))
    out=product(v*spread(1/sqrt(d),1,size(d)),product(transpose(v),w))
  end function
  subroutine contrast(self,z,g,gprime)
    class(fast_ica), intent(in) :: self
    real(dp), intent(in) :: z(:,:)
    real(dp), intent(out) :: g(:,:),gprime(:)
    integer::i
    if(associated(self%contrast_proc))then
      call self%contrast_proc(z,g,gprime);return
    end if
    select case(trim(self%fun))
    case('logcosh')
      g=tanh(self%alpha*z)
      do i=1,size(z,1)
        gprime(i)=numpy_sum(self%alpha*(1-g(i,:)*g(i,:)))/size(z,2)
      end do
    case('exp')
      g=exp(-z*z/2)
      do i=1,size(z,1)
        gprime(i)=numpy_sum((1-z(i,:)*z(i,:))*g(i,:))/size(z,2)
      end do
      g=z*g
    case('cube')
      g=z**3
      do i=1,size(z,1)
        gprime(i)=numpy_sum(3*z(i,:)*z(i,:))/size(z,2)
      end do
    end select
  end subroutine
  subroutine ica_fit(self,x)
    class(fast_ica),intent(inout)::self
    real(dp),intent(in)::x(:,:)
    logical::modes(size(ieee_all)),flags(size(ieee_all))
    call ieee_get_halting_mode(ieee_all,modes);call ieee_get_flag(ieee_all,flags)
    call ieee_set_halting_mode(ieee_all,.false.)
    call ica_fit_core(self,x)
    call ieee_set_flag(ieee_all,flags);call ieee_set_halting_mode(ieee_all,modes)
  end subroutine
  subroutine ica_fit_core(self,x)
    class(fast_ica), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: xc(:,:),s(:),u(:,:),v(:,:),white(:,:),w(:,:),next(:,:),g(:,:),gp(:)
    real(dp), allocatable :: z(:,:),sources(:,:),a(:,:),row(:,:),newrow(:,:)
    real(dp) :: lim,norm,scale
    integer :: n,p,k,i,j,it,l
    logical :: whitening,done
    type(legacy_random_state)::rng
    call check_x(x); n=size(x,1); p=size(x,2)
    call require(n>=2,'FastICA requires at least two samples')
    call require(self%random_state>=0,'FastICA random_state must be nonnegative')
    call require(self%whiten_solver=='svd'.or.self%whiten_solver=='eigh','invalid FastICA whiten_solver')
    call require(self%n_components>=0,'invalid FastICA n_components')
    call require(self%max_iter>0,'invalid FastICA max_iter')
    call require(ieee_is_finite(self%tol).and.self%tol>=0,'invalid FastICA tolerance')
    call require(ieee_is_finite(self%alpha).and.self%alpha>=1.and.self%alpha<=2,'invalid FastICA alpha')
    call require(self%algorithm=='parallel'.or.self%algorithm=='deflation','invalid FastICA algorithm')
    call require(associated(self%contrast_proc).or.self%fun=='logcosh'.or.self%fun=='exp'.or. &
      self%fun=='cube','invalid FastICA contrast')
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
      allocate(s(p),u(n,p),v(p,p))
      if(self%whiten_solver=='svd')then
        call dense_svd(xc,s,u,v)
        call require(s(k)>epsilon(1._dp)*max(n,p)*s(1),'FastICA whitening requires retained full rank')
      else
        a=product(transpose(xc),x)
        do j=1,p
          do i=1,j-1
            a(i,j)=a(j,i)
          end do
        end do
        call require(all(ieee_is_finite(a)),'FastICA whitening covariance must be finite')
        call symmetric_eigen(a,s,v)
        s=sqrt(max(s,10*epsilon(1._dp)))
        deallocate(a)
      end if
      do j=1,k
        if(v(1,j)<0) v(:,j)=-v(:,j)
        if(v(1,j)==0)v(:,j)=0
      end do
      self%whitening_=transpose(v(:,:k))/spread(s(:k),2,p)
      white=product(self%whitening_,transpose(xc))*sqrt(real(n,dp))
    else
      white=transpose(x)
    end if
    if(allocated(self%w_init)) then
      call require(all(shape(self%w_init)==[k,k]),'FastICA w_init shape mismatch')
      call require(all(ieee_is_finite(self%w_init)),'FastICA w_init must be finite')
      w=self%w_init
    else
      call rng%seed(self%random_state);allocate(w(k,k))
      do i=1,k
        do j=1,k
          w(i,j)=rng%normal()
        end do
      end do
    end if
    allocate(next(k,k),g(k,n),gp(k)); self%converged_=.false.; self%n_iter_=0
    if(self%algorithm=='parallel') then
      w=decorrelate(w)
      do it=1,self%max_iter
        z=product(w,white); call contrast(self,z,g,gp)
        next=decorrelate(product(g,transpose(white))/n-spread(gp,2,k)*w)
        lim=0
        do i=1,k
          lim=max(lim,abs(abs(row_inner(next(i,:),w(i,:)))-1))
        end do
        w=next
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
          z=product(row,white); call contrast(self,z,g(:1,:),gp(:1))
          do i=1,k
            newrow(1,i)=numpy_sum(white(i,:)*g(1,:))/n-gp(1)*row(1,i)
          end do
          if(j>1)newrow=newrow-product(product(newrow,transpose(next(:j-1,:))),next(:j-1,:))
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
      self%components_=product(w,self%whitening_)
      if(self%whiten=='unit-variance') then
        sources=transpose(product(self%components_,transpose(xc)))
        do j=1,k
          scale=sqrt(numpy_sum((sources(:,j)-numpy_sum(sources(:,j))/n)**2)/n)
          call require(scale>tiny(1._dp),'FastICA zero source variance')
          w(j,:)=w(j,:)/scale
        end do
        self%components_=product(w,self%whitening_)
      end if
    else
      self%components_=w
    end if
    self%unmixing_=w
    call require(all(ieee_is_finite(self%components_)),'FastICA fitted components must be finite')
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
