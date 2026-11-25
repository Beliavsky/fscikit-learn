! Ridge solvers adapted from sklearn 1.6.1 linear_model/_ridge.py.
! LSQR follows SciPy 1.15.3 sparse/linalg/_isolve/lsqr.py.
! BSD attribution is retained in THIRD_PARTY_NOTICES.md and LICENSES/.
module sklearn_ridge_solver
  use sklearn_common,only:dp,require,check_x,check_xy,dense_svd,solve_lower,solve_upper
  use sklearn_sparse,only:csr_matrix
  use lbfgsb_module,only:setulb
  use,intrinsic::ieee_arithmetic,only:ieee_is_finite
  implicit none
  private
  public::ridge_regression,ridge_solver_options,ridge_solution
  type::ridge_solver_options
    character(16)::solver='auto'
    real(dp)::tol=1.e-4_dp
    integer::max_iter=-1
    logical::positive=.false.
  end type
  type::ridge_solution
    real(dp),allocatable::coefficients(:,:)
    integer,allocatable::n_iter(:),native_iterations(:),stop_code(:)
    logical,allocatable::converged(:)
    character(16)::solver=''
  end type
  type,private::ridge_operator
    real(dp),allocatable::dense(:,:)
    type(csr_matrix)::sparse
    integer::n=0,p=0
    logical::is_sparse=.false.
  end type
  interface ridge_regression
    module procedure dense_vs,dense_vv,dense_ms,dense_mv,sparse_vs,sparse_vv,sparse_ms,sparse_mv
  end interface
contains
  function dense_vs(x,y,alpha,options,sample_weight)result(fit)
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:),alpha
    type(ridge_solver_options),optional,intent(in)::options
    real(dp),optional,intent(in)::sample_weight(:)
    type(ridge_solution)::fit
    type(ridge_operator)::op
    op=prepare_dense(x,sample_weight)
    fit=solve_all(op,reshape(y,[size(y),1]),[alpha],options,sample_weight)
  end function
  function dense_vv(x,y,alpha,options,sample_weight)result(fit)
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:),alpha(:)
    type(ridge_solver_options),optional,intent(in)::options
    real(dp),optional,intent(in)::sample_weight(:)
    type(ridge_solution)::fit
    type(ridge_operator)::op
    op=prepare_dense(x,sample_weight)
    fit=solve_all(op,reshape(y,[size(y),1]),alpha,options,sample_weight)
  end function
  function dense_ms(x,y,alpha,options,sample_weight)result(fit)
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:,:),alpha
    type(ridge_solver_options),optional,intent(in)::options
    real(dp),optional,intent(in)::sample_weight(:)
    type(ridge_solution)::fit
    type(ridge_operator)::op
    op=prepare_dense(x,sample_weight)
    fit=solve_all(op,y,[alpha],options,sample_weight)
  end function
  function dense_mv(x,y,alpha,options,sample_weight)result(fit)
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:,:),alpha(:)
    type(ridge_solver_options),optional,intent(in)::options
    real(dp),optional,intent(in)::sample_weight(:)
    type(ridge_solution)::fit
    type(ridge_operator)::op
    op=prepare_dense(x,sample_weight)
    fit=solve_all(op,y,alpha,options,sample_weight)
  end function
  function sparse_vs(x,y,alpha,options,sample_weight)result(fit)
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:),alpha
    type(ridge_solver_options),optional,intent(in)::options
    real(dp),optional,intent(in)::sample_weight(:)
    type(ridge_solution)::fit
    type(ridge_operator)::op
    op=prepare_sparse(x,sample_weight)
    fit=solve_all(op,reshape(y,[size(y),1]),[alpha],options,sample_weight)
  end function
  function sparse_vv(x,y,alpha,options,sample_weight)result(fit)
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:),alpha(:)
    type(ridge_solver_options),optional,intent(in)::options
    real(dp),optional,intent(in)::sample_weight(:)
    type(ridge_solution)::fit
    type(ridge_operator)::op
    op=prepare_sparse(x,sample_weight)
    fit=solve_all(op,reshape(y,[size(y),1]),alpha,options,sample_weight)
  end function
  function sparse_ms(x,y,alpha,options,sample_weight)result(fit)
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:,:),alpha
    type(ridge_solver_options),optional,intent(in)::options
    real(dp),optional,intent(in)::sample_weight(:)
    type(ridge_solution)::fit
    type(ridge_operator)::op
    op=prepare_sparse(x,sample_weight)
    fit=solve_all(op,y,[alpha],options,sample_weight)
  end function
  function sparse_mv(x,y,alpha,options,sample_weight)result(fit)
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:,:),alpha(:)
    type(ridge_solver_options),optional,intent(in)::options
    real(dp),optional,intent(in)::sample_weight(:)
    type(ridge_solution)::fit
    type(ridge_operator)::op
    op=prepare_sparse(x,sample_weight)
    fit=solve_all(op,y,alpha,options,sample_weight)
  end function
  function prepare_dense(x,weights)result(op)
    real(dp),intent(in)::x(:,:)
    real(dp),optional,intent(in)::weights(:)
    type(ridge_operator)::op
    call check_x(x);op%n=size(x,1);op%p=size(x,2);op%dense=x
    if(present(weights))then
      call weight_check(weights,op%n)
      op%dense=op%dense*spread(sqrt(weights),2,op%p)
    end if
  end function
  function prepare_sparse(x,weights)result(op)
    type(csr_matrix),intent(in)::x
    real(dp),optional,intent(in)::weights(:)
    type(ridge_operator)::op
    integer::i
    call x%validate()
    call require(x%n_rows>0.and.x%n_cols>0,'ridge sparse design must be nonempty')
    call require(all(ieee_is_finite(x%data)),'ridge sparse data must be finite')
    op%n=x%n_rows;op%p=x%n_cols;op%is_sparse=.true.;op%sparse=x
    if(present(weights))then
      call weight_check(weights,op%n)
      do i=1,op%n
        op%sparse%data(x%indptr(i):x%indptr(i+1)-1)= &
          op%sparse%data(x%indptr(i):x%indptr(i+1)-1)*sqrt(weights(i))
      end do
    end if
  end function
  subroutine weight_check(weights,n)
    real(dp),intent(in)::weights(:)
    integer,intent(in)::n
    call require(size(weights)==n,'ridge sample weight length mismatch')
    call require(all(ieee_is_finite(weights)),'ridge weights must be finite')
    call require(all(weights>=0),'ridge weights must be nonnegative')
  end subroutine
  function multiply(op,v,transpose)result(out)
    type(ridge_operator),intent(in)::op
    real(dp),intent(in)::v(:)
    logical,intent(in)::transpose
    real(dp),allocatable::out(:)
    integer::i,k,j
    if(.not.op%is_sparse)then
      if(transpose)then
        out=matmul(transpose_matrix(op%dense),v)
      else
        out=matmul(op%dense,v)
      end if
    else
      if(transpose)then
        allocate(out(op%p))
      else
        allocate(out(op%n))
      end if
      out=0
      do i=1,op%n
        do k=op%sparse%indptr(i),op%sparse%indptr(i+1)-1
          j=op%sparse%indices(k)
          if(transpose)then
            out(j)=out(j)+op%sparse%data(k)*v(i)
          else
            out(i)=out(i)+op%sparse%data(k)*v(j)
          end if
        end do
      end do
    end if
  end function
  function transpose_matrix(a)result(b)
    real(dp),intent(in)::a(:,:)
    real(dp)::b(size(a,2),size(a,1))
    b=transpose(a)
  end function
  function solve_all(op,y,alpha,options,weights)result(fit)
    type(ridge_operator),intent(in)::op
    real(dp),intent(in)::y(:,:),alpha(:)
    type(ridge_solver_options),optional,intent(in)::options
    real(dp),optional,intent(in)::weights(:)
    type(ridge_solution)::fit
    type(ridge_solver_options)::control
    real(dp),allocatable::targets(:,:),penalty(:),a(:,:),u(:,:),s(:),v(:,:),projected(:,:),coef(:),gram(:,:),rhs(:)
    integer::t,k,i,dim,limit
    logical::dual,okay,one_alpha
    if(present(options))control=options
    call require(size(y,1)==op%n,'ridge target sample count mismatch');call check_x(y)
    t=size(y,2)
    call require(size(alpha)==1.or.size(alpha)==t,'ridge penalty count must match targets')
    call require(all(ieee_is_finite(alpha)),'ridge penalties must be finite')
    call require(all(alpha>=0),'ridge penalties must be nonnegative')
    call require(ieee_is_finite(control%tol),'ridge tolerance must be finite')
    call require(control%tol>=0.and.control%max_iter>=-1,'invalid ridge iteration controls')
    fit%solver=control%solver
    if(fit%solver=='auto')then
      fit%solver='cholesky'
      if(op%is_sparse)fit%solver='sparse_cg'
      if(control%positive)fit%solver='lbfgs'
    end if
    call require(any(fit%solver==[character(16)::'svd','cholesky','sparse_cg','lsqr','lbfgs']), &
      'unsupported ridge solver; supported: auto svd cholesky sparse_cg lsqr lbfgs')
    call require(.not.control%positive.or.fit%solver=='lbfgs','positive ridge requires lbfgs')
    call require(fit%solver/='lbfgs'.or.control%positive,'ridge lbfgs requires positive coefficients')
    call require(.not.(op%is_sparse.and.fit%solver=='svd'),'ridge SVD requires dense inputs')
    targets=y
    if(present(weights))targets=targets*spread(sqrt(weights),2,t)
    allocate(penalty(t));penalty=alpha(1)
    if(size(alpha)==t)penalty=alpha
    one_alpha=all(penalty==penalty(1));dual=op%p>op%n
    allocate(fit%coefficients(t,op%p),fit%n_iter(t),fit%native_iterations(t),fit%stop_code(t),fit%converged(t))
    fit%coefficients=0;fit%n_iter=-1;fit%native_iterations=0;fit%stop_code=0;fit%converged=.true.
    if(fit%solver=='cholesky')then
      dim=op%p;if(dual)dim=op%n
      allocate(gram(dim,dim),rhs(dim));rhs=0
      do k=1,dim
        rhs=0;rhs(k)=1
        if(dual)then
          coef=multiply(op,rhs,.true.);gram(:,k)=multiply(op,coef,.false.)
        else
          coef=multiply(op,rhs,.false.);gram(:,k)=multiply(op,coef,.true.)
        end if
      end do
    end if
    do k=1,t
      select case(trim(fit%solver))
      case('svd')
        call svd_ridge(op%dense,targets(:,k),penalty(k),coef)
      case('cholesky')
        a=gram
        do i=1,size(a,1)
          a(i,i)=a(i,i)+penalty(k)
        end do
        if(dual)then
          rhs=targets(:,k)
        else
          rhs=multiply(op,targets(:,k),.true.)
        end if
        call positive_solve(a,rhs,coef,okay)
        if(.not.okay)then
          if(dual.and.one_alpha)then
            call least_squares(a,rhs,coef)
          else if(.not.dual.and..not.op%is_sparse)then
            call svd_ridge(op%dense,targets(:,k),penalty(k),coef)
            fit%stop_code(k)=1
          else
            call require(.false.,'ridge singular Cholesky system')
          end if
        end if
        if(dual)coef=multiply(op,coef,.true.)
      case('sparse_cg')
        dim=op%p;if(dual)dim=op%n
        limit=control%max_iter
        if(limit<0.or.dual)limit=10*dim ! sklearn's dual route ignores supplied max_iter.
        if(dual)then
          rhs=targets(:,k)
        else
          rhs=multiply(op,targets(:,k),.true.)
        end if
        call cg_solve(op,rhs,penalty(k),dual,limit,control%tol,coef,fit%native_iterations(k),fit%converged(k))
        if(dual)coef=multiply(op,coef,.true.)
      case('lsqr')
        limit=control%max_iter;if(limit<0)limit=2*op%p
        call lsqr_solve(op,targets(:,k),sqrt(penalty(k)),control%tol,limit,coef,fit%n_iter(k),fit%stop_code(k))
        fit%native_iterations(k)=fit%n_iter(k);fit%converged(k)=any(fit%stop_code(k)==[1,2,4,5])
        if(all(targets(:,k)==0))fit%converged(k)=.true.
      case('lbfgs')
        limit=control%max_iter;if(limit<0)limit=15000
        call lbfgs_solve(op,targets(:,k),penalty(k),control%tol,limit,coef, &
          fit%native_iterations(k),fit%converged(k))
      end select
      fit%coefficients(k,:)=coef
    end do
  end function
  subroutine svd_ridge(a,y,alpha,coef)
    real(dp),intent(in)::a(:,:),y(:),alpha
    real(dp),allocatable,intent(out)::coef(:)
    real(dp),allocatable::u(:,:),v(:,:),s(:)
    integer::n,p,j
    n=size(a,1);p=size(a,2);allocate(u(n,p),v(p,p),s(p),coef(p))
    call dense_svd(a,s,u,v);coef=0
    do j=1,p
      if(s(j)>1.e-15_dp)coef=coef+v(:,j)*(s(j)/(s(j)**2+alpha))*dot_product(u(:,j),y)
    end do
  end subroutine
  subroutine least_squares(a,y,coef)
    real(dp),intent(in)::a(:,:),y(:)
    real(dp),allocatable,intent(out)::coef(:)
    real(dp),allocatable::u(:,:),v(:,:),s(:)
    integer::p,j
    p=size(a,2);allocate(u(size(a,1),p),v(p,p),s(p),coef(p));call dense_svd(a,s,u,v);coef=0
    do j=1,p
      if(s(j)>epsilon(1._dp)*maxval(s))coef=coef+v(:,j)*dot_product(u(:,j),y)/s(j)
    end do
  end subroutine
  subroutine positive_solve(a,b,x,okay)
    real(dp),intent(in)::a(:,:),b(:)
    real(dp),allocatable,intent(out)::x(:)
    logical,intent(out)::okay
    real(dp)::l(size(a,1),size(a,2)),diagonal
    integer::i,j
    l=0;okay=.false.
    do i=1,size(b)
      do j=1,i-1
        l(i,j)=(a(i,j)-dot_product(l(i,:j-1),l(j,:j-1)))/l(j,j)
      end do
      diagonal=a(i,i)-sum(l(i,:i-1)**2)
      if(diagonal<=0)return
      l(i,i)=sqrt(diagonal)
    end do
    x=solve_upper(transpose(l),solve_lower(l,b));okay=.true.
  end subroutine
  subroutine cg_solve(op,b,alpha,dual,limit,tol,x,iterations,converged)
    type(ridge_operator),intent(in)::op
    real(dp),intent(in)::b(:),alpha,tol
    logical,intent(in)::dual
    integer,intent(in)::limit
    real(dp),allocatable,intent(out)::x(:)
    integer,intent(out)::iterations
    logical,intent(out)::converged
    real(dp),allocatable::r(:),p(:),ap(:),tmp(:)
    real(dp)::rho,oldrho,step,denominator,threshold
    integer::it
    allocate(x(size(b)));x=0;r=b;threshold=tol*sqrt(dot_product(b,b));oldrho=0;iterations=0;converged=.false.
    if(all(b==0))then
      converged=.true.;return
    end if
    do it=1,limit
      rho=dot_product(r,r)
      if(sqrt(rho)<threshold.or.rho==0)then
        converged=.true.;return
      end if
      if(it==1)then
        p=r
      else
        p=r+(rho/oldrho)*p
      end if
      if(dual)then
        tmp=multiply(op,p,.true.);ap=multiply(op,tmp,.false.)+alpha*p
      else
        tmp=multiply(op,p,.false.);ap=multiply(op,tmp,.true.)+alpha*p
      end if
      denominator=dot_product(p,ap)
      call require(denominator/=0,'ridge conjugate-gradient breakdown')
      step=rho/denominator;x=x+step*p;r=r-step*ap;oldrho=rho;iterations=it
    end do
    converged=sqrt(dot_product(r,r))<threshold.or.all(r==0)
  end subroutine
  subroutine rotation(a,b,c,s,r)
    real(dp),intent(in)::a,b
    real(dp),intent(out)::c,s,r
    real(dp)::tau
    if(b==0)then
      c=sign(1._dp,a);if(a==0)c=0
      s=0;r=abs(a)
    else if(a==0)then
      c=0;s=sign(1._dp,b);r=abs(b)
    else if(abs(b)>abs(a))then
      tau=a/b;s=sign(1._dp,b)/sqrt(1+tau*tau);c=s*tau;r=b/s
    else
      tau=b/a;c=sign(1._dp,a)/sqrt(1+tau*tau);s=c*tau;r=a/c
    end if
  end subroutine
  subroutine lsqr_solve(op,b,damp,tol,limit,x,iterations,stop)
    type(ridge_operator),intent(in)::op
    real(dp),intent(in)::b(:),damp,tol
    integer,intent(in)::limit
    real(dp),allocatable,intent(out)::x(:)
    integer,intent(out)::iterations,stop
    real(dp),allocatable::u(:),v(:),w(:),dk(:)
    real(dp)::bnorm,beta,alfa,anorm,ddnorm,res2,xxnorm,z,cs2,sn2,rhobar,phibar,arnorm
    real(dp)::rhobar1,cs1,sn1,psi,cs,sn,rho,theta,phi,tau,t1,t2,delta,gambar,rhs,zbar,xnorm,gamma
    real(dp)::acond,rnorm,test1,test2,test3,rtol
    allocate(x(op%p));x=0;u=b;bnorm=sqrt(sum(b*b));beta=bnorm;alfa=0
    allocate(v(op%p));v=0;iterations=0;stop=0
    if(beta>0)then
      u=(1._dp/beta)*u;v=multiply(op,u,.true.);alfa=sqrt(sum(v*v))
    end if
    if(alfa>0)v=(1._dp/alfa)*v
    w=v;rhobar=alfa;phibar=beta;arnorm=alfa*beta
    if(arnorm==0)return
    anorm=0;ddnorm=0;res2=0;xxnorm=0;z=0;cs2=-1;sn2=0
    do while(iterations<limit)
      iterations=iterations+1;u=multiply(op,v,.false.)-alfa*u;beta=sqrt(sum(u*u))
      if(beta>0)then
        u=(1._dp/beta)*u;anorm=sqrt(anorm**2+alfa**2+beta**2+damp**2)
        v=multiply(op,u,.true.)-beta*v;alfa=sqrt(sum(v*v));if(alfa>0)v=(1._dp/alfa)*v
      end if
      psi=0;rhobar1=rhobar
      if(damp>0)then
        rhobar1=sqrt(rhobar**2+damp**2);cs1=rhobar/rhobar1;sn1=damp/rhobar1
        psi=sn1*phibar;phibar=cs1*phibar
      end if
      call rotation(rhobar1,beta,cs,sn,rho)
      call require(rho/=0,'ridge LSQR rotation breakdown')
      theta=sn*alfa;rhobar=-cs*alfa;phi=cs*phibar;phibar=sn*phibar;tau=sn*phi
      t1=phi/rho;t2=-theta/rho;dk=(1._dp/rho)*w;x=x+t1*w;w=v+t2*w;ddnorm=ddnorm+sum(dk*dk)
      delta=sn2*rho;gambar=-cs2*rho;rhs=phi-delta*z
      call require(gambar/=0,'ridge LSQR norm estimate breakdown')
      zbar=rhs/gambar;xnorm=sqrt(xxnorm+zbar**2);gamma=sqrt(gambar**2+theta**2)
      cs2=gambar/gamma;sn2=theta/gamma;z=rhs/gamma;xxnorm=xxnorm+z*z
      acond=anorm*sqrt(ddnorm);res2=res2+psi**2;rnorm=sqrt(phibar**2+res2);arnorm=alfa*abs(tau)
      test1=rnorm/bnorm;test2=arnorm/(anorm*rnorm+epsilon(1._dp));test3=1/(acond+epsilon(1._dp))
      t1=test1/(1+anorm*xnorm/bnorm);rtol=tol+tol*anorm*xnorm/bnorm
      if(iterations>=limit)stop=7
      if(1+test3<=1)stop=6
      if(1+test2<=1)stop=5
      if(1+t1<=1)stop=4
      if(test3<=1.e-8_dp)stop=3
      if(test2<=tol)stop=2
      if(test1<=rtol)stop=1
      if(stop/=0)exit
    end do
  end subroutine
  subroutine lbfgs_solve(op,y,alpha,tol,limit,x,iterations,converged)
    type(ridge_operator),intent(in)::op
    real(dp),intent(in)::y(:),alpha,tol
    integer,intent(in)::limit
    real(dp),allocatable,intent(out)::x(:)
    integer,intent(out)::iterations
    logical,intent(out)::converged
    integer,parameter::memory=10
    real(dp),allocatable::g(:),lower(:),upper(:),wa(:),residual(:)
    integer,allocatable::nbd(:),iwa(:)
    real(dp)::f,dsave(29)
    integer::isave(44),p,evaluations
    logical::lsave(4)
    character(60)::task,csave
    p=op%p
    call require(p<=(huge(p)-11*memory*memory-8*memory)/(2*memory+5),'ridge optimizer dimension exceeds capacity')
    allocate(x(p),g(p),lower(p),upper(p),nbd(p),iwa(3*p),wa((2*memory+5)*p+11*memory*memory+8*memory))
    x=0;g=0;f=0;lower=0;upper=0;nbd=1;iwa=0;wa=0;isave=0;dsave=0;lsave=.false.
    task='START';csave='';iterations=0;evaluations=0
    do
      call setulb(p,memory,x,lower,upper,nbd,f,g,tol/epsilon(1._dp),tol, &
        wa,iwa,task,-1,csave,lsave,isave,dsave)
      if(task(:2)=='FG')then
        residual=multiply(op,x,.false.)-y;f=.5_dp*sum(residual**2)+.5_dp*alpha*sum(x*x)
        g=multiply(op,residual,.true.)+alpha*x;evaluations=evaluations+1
      else if(task(:5)=='NEW_X')then
        iterations=iterations+1
        if(iterations>=limit)then
          task='STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT'
        else if(evaluations>15000)then
          task='STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT'
        end if
      else
        exit
      end if
    end do
    converged=task(:4)=='CONV'
  end subroutine
end module
