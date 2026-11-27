! sklearn 1.6.1 _glm/glm.py, _linear_loss.py and _loss/_loss.pyx.tp
! define the BSD loss/solver contracts. Native dense/CSR integration is MIT.
module sklearn_glm_lbfgs
  use sklearn_common,only:dp,require
  use sklearn_sparse,only:csr_matrix
  use sklearn_boosting_loss,only:numpy_sum=>boosting_sum
  use sklearn_kernel_dispatch,only:kernel_blas_dot,kernel_blas_transpose_column,kernel_blas_inner_product
  use lbfgsb_module,only:setulb
  use,intrinsic::ieee_arithmetic
  use,intrinsic::ieee_exceptions
  use,intrinsic::iso_fortran_env,only:int64
  implicit none
  private
  public::glm_lbfgs_solve,glm_loss_gradient
contains
  subroutine glm_loss_gradient(point,y,weight,power,link,alpha,loss,g,dense,sparse)
    real(dp),intent(in)::point(:),y(:),weight(:),power,alpha
    character(*),intent(in)::link
    real(dp),intent(out)::loss,g(:)
    real(dp),optional,intent(in)::dense(:,:)
    type(csr_matrix),optional,intent(in)::sparse
    real(dp)::raw(size(y)),value(size(y)),gradient(size(y)),mu,tmp,first,second,mass
    real(dp),allocatable::prediction(:,:)
    integer::p,i,j,k
    logical::halting(size(ieee_all)),raised(size(ieee_all))
    call suspend(halting,raised)
    p=0
    if(present(dense))then
      p=size(dense,2)
      do i=1,size(y)
        raw(i)=kernel_blas_dot(dense(i,:),point(:p),size(y),1,i,1)
      end do
    else
      p=sparse%n_cols;prediction=sparse%matmul(reshape(point(:p),[p,1]));raw=prediction(:,1)
    end if
    if(size(point)>p)raw=raw+point(p+1)
    do i=1,size(y)
      if(link=='log')then
        mu=exp(raw(i))
        if(power==0)then
          gradient(i)=mu*(mu-y(i));value(i)=0.5_dp*(mu-y(i))**2
        else if(power==1)then
          gradient(i)=mu-y(i);value(i)=mu-y(i)*raw(i)
        else if(power==2)then
          tmp=y(i)*exp(-raw(i));gradient(i)=1-tmp;value(i)=raw(i)+tmp
        else
          first=exp((1-power)*raw(i));second=exp((2-power)*raw(i))
          value(i)=second/(2-power)-y(i)*first/(1-power)
          gradient(i)=second-y(i)*first
        end if
      else
        mu=raw(i)
        if(power==0)then
          gradient(i)=mu-y(i);value(i)=0.5_dp*gradient(i)**2
        else if(power==1)then
          value(i)=mu
          if(y(i)/=0)value(i)=y(i)*log(y(i)/mu)+mu-y(i)
          gradient(i)=1-y(i)/mu
        else if(power==2)then
          value(i)=log(mu/y(i))+y(i)/mu-1;gradient(i)=(mu-y(i))/(mu*mu)
        else
          tmp=mu**(1-power);value(i)=mu*tmp/(2-power)-y(i)*tmp/(1-power)
          if(y(i)>0)value(i)=value(i)+y(i)**(2-power)/((1-power)*(2-power))
          gradient(i)=tmp*(1-y(i)/mu)
        end if
      end if
    end do
    mass=numpy_sum(weight);loss=numpy_sum(value*weight)/mass
    loss=loss+0.5_dp*alpha*kernel_blas_inner_product(point(:p),point(:p))
    gradient=gradient*weight/mass;g=0
    if(present(dense))then
      do j=1,p
        g(j)=kernel_blas_transpose_column(dense(:,j),gradient,j,p)
      end do
    else
      do i=1,size(y)
        do k=sparse%indptr(i),sparse%indptr(i+1)-1
          j=sparse%indices(k);g(j)=g(j)+sparse%data(k)*gradient(i)
        end do
      end do
    end if
    g(:p)=g(:p)+alpha*point(:p)
    if(size(point)>p)g(p+1)=numpy_sum(gradient)
    call restore(halting,raised)
  end subroutine
  subroutine glm_lbfgs_solve(point,y,weight,power,link,alpha,tol,max_iter,n_iter,converged,message,dense,sparse)
    real(dp),intent(inout)::point(:)
    real(dp),intent(in)::y(:),weight(:),power,alpha,tol
    character(*),intent(in)::link
    integer,intent(in)::max_iter
    integer,intent(out)::n_iter
    logical,intent(out)::converged
    character(*),intent(out)::message
    real(dp),optional,intent(in)::dense(:,:)
    type(csr_matrix),optional,intent(in)::sparse
    integer,parameter::memory=10
    real(dp),allocatable::g(:),wa(:),lower(:),upper(:)
    integer,allocatable::iwa(:),nbd(:)
    real(dp)::dsave(29),loss
    integer::isave(44),q,evaluations
    logical::lsave(4),halting(size(ieee_all)),raised(size(ieee_all))
    character(60)::task,csave
    q=size(point)
    call require(int(q,int64)*(2*memory+5)+11*memory**2+8*memory<=huge(q),'GLM optimizer exceeds integer capacity')
    allocate(g(q),wa(2*memory*q+5*q+11*memory**2+8*memory),lower(q),upper(q),iwa(3*q),nbd(q))
    nbd=0;lower=0;upper=0;wa=0;iwa=0;isave=0;dsave=0;lsave=.false.;csave='';task='START';loss=0;g=0
    n_iter=0;evaluations=0
    ! Source permits nonfinite trial losses and reports line-search failure as
    ! a warning/status. Preserve caller IEEE modes/flags around that arithmetic.
    call suspend(halting,raised)
    do
      call setulb(q,memory,point,lower,upper,nbd,loss,g,64._dp,tol,wa,iwa,task,-1,csave,lsave,isave,dsave, &
        ScipyNorm=.true.,MaxLineSearch=50)
      if(task(:2)=='FG')then
        call glm_loss_gradient(point,y,weight,power,link,alpha,loss,g,dense,sparse)
        evaluations=evaluations+1
      else if(task(:5)=='NEW_X')then
        n_iter=n_iter+1
        if(n_iter>=max_iter)then
          task='STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT'
        else if(evaluations>15000)then
          task='STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT'
        end if
      else
        exit
      end if
    end do
    call restore(halting,raised)
    converged=task(:4)=='CONV';message=task
  end subroutine
  subroutine suspend(halting,raised)
    logical,intent(out)::halting(:),raised(:)
    integer::i
    do i=1,size(ieee_all)
      call ieee_get_flag(ieee_all(i),raised(i))
    end do
    do i=1,size(ieee_all)
      call ieee_get_halting_mode(ieee_all(i),halting(i))
    end do
    do i=1,size(ieee_all)
      call ieee_set_halting_mode(ieee_all(i),.false.)
    end do
  end subroutine
  subroutine restore(halting,raised)
    logical,intent(in)::halting(:),raised(:)
    integer::i
    do i=1,size(ieee_all)
      call ieee_set_halting_mode(ieee_all(i),halting(i))
    end do
    do i=1,size(ieee_all)
      call ieee_set_flag(ieee_all(i),raised(i))
    end do
  end subroutine
end module
