! sklearn 1.6.1 linear_model/_huber.py (BSD-3-Clause).
! Native dense/CSR integration is MIT; see THIRD_PARTY_NOTICES.md.
module sklearn_huber_solver
  use sklearn_common,only:dp,require,check_xy
  use sklearn_sparse,only:csr_matrix
  use sklearn_boosting_loss,only:numpy_sum=>boosting_sum
  use lbfgsb_module,only:setulb
  use sklearn_kernel_dispatch,only:kernel_blas_dot,kernel_blas_inner_product,kernel_blas_transpose_column
  use,intrinsic::ieee_arithmetic,only:ieee_is_finite
  use,intrinsic::iso_fortran_env,only:int64
  implicit none
  private
  public::huber_solve,huber_loss_gradient
contains
  subroutine huber_loss_gradient(point,y,weight,epsilon,alpha,loss,g,dense,sparse,dense_fortran)
    real(dp),intent(in)::point(:),y(:),weight(:),epsilon,alpha
    real(dp),intent(out)::loss,g(:)
    real(dp),optional,intent(in)::dense(:,:)
    logical,optional,intent(in)::dense_fortran
    type(csr_matrix),optional,intent(in)::sparse
    real(dp)::r(size(y)),signed_out(size(y)),weighted_in(size(y)),sigma,squared,out_weight,out_loss
    real(dp),allocatable::column(:),prediction(:,:)
    logical::out(size(y)),intercept,fortran_order
    integer::p,q,j,i,k
    q=size(point);p=q-1
    if(present(dense))p=size(dense,2)
    if(present(sparse))p=sparse%n_cols
    intercept=q==p+2;sigma=point(q)
    fortran_order=.false.;if(present(dense_fortran))fortran_order=dense_fortran
    if(present(dense))then
      do i=1,size(y)
        if(fortran_order)then
          r(i)=y(i)-kernel_blas_transpose_column(dense(i,:),point(:p),i,size(y))
        else
          r(i)=y(i)-kernel_blas_dot(dense(i,:),point(:p),size(y),1,i,1)
        end if
      end do
    else
      prediction=sparse%matmul(reshape(point(:p),[p,1]));r=y-prediction(:,1)
    end if
    if(intercept)r=r-point(q-1)
    out=abs(r)>epsilon*sigma
    weighted_in=0;signed_out=0
    where(.not.out)weighted_in=weight*r
    where(out)signed_out=weight*sign(1._dp,r)
    squared=kernel_blas_inner_product(pack(weighted_in,.not.out),pack(r,.not.out))/sigma
    out_weight=numpy_sum(pack(weight,out))
    out_loss=2*epsilon*numpy_sum(pack(weight*abs(r),out))-sigma*out_weight*epsilon**2
    loss=numpy_sum(weight)*sigma+squared+out_loss
    loss=loss+alpha*kernel_blas_inner_product(point(:p),point(:p))
    g=0
    if(present(dense))then
      do j=1,p
        column=pack(dense(:,j),.not.out)
        g(j)=2/sigma*kernel_blas_transpose_column(-column,pack(weighted_in,.not.out),j,p)
        column=pack(dense(:,j),out)
        g(j)=g(j)-2*epsilon*kernel_blas_transpose_column(column,pack(signed_out,out),j,p)
      end do
    else
      ! Slice the two row sets logically, retaining raw duplicate-entry order.
      do i=1,size(y)
        if(out(i))cycle
        do k=sparse%indptr(i),sparse%indptr(i+1)-1
          j=sparse%indices(k);g(j)=g(j)+weighted_in(i)*(-sparse%data(k))
        end do
      end do
      g(:p)=2/sigma*g(:p)
      allocate(column(p));column=0
      do i=1,size(y)
        if(.not.out(i))cycle
        do k=sparse%indptr(i),sparse%indptr(i+1)-1
          j=sparse%indices(k);column(j)=column(j)+signed_out(i)*sparse%data(k)
        end do
      end do
      g(:p)=g(:p)-2*epsilon*column
    end if
    g(:p)=g(:p)+2*alpha*point(:p)
    g(q)=numpy_sum(weight)-out_weight*epsilon**2-squared/sigma
    if(intercept)g(q-1)=-2*numpy_sum(pack(weighted_in,.not.out))/sigma &
      -2*epsilon*numpy_sum(pack(signed_out,out))
  end subroutine

  subroutine huber_solve(y,epsilon,alpha,tol,max_iter,point,weight,n_iter,loss,converged,message,dense,sparse,dense_fortran)
    real(dp),intent(in)::y(:),epsilon,alpha,tol,weight(:)
    integer,intent(in)::max_iter
    real(dp),intent(inout)::point(:)
    integer,intent(out)::n_iter
    real(dp),intent(out)::loss
    logical,intent(out)::converged
    character(*),intent(out)::message
    real(dp),optional,intent(in)::dense(:,:)
    logical,optional,intent(in)::dense_fortran
    type(csr_matrix),optional,intent(in)::sparse
    integer,parameter::memory=10
    real(dp),allocatable::g(:),wa(:),lower(:),upper(:)
    integer,allocatable::iwa(:),nbd(:)
    real(dp)::dsave(29)
    integer::isave(44),q,p,evaluations,iterations
    logical::lsave(4)
    character(60)::task,csave
    call require(present(dense).neqv.present(sparse),'Huber requires one feature representation')
    call require(all(ieee_is_finite([epsilon,alpha,tol])),'Huber controls must be finite')
    call require(epsilon>=1.and.alpha>=0.and.tol>=0.and.max_iter>=0,'invalid HuberRegressor controls')
    call require(size(weight)==size(y),'Huber sample_weight length mismatch')
    call require(all(ieee_is_finite(weight)),'Huber sample_weight must be finite')
    call require(all(ieee_is_finite(y)),'Huber target must be finite')
    if(present(dense))then
      call check_xy(dense,size(y));p=size(dense,2)
    else
      call sparse%validate()
      call require(sparse%n_rows==size(y).and.sparse%n_rows>0.and.sparse%n_cols>0,'invalid Huber CSR shape')
      call require(all(ieee_is_finite(sparse%data)),'Huber CSR data must be finite');p=sparse%n_cols
    end if
    q=size(point)
    call require(q==p+1.or.q==p+2,'Huber warm parameter shape differs from feature count')
    call require(all(ieee_is_finite(point)),'Huber initial parameters must be finite')
    call require(int(q,int64)*(2*memory+5)+11*memory**2+8*memory<=huge(q), &
      'Huber optimizer dimension exceeds integer capacity')
    allocate(g(q),wa(2*memory*q+5*q+11*memory**2+8*memory),lower(q),upper(q),iwa(3*q),nbd(q))
    nbd=0;nbd(q)=1;lower=0;lower(q)=10*spacing(1._dp);upper=0
    wa=0;iwa=0;isave=0;dsave=0;lsave=.false.;csave='';task='START';loss=0;g=0
    iterations=0;evaluations=0
    do
      call setulb(q,memory,point,lower,upper,nbd,loss,g,1.e7_dp,tol,wa,iwa,task,-1,csave,lsave,isave,dsave,ScipyNorm=.true.)
      if(task(:2)=='FG')then
        call huber_loss_gradient(point,y,weight,epsilon,alpha,loss,g,dense,sparse,dense_fortran)
        call require(ieee_is_finite(loss).and.all(ieee_is_finite(g)),'Huber objective is nonfinite')
        evaluations=evaluations+1
      else if(task(:5)=='NEW_X')then
        iterations=iterations+1
        if(iterations>=max_iter)then
          task='STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT'
        else if(evaluations>15000)then
          task='STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT'
        end if
      else
        exit
      end if
    end do
    message=task;converged=task(:4)=='CONV';n_iter=min(iterations,max_iter)
    call require(task(:5)/='ABNOR'.and.task(:5)/='ERROR','HuberRegressor L-BFGS-B convergence failed: '//trim(task))
  end subroutine
end module
