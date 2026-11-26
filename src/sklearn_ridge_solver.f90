! Ridge solvers adapted from sklearn 1.6.1 linear_model/_ridge.py.
! LSQR follows SciPy 1.15.3 sparse/linalg/_isolve/lsqr.py.
! BSD attribution is retained in THIRD_PARTY_NOTICES.md and LICENSES/.
module sklearn_ridge_solver
  use sklearn_common,only:dp,require,check_x,check_xy,dense_svd,solve_lower,solve_upper
  use sklearn_sparse,only:csr_matrix
  use sklearn_legacy_rng,only:legacy_random_state
  use sklearn_kernel_dispatch,only:kernel_blas_inner_product,kernel_blas_dot
  use,intrinsic::iso_fortran_env,only:int64
  use,intrinsic::iso_c_binding,only:c_double
  use lbfgsb_module,only:setulb
  use,intrinsic::ieee_arithmetic,only:ieee_is_finite
  use,intrinsic::ieee_exceptions
  implicit none
  private
  public::ridge_regression,ridge_solver_options,ridge_solution
  ! Internal matrix-free centered CSR entry point for the Ridge estimator.
  public::ridge_centered_sparse
  type::ridge_solver_options
    character(16)::solver='auto'
    real(dp)::tol=1.e-4_dp
    integer::max_iter=-1,random_state=0
    logical::positive=.false.,return_intercept=.false.
  end type
  type::ridge_solution
    real(dp),allocatable::coefficients(:,:),intercept(:)
    integer,allocatable::n_iter(:),native_iterations(:),stop_code(:)
    logical,allocatable::converged(:)
    character(16)::solver=''
  end type
  type,private::ridge_operator
    real(dp),allocatable::dense(:,:),offset(:),root_weight(:)
    type(csr_matrix)::sparse
    integer::n=0,p=0
    logical::is_sparse=.false.
    logical::unweighted_sum=.false.
  end type
  interface ridge_regression
    module procedure dense_vs,dense_vv,dense_ms,dense_mv,sparse_vs,sparse_vv,sparse_ms,sparse_mv
  end interface
  interface
    function ridge_fma(a,b,c)bind(C,name='fma')result(v)
      import c_double
      real(c_double),value::a,b,c
      real(c_double)::v
    end function
  end interface
contains
  recursive real(dp)function numpy_sum(a)result(v)
    ! NumPy pairwise sum for LinearRegression's unweighted sparse adjoint.
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
  function ridge_centered_sparse(x,y,alpha,offset,options,sample_weight,unweighted_sum)result(fit)
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:,:),alpha(:),offset(:)
    type(ridge_solver_options),intent(in)::options
    real(dp),optional,intent(in)::sample_weight(:)
    logical,optional,intent(in)::unweighted_sum
    type(ridge_solution)::fit
    type(ridge_operator)::op
    call require(.not.stochastic_options(options),'centered CSR operator requires a deterministic ridge solver')
    op=prepare_sparse(x,sample_weight,options)
    if(present(unweighted_sum))op%unweighted_sum=unweighted_sum
    call require(size(offset)==op%p,'ridge centering offset feature count differs')
    op%offset=offset;allocate(op%root_weight(op%n));op%root_weight=1
    if(present(sample_weight))op%root_weight=sqrt(sample_weight)
    fit=solve_all(op,y,alpha,options,sample_weight)
  end function
  function dense_vs(x,y,alpha,options,sample_weight)result(fit)
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:),alpha
    type(ridge_solver_options),optional,intent(in)::options
    real(dp),optional,intent(in)::sample_weight(:)
    type(ridge_solution)::fit
    type(ridge_operator)::op
    op=prepare_dense(x,sample_weight,options)
    fit=solve_all(op,reshape(y,[size(y),1]),[alpha],options,sample_weight)
  end function
  function dense_vv(x,y,alpha,options,sample_weight)result(fit)
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:),alpha(:)
    type(ridge_solver_options),optional,intent(in)::options
    real(dp),optional,intent(in)::sample_weight(:)
    type(ridge_solution)::fit
    type(ridge_operator)::op
    op=prepare_dense(x,sample_weight,options)
    fit=solve_all(op,reshape(y,[size(y),1]),alpha,options,sample_weight)
  end function
  function dense_ms(x,y,alpha,options,sample_weight)result(fit)
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:,:),alpha
    type(ridge_solver_options),optional,intent(in)::options
    real(dp),optional,intent(in)::sample_weight(:)
    type(ridge_solution)::fit
    type(ridge_operator)::op
    op=prepare_dense(x,sample_weight,options)
    fit=solve_all(op,y,[alpha],options,sample_weight)
  end function
  function dense_mv(x,y,alpha,options,sample_weight)result(fit)
    real(dp),intent(in)::x(:,:)
    real(dp),intent(in)::y(:,:),alpha(:)
    type(ridge_solver_options),optional,intent(in)::options
    real(dp),optional,intent(in)::sample_weight(:)
    type(ridge_solution)::fit
    type(ridge_operator)::op
    op=prepare_dense(x,sample_weight,options)
    fit=solve_all(op,y,alpha,options,sample_weight)
  end function
  function sparse_vs(x,y,alpha,options,sample_weight)result(fit)
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:),alpha
    type(ridge_solver_options),optional,intent(in)::options
    real(dp),optional,intent(in)::sample_weight(:)
    type(ridge_solution)::fit
    type(ridge_operator)::op
    op=prepare_sparse(x,sample_weight,options)
    fit=solve_all(op,reshape(y,[size(y),1]),[alpha],options,sample_weight)
  end function
  function sparse_vv(x,y,alpha,options,sample_weight)result(fit)
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:),alpha(:)
    type(ridge_solver_options),optional,intent(in)::options
    real(dp),optional,intent(in)::sample_weight(:)
    type(ridge_solution)::fit
    type(ridge_operator)::op
    op=prepare_sparse(x,sample_weight,options)
    fit=solve_all(op,reshape(y,[size(y),1]),alpha,options,sample_weight)
  end function
  function sparse_ms(x,y,alpha,options,sample_weight)result(fit)
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:,:),alpha
    type(ridge_solver_options),optional,intent(in)::options
    real(dp),optional,intent(in)::sample_weight(:)
    type(ridge_solution)::fit
    type(ridge_operator)::op
    op=prepare_sparse(x,sample_weight,options)
    fit=solve_all(op,y,[alpha],options,sample_weight)
  end function
  function sparse_mv(x,y,alpha,options,sample_weight)result(fit)
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::y(:,:),alpha(:)
    type(ridge_solver_options),optional,intent(in)::options
    real(dp),optional,intent(in)::sample_weight(:)
    type(ridge_solution)::fit
    type(ridge_operator)::op
    op=prepare_sparse(x,sample_weight,options)
    fit=solve_all(op,y,alpha,options,sample_weight)
  end function
  logical function stochastic_options(options)result(stochastic)
    type(ridge_solver_options),optional,intent(in)::options
    type(ridge_solver_options)::control
    if(present(options))control=options
    stochastic=control%solver=='sag'.or.control%solver=='saga'
    if(control%solver=='auto'.and.control%return_intercept.and..not.control%positive)stochastic=.true.
  end function
  function prepare_dense(x,weights,options)result(op)
    real(dp),intent(in)::x(:,:)
    real(dp),optional,intent(in)::weights(:)
    type(ridge_solver_options),optional,intent(in)::options
    type(ridge_operator)::op
    call check_x(x);op%n=size(x,1);op%p=size(x,2);op%dense=x
    if(present(weights))then
      call weight_check(weights,op%n,stochastic_options(options))
      if(.not.stochastic_options(options))op%dense=op%dense*spread(sqrt(weights),2,op%p)
    end if
  end function
  function prepare_sparse(x,weights,options)result(op)
    type(csr_matrix),intent(in)::x
    real(dp),optional,intent(in)::weights(:)
    type(ridge_solver_options),optional,intent(in)::options
    type(ridge_operator)::op
    integer::i
    call x%validate()
    call require(x%n_rows>0.and.x%n_cols>0,'ridge sparse design must be nonempty')
    call require(all(ieee_is_finite(x%data)),'ridge sparse data must be finite')
    op%n=x%n_rows;op%p=x%n_cols;op%is_sparse=.true.;op%sparse=x
    if(present(weights))then
      call weight_check(weights,op%n,stochastic_options(options))
      if(stochastic_options(options))return
      op%sparse=rescale_csr(x,weights)
    end if
  end function
  function rescale_csr(x,weights)result(y)
    ! SciPy 1.15.3 dia(sqrt(weights)) @ CSR: reverse first-visited columns,
    ! merge duplicate products in traversal order, and omit exact zeros.
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::weights(:)
    type(csr_matrix)::y
    real(dp)::sums(x%n_cols),scale
    real(dp),allocatable::values(:)
    integer::seen(x%n_cols),columns(x%n_cols),i,j,k,used,nnz
    integer,allocatable::indices(:)
    y%n_rows=x%n_rows;y%n_cols=x%n_cols;allocate(y%indptr(x%n_rows+1))
    allocate(values(size(x%data)),indices(size(x%data)));seen=0;nnz=0;y%indptr(1)=1
    do i=1,x%n_rows
      used=0;scale=sqrt(weights(i))
      do k=x%indptr(i),x%indptr(i+1)-1
        j=x%indices(k)
        if(seen(j)/=i)then
          seen(j)=i;used=used+1;columns(used)=j;sums(j)=0
        end if
        sums(j)=sums(j)+scale*x%data(k)
      end do
      do k=used,1,-1
        j=columns(k)
        if(sums(j)/=0)then
          nnz=nnz+1;indices(nnz)=j;values(nnz)=sums(j)
        end if
      end do
      y%indptr(i+1)=nnz+1
    end do
    y%indices=indices(:nnz);y%data=values(:nnz)
  end function
  subroutine weight_check(weights,n,signed)
    real(dp),intent(in)::weights(:)
    integer,intent(in)::n
    logical,intent(in)::signed
    call require(size(weights)==n,'ridge sample weight length mismatch')
    call require(all(ieee_is_finite(weights)),'ridge weights must be finite')
    if(.not.signed)call require(all(weights>=0),'ridge weights must be nonnegative')
  end subroutine
  function multiply(op,v,transpose)result(out)
    type(ridge_operator),intent(in)::op
    real(dp),intent(in)::v(:)
    logical,intent(in)::transpose
    real(dp),allocatable::out(:)
    integer::i,k,j
    if(.not.op%is_sparse)then
      if(transpose)then
        allocate(out(op%p))
        do j=1,op%p
          out(j)=gemv_n_column(op%dense(:,j),v,j,op%p)
        end do
      else
        allocate(out(op%n))
        do i=1,op%n
          out(i)=kernel_blas_dot(op%dense(i,:),v,op%n,1,i,1)
        end do
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
    if(allocated(op%offset))then
      if(transpose)then
        if(op%unweighted_sum)then
          out=out-op%offset*numpy_sum(v)
        else
          out=out-op%offset*kernel_blas_inner_product(op%root_weight,v)
        end if
      else
        out=out-op%root_weight*kernel_blas_inner_product(op%offset,v)
      end if
    end if
  end function
  real(dp)function gemv_n_column(a,x,index,outputs)result(total)
    ! OpenBLAS 0.3.29 dgemv_n_4 / SkylakeX microkernel, 2014 OpenBLAS Project (BSD).
    ! X.T is Fortran-contiguous in NumPy, unlike the forward C-contiguous route.
    real(dp),intent(in)::a(:),x(:)
    integer,intent(in)::index,outputs
    real(dp)::part
    integer::i,n,whole
    n=size(a);whole=n-mod(n,4);total=0
    if(outputs==1)then
      total=kernel_blas_inner_product(a,x)
    else if(outputs<=3)then
      do i=1,whole,2
        part=ridge_fma(a(i),x(i),a(i+1)*x(i+1));total=total+part
      end do
      do i=whole+1,n
        total=ridge_fma(a(i),x(i),total)
      end do
    else if(index<=outputs-mod(outputs,4))then
      do i=1,whole,4
        part=ridge_fma(a(i),x(i),a(i+1)*x(i+1))
        part=ridge_fma(a(i+2),x(i+2),part);part=ridge_fma(a(i+3),x(i+3),part)
        total=total+part
      end do
      i=whole+1
      if(n-whole>=2)then
        total=total+ridge_fma(a(i),x(i),a(i+1)*x(i+1));i=i+2
      end if
      if(i<=n)total=total+a(i)*x(i)
    else
      do i=1,n
        total=ridge_fma(a(i),x(i),total)
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
      if(control%return_intercept)fit%solver='sag'
      if(control%positive)fit%solver='lbfgs'
    end if
    call require(any(fit%solver==[character(16)::'svd','cholesky','sparse_cg','lsqr','lbfgs','sag','saga']), &
      'unsupported ridge solver; supported: auto svd cholesky sparse_cg lsqr lbfgs sag saga')
    call require(.not.control%positive.or.fit%solver=='lbfgs','positive ridge requires lbfgs')
    call require(fit%solver/='lbfgs'.or.control%positive,'ridge lbfgs requires positive coefficients')
    call require(.not.(op%is_sparse.and.fit%solver=='svd'),'ridge SVD requires dense inputs')
    call require(.not.control%return_intercept.or.fit%solver=='sag','ridge direct intercept requires sag')
    if(stochastic_options(control))call require(control%random_state>=0,'ridge random state must be nonnegative')
    targets=y
    if(present(weights))then
      if(.not.stochastic_options(control))targets=targets*spread(sqrt(weights),2,t)
    end if
    allocate(penalty(t));penalty=alpha(1)
    if(size(alpha)==t)penalty=alpha
    one_alpha=all(penalty==penalty(1));dual=op%p>op%n
    allocate(fit%coefficients(t,op%p),fit%n_iter(t),fit%native_iterations(t),fit%stop_code(t),fit%converged(t))
    allocate(fit%intercept(t));fit%intercept=0
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
      case('sag','saga')
        limit=control%max_iter;if(limit<0)limit=1000
        call stochastic_ridge(op,targets(:,k),penalty(k),control%tol,limit,control%random_state, &
          fit%solver=='saga',control%return_intercept,coef,fit%intercept(k),fit%n_iter(k),fit%converged(k),weights)
        fit%native_iterations(k)=fit%n_iter(k)
      case('lbfgs')
        limit=control%max_iter;if(limit<0)limit=15000
        call lbfgs_solve(op,targets(:,k),penalty(k),control%tol,limit,coef, &
          fit%native_iterations(k),fit%converged(k))
      end select
      fit%coefficients(k,:)=coef
    end do
  end function
  ! Squared-loss SAG/SAGA: sklearn _sag.py and _sag_fast.pyx.tp.
  ! Raw CSR entries are traversed in source order; lagged updates never densify X.
  subroutine stochastic_ridge(op,y,alpha,tol,limit,seed,saga,fit_intercept,coef,intercept,iterations,converged,weights)
    type(ridge_operator),intent(in)::op
    real(dp),intent(in)::y(:),alpha,tol
    integer,intent(in)::limit,seed
    logical,intent(in)::saga,fit_intercept
    real(dp),allocatable,intent(out)::coef(:)
    real(dp),intent(out)::intercept
    integer,intent(out)::iterations
    logical,intent(out)::converged
    real(dp),optional,intent(in)::weights(:)
    type(legacy_random_state)::rng
    real(dp),allocatable::sum_gradient(:),memory(:),previous(:),cumulative(:),values(:)
    integer,allocatable::hist(:),columns(:)
    logical,allocatable::seen(:)
    integer::n,p,epoch,it,row,j,k,first,last,nnz,num_seen
    integer(int64)::state
    real(dp)::maxnorm,norm,scaled_alpha,lipschitz,step,scale_update,wscale,prediction,gradient,correction
    real(dp)::intercept_gradient,decay,max_change,max_weight,cum
    n=op%n;p=op%p;maxnorm=0
    do row=1,n
      if(op%is_sparse)then
        norm=sum(op%sparse%data(op%sparse%indptr(row):op%sparse%indptr(row+1)-1)**2)
      else
        norm=sum(op%dense(row,:)**2)
      end if
      maxnorm=max(maxnorm,norm)
    end do
    scaled_alpha=alpha/n;lipschitz=maxnorm
    if(fit_intercept)lipschitz=lipschitz+1
    lipschitz=lipschitz+scaled_alpha
    call require(lipschitz>0,'ridge stochastic step has zero Lipschitz constant')
    if(saga)then
      step=1._dp/(2*lipschitz+min(2._dp*n*scaled_alpha,lipschitz))
    else
      step=1._dp/lipschitz
    end if
    call require(step*scaled_alpha/=1,'ridge SAG step times penalty must differ from one')
    scale_update=1-step*scaled_alpha
    allocate(coef(p),sum_gradient(p),memory(n),previous(p),cumulative(n),hist(p),seen(n))
    coef=0;sum_gradient=0;memory=0;previous=0;cumulative=0;hist=0;seen=.false.
    intercept=0;intercept_gradient=0;wscale=1;num_seen=0;converged=.false.;iterations=1
    decay=1;if(op%is_sparse)decay=.01_dp
    call rng%seed(seed);state=int(rng%integer_index(2147483646),int64)
    if(.not.op%is_sparse)then
      allocate(columns(p),values(p));columns=[(j,j=1,p)]
    end if
    do epoch=1,limit
      do it=0,n-1
        if(state==0)state=1
        state=iand(ieor(state,ishft(state,13)),4294967295_int64)
        state=ieor(state,ishft(state,-17))
        state=iand(ieor(state,ishft(state,5)),4294967295_int64)
        row=int(modulo(iand(state,2147483647_int64),int(n,int64)))+1
        if(op%is_sparse)then
          first=op%sparse%indptr(row);last=op%sparse%indptr(row+1)-1
          columns=op%sparse%indices(first:last);values=op%sparse%data(first:last)
        else
          values=op%dense(row,:)
        end if
        nnz=size(columns)
        if(.not.seen(row))then
          seen(row)=.true.;num_seen=num_seen+1
        end if
        if(it>0)then
          do k=1,nnz
            j=columns(k);cum=cumulative(it)
            if(hist(j)/=0)cum=cum-cumulative(hist(j))
            coef(j)=coef(j)-cum*sum_gradient(j);hist(j)=it
          end do
        end if
        prediction=0
        do k=1,nnz
          prediction=prediction+coef(columns(k))*values(k)
        end do
        prediction=wscale*prediction+intercept
        gradient=prediction-y(row)
        if(present(weights))gradient=gradient*weights(row)
        wscale=wscale*scale_update
        do k=1,nnz
          j=columns(k);correction=values(k)*(gradient-memory(row))
          if(saga)coef(j)=coef(j)-(correction*step*(1-1._dp/num_seen)/wscale)
          sum_gradient(j)=sum_gradient(j)+correction
        end do
        if(fit_intercept)then
          correction=gradient-memory(row);intercept_gradient=intercept_gradient+correction
          correction=correction*step*(1-1._dp/num_seen)
          if(saga)then
            intercept=intercept-(step*intercept_gradient/num_seen*decay+correction)
          else
            intercept=intercept-(step*intercept_gradient/num_seen*decay)
          end if
          call require(ieee_is_finite(intercept),'ridge stochastic intercept is nonfinite')
        end if
        memory(row)=gradient
        if(it==0)then
          cumulative(1)=step/(wscale*num_seen)
        else
          cumulative(it+1)=cumulative(it)+step/(wscale*num_seen)
        end if
        if(wscale<1.e-9_dp)call reset_scale(it+1)
      end do
      call reset_scale(n)
      max_weight=maxval(abs(coef));max_change=maxval(abs(coef-previous));previous=coef
      iterations=epoch
      if(max_weight==0)then
        converged=max_change==0
      else
        converged=max_change/max_weight<=tol
      end if
      if(converged)exit
    end do
  contains
    subroutine reset_scale(sample_itr)
      integer,intent(in)::sample_itr
      integer::feature
      real(dp)::pending
      do feature=1,p
        pending=cumulative(sample_itr)
        if(hist(feature)/=0)pending=pending-cumulative(hist(feature))
        coef(feature)=(coef(feature)-pending*sum_gradient(feature))*wscale
        call require(ieee_is_finite(coef(feature)),'ridge stochastic coefficient is nonfinite')
        hist(feature)=modulo(sample_itr,n)
      end do
      cumulative(sample_itr)=0;wscale=1
    end subroutine
  end subroutine
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
    logical::modes(size(ieee_all)),flags(size(ieee_all))
    call ieee_get_halting_mode(ieee_all,modes);call ieee_get_flag(ieee_all,flags)
    call ieee_set_halting_mode(ieee_all,.false.)
    allocate(x(size(b)));x=0;r=b;rho=kernel_blas_inner_product(b,b)
    threshold=tol*sqrt(rho);oldrho=0;iterations=0;converged=.false.
    if(sqrt(rho)==0)then
      x=b;converged=.true.
    else
      do it=1,limit
        rho=kernel_blas_inner_product(r,r)
        if(sqrt(rho)<threshold)then
          converged=.true.;exit
        end if
        if(it==1)then
          p=r
        else
          p=p*(rho/oldrho);p=p+r
        end if
        if(dual)then
          tmp=multiply(op,p,.true.);ap=multiply(op,tmp,.false.)+alpha*p
        else
          tmp=multiply(op,p,.false.);ap=multiply(op,tmp,.true.)+alpha*p
        end if
        denominator=kernel_blas_inner_product(p,ap)
        step=rho/denominator;x=x+step*p;r=r-step*ap;oldrho=rho;iterations=it
      end do
    end if
    call ieee_set_flag(ieee_all,flags);call ieee_set_halting_mode(ieee_all,modes)
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
    allocate(x(op%p));x=0;u=b;bnorm=sqrt(kernel_blas_inner_product(b,b));beta=bnorm;alfa=0
    allocate(v(op%p));v=0;iterations=0;stop=0
    if(beta>0)then
      u=(1._dp/beta)*u;v=multiply(op,u,.true.);alfa=sqrt(kernel_blas_inner_product(v,v))
    end if
    if(alfa>0)v=(1._dp/alfa)*v
    w=v;rhobar=alfa;phibar=beta;arnorm=alfa*beta
    if(arnorm==0)return
    anorm=0;ddnorm=0;res2=0;xxnorm=0;z=0;cs2=-1;sn2=0
    do while(iterations<limit)
      iterations=iterations+1;u=multiply(op,v,.false.)-alfa*u;beta=sqrt(kernel_blas_inner_product(u,u))
      if(beta>0)then
        u=(1._dp/beta)*u;anorm=sqrt(anorm**2+alfa**2+beta**2+damp**2)
        v=multiply(op,u,.true.)-beta*v;alfa=sqrt(kernel_blas_inner_product(v,v));if(alfa>0)v=(1._dp/alfa)*v
      end if
      psi=0;rhobar1=rhobar
      if(damp>0)then
        rhobar1=sqrt(rhobar**2+damp**2);cs1=rhobar/rhobar1;sn1=damp/rhobar1
        psi=sn1*phibar;phibar=cs1*phibar
      end if
      call rotation(rhobar1,beta,cs,sn,rho)
      call require(rho/=0,'ridge LSQR rotation breakdown')
      theta=sn*alfa;rhobar=-cs*alfa;phi=cs*phibar;phibar=sn*phibar;tau=sn*phi
      t1=phi/rho;t2=-theta/rho;dk=(1._dp/rho)*w;x=x+t1*w;w=v+t2*w;ddnorm=ddnorm+kernel_blas_inner_product(dk,dk)
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
    type(ridge_operator)::raw
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
    raw=op
    if(allocated(raw%offset))deallocate(raw%offset,raw%root_weight)
    task='START';csave='';iterations=0;evaluations=0
    do
      call setulb(p,memory,x,lower,upper,nbd,f,g,tol/epsilon(1._dp),tol, &
        wa,iwa,task,-1,csave,lsave,isave,dsave)
      if(task(:2)=='FG')then
        residual=multiply(raw,x,.false.)-y
        if(allocated(op%offset))residual=residual-op%root_weight*kernel_blas_inner_product(x,op%offset)
        f=.5_dp*kernel_blas_inner_product(residual,residual)+.5_dp*alpha*kernel_blas_inner_product(x,x)
        g=multiply(raw,residual,.true.)+alpha*x
        if(allocated(op%offset))g=g-op%offset*kernel_blas_inner_product(residual,op%root_weight)
        evaluations=evaluations+1
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
