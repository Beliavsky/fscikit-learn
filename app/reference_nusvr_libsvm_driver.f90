program reference_nusvr_libsvm_driver
  use sklearn_kernel_svm,only:nu_svr
  use sklearn_sparse,only:csr_matrix
  use sklearn_common,only:dp
  implicit none
  type(nu_svr)::model
  type(csr_matrix)::sx,sq
  real(dp),allocatable::x(:,:),q(:,:),w(:),y(:)
  integer,allocatable::positive(:),original(:),labels(:)
  integer::cases,c,fmt,n,p,m,rule,shrinking,i
  read(*,*)cases
  do c=1,cases
    model=nu_svr()
    read(*,*)fmt,n,p,m,model%kernel,model%c,model%nu,model%gamma,model%coef0,model%degree,model%tol,model%max_iter,rule,shrinking
    model%gamma_rule='scale';if(rule==1)model%gamma_rule='auto'
    model%shrinking=shrinking/=0
    allocate(w(n),y(n))
    if(fmt==1)then
      call read_csr(sx);call read_csr(sq);read(*,*)y;read(*,*)w;call model%fit(sx,y,w)
    else
      allocate(x(n,p),q(m,p));read(*,*)x;read(*,*)q;read(*,*)y;read(*,*)w;call model%fit(x,y,w)
    end if
    positive=pack([(i,i=1,n)],w>0);original=model%support_
    if(fmt==1)original=positive(model%support_)
    print *,'BEGIN',c
    call emit('support',real(model%support_-1,dp));call emit('support_original',real(original-1,dp))
    call emit('n_support',[real(model%n_support_,dp)]);call emit('dual_coef',model%dual_coef_)
    call emit('intercept',[model%intercept_]);call emit('gamma',[model%gamma_])
    if(allocated(model%coef_))then
      call emit('coef',model%coef_)
    else
      call emit('coef',[real(dp)::])
    end if
    if(fmt==1)then
      call emit('vectors',[real(dp)::]);call emit('sv_indptr',real(model%support_vectors_csr_%indptr,dp))
      call emit('sv_indices',real(model%support_vectors_csr_%indices,dp));call emit('sv_data',model%support_vectors_csr_%data)
      call emit('train',model%decision_function(sx));call emit('query',model%decision_function(sq))
      call emit('score',[model%score(sx,y),model%score(sx,y,w)])
      call emit('predict',model%predict(sq))
      call emit('dense_query',model%decision_function(sq%to_dense()));call emit('dense_predict',model%predict(sq%to_dense()))
    else
      call emit('vectors',reshape(model%support_vectors_,[size(model%support_vectors_)]))
      call emit('sv_indptr',[real(dp)::]);call emit('sv_indices',[real(dp)::]);call emit('sv_data',[real(dp)::])
      call emit('train',model%decision_function(x));call emit('query',model%decision_function(q))
      call emit('score',[model%score(x,y),model%score(x,y,w)])
      call emit('predict',model%predict(q))
      deallocate(x,q)
    end if
    call emit('status',[real(model%fit_status_,dp)]);call emit('native_iterations',[real(model%n_iter_,dp)])
    call emit('native_kkt',[model%kkt_violation_])
    call emit('native_alpha',reshape(model%dual_variables_,[size(model%dual_variables_)]))
    call emit('native_epsilon',[model%epsilon_]);call emit('native_epsilon_signed',[model%epsilon_signed_])
    print *,'END';deallocate(w,y)
  end do
contains
  subroutine read_csr(a)
    type(csr_matrix),intent(out)::a
    integer::nnz
    read(*,*)a%n_rows,a%n_cols,nnz
    allocate(a%indptr(a%n_rows+1),a%indices(nnz),a%data(nnz));read(*,*)a%indptr;read(*,*)a%indices;read(*,*)a%data
  end subroutine
  subroutine emit(name,a)
    character(*),intent(in)::name
    real(dp),intent(in)::a(:)
    print *,name,size(a)
    write(*,'(*(es25.16e3,1x))')a
  end subroutine
end program
