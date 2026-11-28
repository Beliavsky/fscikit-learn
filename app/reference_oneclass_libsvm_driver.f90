program reference_oneclass_libsvm_driver
  use sklearn_kernel_svm,only:one_class_svm
  use sklearn_sparse,only:csr_matrix
  use sklearn_common,only:dp
  implicit none
  type(one_class_svm)::model
  type(csr_matrix)::sx,sq
  real(dp),allocatable::x(:,:),q(:,:),w(:),values(:)
  integer,allocatable::positive(:),original(:),labels(:)
  integer::cases,c,fmt,n,p,m,rule,shrinking,i
  read(*,*)cases
  do c=1,cases
    model=one_class_svm()
    read(*,*)fmt,n,p,m,model%kernel,model%nu,model%gamma,model%coef0,model%degree,model%tol,model%max_iter,rule,shrinking
    model%gamma_rule='scale';if(rule==1)model%gamma_rule='auto'
    model%shrinking=shrinking/=0
    allocate(w(n))
    if(fmt==1)then
      call read_csr(sx);call read_csr(sq);read(*,*)w;call model%fit(sx,w)
    else
      allocate(x(n,p),q(m,p));read(*,*)x;read(*,*)q;read(*,*)w;call model%fit(x,w)
    end if
    positive=pack([(i,i=1,n)],w>0);original=model%support_
    if(fmt==1)original=positive(model%support_)
    print *,'BEGIN',c
    call emit('support',real(model%support_-1,dp));call emit('support_original',real(original-1,dp))
    call emit('n_support',[real(model%n_support_,dp)]);call emit('dual_coef',model%dual_coef_)
    call emit('intercept',[model%intercept_]);call emit('offset',[model%offset_]);call emit('gamma',[model%gamma_])
    if(allocated(model%coef_))then
      call emit('coef',model%coef_)
    else
      call emit('coef',[real(dp)::])
    end if
    if(fmt==1)then
      call emit('vectors',[real(dp)::]);call emit('sv_indptr',real(model%support_vectors_csr_%indptr,dp))
      call emit('sv_indices',real(model%support_vectors_csr_%indices,dp));call emit('sv_data',model%support_vectors_csr_%data)
      call emit('train',model%decision_function(sx));call emit('query',model%decision_function(sq))
      call emit('train_raw',model%score_samples(sx));call emit('query_raw',model%score_samples(sq))
      call emit('predict',real(model%predict(sq),dp));labels=model%fit_predict(sx,w);call emit('fit_predict',real(labels,dp))
      call emit('dense_query',model%decision_function(sq%to_dense()));call emit('dense_predict',real(model%predict(sq%to_dense()),dp))
    else
      call emit('vectors',reshape(model%support_vectors_,[size(model%support_vectors_)]))
      call emit('sv_indptr',[real(dp)::]);call emit('sv_indices',[real(dp)::]);call emit('sv_data',[real(dp)::])
      call emit('train',model%decision_function(x));call emit('query',model%decision_function(q))
      call emit('train_raw',model%score_samples(x));call emit('query_raw',model%score_samples(q))
      call emit('predict',real(model%predict(q),dp));labels=model%fit_predict(x,w);call emit('fit_predict',real(labels,dp))
      deallocate(x,q)
    end if
    call emit('status',[real(model%fit_status_,dp)]);call emit('native_iterations',[real(model%n_iter_,dp)])
    call emit('native_kkt',[model%kkt_violation_])
    print *,'END';deallocate(w)
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
