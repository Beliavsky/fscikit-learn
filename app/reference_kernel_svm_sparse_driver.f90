program reference_kernel_svm_sparse_driver
  use sklearn_common, only:dp
  use sklearn_kernel_svm, only:svr,nu_svr,one_class_svm
  use sklearn_sparse, only:csr_matrix
  implicit none
  type(svr)::s
  type(nu_svr)::v
  type(one_class_svm)::o
  type(csr_matrix)::x,q
  real(dp),allocatable::y(:),w(:),values(:),dense(:,:)
  real(dp)::gamma,c,nu,eps
  integer::cases,case_id,family,kernel,rule,weighted
  character(16),parameter::kernels(3)=[character(16)::'rbf','linear','poly']
  read(*,*)cases
  do case_id=1,cases
    read(*,*)family,kernel,rule,weighted,gamma,c,nu,eps
    call read_csr(x);call read_csr(q)
    allocate(y(x%n_rows),w(x%n_rows));read(*,*)y;read(*,*)w
    s=svr();v=nu_svr();o=one_class_svm()
    s%kernel=kernels(kernel);v%kernel=kernels(kernel);o%kernel=kernels(kernel)
    s%gamma=gamma;v%gamma=gamma;o%gamma=gamma
    if(rule==1)then
      s%gamma_rule='auto';v%gamma_rule='auto';o%gamma_rule='auto'
    end if
    s%degree=2;v%degree=2;o%degree=2;s%coef0=.4_dp;v%coef0=.4_dp;o%coef0=.4_dp
    s%tol=1.e-8_dp;v%tol=1.e-8_dp;o%tol=1.e-8_dp
    s%max_iter=200000;v%max_iter=200000;o%max_iter=200000
    s%c=c;v%c=c;s%epsilon=eps;v%nu=nu;o%nu=nu
    dense=q%to_dense()
    print *,'BEGIN',case_id
    select case(family)
    case(1)
      if(weighted==1)then
        call s%fit(x,y,w)
      else
        call s%fit(x,y)
      end if
      call emit('query',s%predict(q));call emit('dense_query',s%predict(dense))
      call emit('parameters',[s%intercept_,s%gamma_]);call emit('support',real(s%support_,dp))
      call emit('dual',s%dual_coef_);call emit_csr(s%support_vectors_csr_)
      call emit('score',[s%score(x,y)])
      call emit('storage',real([merge(1,0,allocated(s%x_fit_)),merge(1,0,allocated(s%support_vectors_))],dp))
      call emit('converged',[real(merge(1,0,s%converged_),dp)])
      if(kernel==2)then
        dense=s%coef_csr_%to_dense();call emit('coef',dense(1,:))
      end if
    case(2)
      if(weighted==1)then
        call v%fit(x,y,w)
      else
        call v%fit(x,y)
      end if
      call emit('query',v%predict(q));call emit('dense_query',v%predict(dense))
      call emit('parameters',[v%intercept_,v%gamma_]);call emit('support',real(v%support_,dp))
      call emit('dual',v%dual_coef_);call emit_csr(v%support_vectors_csr_)
      call emit('score',[v%score(x,y)])
      call emit('storage',real([merge(1,0,allocated(v%x_fit_)),merge(1,0,allocated(v%support_vectors_))],dp))
      call emit('converged',[real(merge(1,0,v%converged_),dp)])
      if(kernel==2)then
        dense=v%coef_csr_%to_dense();call emit('coef',dense(1,:))
      end if
    case(3)
      if(weighted==1)then
        call o%fit(x,w)
      else
        call o%fit(x)
      end if
      call emit('query',o%decision_function(q));call emit('dense_query',o%decision_function(dense))
      call emit('raw',o%score_samples(q));call emit('labels',real(o%predict(q),dp))
      call emit('parameters',[o%intercept_,o%gamma_]);call emit('support',real(o%support_,dp))
      call emit('dual',o%dual_coef_);call emit_csr(o%support_vectors_csr_)
      call emit('storage',real([merge(1,0,allocated(o%x_fit_)),merge(1,0,allocated(o%support_vectors_))],dp))
      call emit('converged',[real(merge(1,0,o%converged_),dp)])
      if(kernel==2)then
        dense=o%coef_csr_%to_dense();call emit('coef',dense(1,:))
      end if
    end select
    print *,'END',case_id
    deallocate(y,w)
  end do
contains
  subroutine read_csr(a)
    type(csr_matrix),intent(out)::a
    integer::nnz
    read(*,*)a%n_rows,a%n_cols,nnz
    allocate(a%indptr(a%n_rows+1),a%indices(nnz),a%data(nnz))
    read(*,*)a%indptr;read(*,*)a%indices;read(*,*)a%data
  end subroutine
  subroutine emit(name,values)
    character(*),intent(in)::name
    real(dp),intent(in)::values(:)
    print *,name,size(values)
    write(*,'(*(es25.16e3,1x))')values
  end subroutine
  subroutine emit_csr(a)
    type(csr_matrix),intent(in)::a
    call emit('sv_indptr',real(a%indptr,dp));call emit('sv_indices',real(a%indices,dp));call emit('sv_data',a%data)
  end subroutine
end program
