program reference_svc_sparse_driver
  use sklearn_svm,only:svc
  use sklearn_common,only:dp,unique_labels
  use sklearn_sparse,only:csr_matrix
  implicit none
  type(svc)::model
  type(csr_matrix)::x,q
  real(dp),allocatable::weights(:),cw(:),scores(:,:),dense(:,:)
  integer,allocatable::y(:),classes(:)
  integer::cases,c,n,p,m,balanced,shape,ties,gamma_rule
  read(*,*)cases
  do c=1,cases
    model=svc()
    read(*,*)n,p,m,model%kernel,model%c,model%gamma,model%coef0,model%degree,model%tol, &
      model%max_iter,balanced,shape,ties,gamma_rule
    model%decision_function_shape='ovr';if(shape==0)model%decision_function_shape='ovo'
    model%break_ties=ties/=0;model%gamma_rule='scale';if(gamma_rule==1)model%gamma_rule='auto'
    allocate(weights(n),y(n));call read_csr(x);call read_csr(q);read(*,*)y;read(*,*)weights
    classes=unique_labels(y);allocate(cw(size(classes)));read(*,*)cw
    model%class_weight_balanced=balanced==1
    if(balanced==2)then
      call model%fit(x,y,weights,cw)
    else
      call model%fit(x,y,weights)
    end if
    print *,'BEGIN',c
    call emit('classes',real(model%classes_,dp));call emit('support',real(model%support_-1,dp))
    call emit('n_support',real(model%n_support_,dp));call emit('sv_indptr',real(model%support_vectors_csr_%indptr,dp));call emit('sv_indices',real(model%support_vectors_csr_%indices,dp))
    call emit('sv_data',model%support_vectors_csr_%data)
    call emit('dual_coef',reshape(model%dual_coef_,[size(model%dual_coef_)]));call emit('intercept',model%intercept_)
    if(allocated(model%coef_csr_))then
      dense=model%coef_csr_%to_dense();call emit('coef',reshape(dense,[size(dense)]))
    else
      call emit('coef',[real(dp)::])
    end if
    call emit('status',[real(model%fit_status_,dp)])
    call emit('gamma',[model%gamma_]);call emit('class_weight',model%class_weight_)
    scores=model%decision_function(x);call emit('train',reshape(scores,[size(scores)]))
    scores=model%decision_function(q);call emit('query',reshape(scores,[size(scores)]))
    call emit('predict',real(model%predict(q),dp));
    scores=model%decision_function(q%to_dense());call emit('dense_query',reshape(scores,[size(scores)]))
    call emit('dense_predict',real(model%predict(q%to_dense()),dp));call emit('score',[model%score(x,y),model%score(x,y,weights)])
    call emit('native_storage',real([merge(1,0,allocated(model%x_fit_)),merge(1,0,allocated(model%support_vectors_)), &
      merge(1,0,allocated(model%coef_))],dp))
    call emit('native_support_original',real(model%support_original_-1,dp))
    call emit('native_iterations',real(model%n_iter_per_pair_,dp));call emit('native_kkt',model%kkt_violation_)
    print *,'END';deallocate(weights,y,cw)
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
end program
