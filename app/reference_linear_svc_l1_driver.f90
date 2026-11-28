program reference_linear_svc_l1_driver
  use sklearn_common,only:dp,unique_labels
  use sklearn_linear_svm,only:linear_svc
  use sklearn_sparse,only:csr_matrix
  implicit none
  type(linear_svc)::model
  type(csr_matrix)::x,q
  real(dp),allocatable::dense(:,:),w(:),cw(:),scores(:,:)
  integer,allocatable::y(:),classes(:)
  real(dp)::scaling,tol,cost
  integer::cases,case_id,input_mode,loss,dual,seed,intercept,cw_mode,max_iter
  read(*,*)cases
  do case_id=1,cases
    read(*,*)input_mode,loss,dual,seed,intercept,cw_mode,max_iter,scaling,tol,cost
    call read_csr(x);call read_csr(q)
    allocate(y(x%n_rows),w(x%n_rows));read(*,*)y;read(*,*)w
    classes=unique_labels(y);allocate(cw(size(classes)));read(*,*)cw
    model=linear_svc();model%penalty='l1';model%random_state=seed;model%fit_intercept=intercept/=0
    model%max_iter=max_iter;model%intercept_scaling=scaling;model%tol=tol;model%c=cost
    model%store_dual_variables=.true.
    if(loss==0)model%loss='hinge'
    select case(dual)
    case(0);model%dual='false'
    case(1);model%dual='true'
    case(2);model%dual='auto'
    end select
    model%class_weight_balanced=cw_mode==1
    if(input_mode==0)then
      dense=x%to_dense()
      if(cw_mode==2)then
        call model%fit(dense,y,w,cw)
      else
        call model%fit(dense,y,w)
      end if
    else
      if(cw_mode==2)then
        call model%fit(x,y,w,cw)
      else
        call model%fit(x,y,w)
      end if
    end if
    print *,'BEGIN',case_id
    call emit('classes',real(model%classes_,dp));call emit('coef',reshape(model%coef_,[size(model%coef_)]))
    call emit('intercept',model%intercept_);call emit('iterations',[real(model%n_iter_,dp)])
    scores=model%decision_function(x);call emit('train',reshape(scores,[size(scores)]))
    scores=model%decision_function(q);call emit('query',reshape(scores,[size(scores)]))
    scores=model%decision_function(q%to_dense());call emit('dense_query',reshape(scores,[size(scores)]))
    call emit('predict',real(model%predict(q),dp));call emit('score',[model%score(x,y)])
    call emit('objective',model%objective_);call emit('dual_objective',model%dual_objective_)
    call emit('gap',model%duality_gap_);call emit('kkt',model%kkt_violation_)
    call emit('gradient',model%gradient_norm_)
    if(allocated(model%dual_variables_))then
      call emit('alpha',reshape(model%dual_variables_,[size(model%dual_variables_)]))
    else
      call emit('alpha',[real(dp)::])
    end if
    print *,'END',case_id
    deallocate(y,w,cw)
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
