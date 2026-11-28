program reference_linear_svr_dual_driver
  use sklearn_common,only:dp
  use sklearn_linear_svm,only:linear_svr
  use sklearn_sparse,only:csr_matrix
  implicit none
  type(linear_svr)::model
  type(csr_matrix)::x,q
  real(dp),allocatable::dense(:,:),query(:,:),y(:),w(:),values(:)
  real(dp)::tube,scaling,tolerance,cost
  integer::cases,case_id,input_mode,loss,dual,seed,intercept,weighted,max_iter
  read(*,*)cases
  do case_id=1,cases
    read(*,*)input_mode,loss,dual,seed,intercept,weighted,max_iter,tube,scaling,tolerance,cost
    call read_csr(x);call read_csr(q)
    allocate(y(x%n_rows),w(x%n_rows));read(*,*)y;read(*,*)w
    model=linear_svr();model%random_state=seed;model%fit_intercept=intercept/=0
    model%max_iter=max_iter;model%epsilon=tube;model%intercept_scaling=scaling;model%tol=tolerance;model%c=cost
    if(loss==1)model%loss='squared_epsilon_insensitive'
    select case(dual)
    case(0);model%dual='false'
    case(1);model%dual='true'
    case(2);model%dual='auto'
    end select
    if(input_mode==0)then
      dense=x%to_dense();query=q%to_dense()
      if(weighted==1)then
        call model%fit(dense,y,w)
      else
        call model%fit(dense,y)
      end if
    else
      if(weighted==1)then
        call model%fit(x,y,w)
      else
        call model%fit(x,y)
      end if
    end if
    print *,'BEGIN',case_id
    call emit('coef',model%coef_);call emit('intercept',[model%intercept_])
    call emit('iterations',[real(model%n_iter_,dp)])
    call emit('train',model%predict(x));call emit('query',model%predict(q))
    call emit('dense_query',model%predict(q%to_dense()));call emit('score',[model%score(x,y)])
    call emit('diagnostics',[model%objective_,model%dual_objective_,model%duality_gap_,model%kkt_violation_])
    if(allocated(model%dual_variables_))then
      call emit('beta',model%dual_variables_)
    else
      call emit('beta',[real(dp)::])
    end if
    call emit('converged',[real(merge(1,0,model%converged_),dp)])
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
end program
