program reference_sgd_one_class_sparse_driver
  use sklearn_sgd_one_class,only:sgd_one_class_svm
  use sklearn_common,only:dp
  use sklearn_sparse,only:csr_matrix
  implicit none
  type(sgd_one_class_svm)::model
  type(csr_matrix)::x,q
  real(dp),allocatable::dense(:,:),w(:)
  integer,allocatable::labels(:)
  integer::cases,c,stages,s,operation,representation,shuffle,intercept
  read(*,*)cases
  do c=1,cases
    model=sgd_one_class_svm()
    read(*,*)stages,model%learning_rate,model%nu,model%eta0,model%power_t,model%tol, &
      model%max_iter,model%n_iter_no_change,model%average,model%random_state,shuffle,intercept
    model%shuffle=shuffle/=0;model%fit_intercept=intercept/=0
    do s=1,stages
      read(*,*)operation,representation;call read_csr(x);call read_csr(q)
      allocate(w(x%n_rows));read(*,*)w
      model%warm_start=operation==2
      if(representation==0)then
        dense=x%to_dense()
        select case(operation)
        case(1);call model%partial_fit(dense,w)
        case(3);call model%fit(dense,w,coef_init=spread(.1_dp,1,x%n_cols),offset_init=.2_dp)
        case(4);labels=model%fit_predict(dense,w)
        case default;call model%fit(dense,w)
        end select
      else
        select case(operation)
        case(1);call model%partial_fit(x,w)
        case(3);call model%fit(x,w,coef_init=spread(.1_dp,1,x%n_cols),offset_init=.2_dp)
        case(4);labels=model%fit_predict(x,w)
        case default;call model%fit(x,w)
        end select
      end if
      print *,'BEGIN',c,s
      call emit('coef',model%coef_);call emit('offset',[model%offset_])
      call emit('decision',model%decision_function(q));call emit('samples',model%score_samples(q))
      call emit('predict',real(model%predict(q),dp))
      call emit('dense_decision',model%decision_function(q%to_dense()))
      call emit('dense_samples',model%score_samples(q%to_dense()))
      call emit('dense_predict',real(model%predict(q%to_dense()),dp))
      call emit('state',[model%t_,real(model%n_iter_,dp),real(model%n_features_in_,dp)])
      print *,'END';deallocate(w)
    end do
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
