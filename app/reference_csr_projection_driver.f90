program reference_csr_projection_driver
  use sklearn_common, only: dp
  use sklearn_random_projection
  use sklearn_sparse, only: csr_matrix
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  class(gaussian_random_projection),allocatable :: model
  type(csr_matrix) :: x,q,v,original_x,original_q,original_v,indicator
  real(dp),allocatable :: z(:,:),query(:,:),back(:,:),first(:,:)
  integer,allocatable :: before(:),after(:)
  integer :: c,cases,kind,k,seed,cache,mode,ns
  real(dp) :: eps,density
  call random_seed(size=ns);allocate(before(ns),after(ns))
  read(*,*) cases
  do c=1,cases
    read(*,*) kind,k,seed,cache,eps,density,mode
    call read_csr(x);call read_csr(q);call read_csr(v)
    if(kind==0) then
      allocate(gaussian_random_projection::model)
    else
      allocate(sparse_random_projection::model)
      select type(model)
      type is(sparse_random_projection);model%density=density
      end select
    end if
    model%n_components=k;model%random_state=seed;model%eps=eps;model%compute_inverse_components=cache/=0
    select case(mode)
    case(1);z=model%transform(q)
    case(2);z=model%inverse_transform(v)
    case(3);x%data(1)=ieee_value(0._dp,ieee_quiet_nan)
    case(4);x%data(1)=ieee_value(0._dp,ieee_positive_inf)
    case(5);x%indices(1)=0
    case(6);x%indptr(1)=0
    case(7);x%indptr(2)=size(x%data)+2
    case(8);q%data(1)=ieee_value(0._dp,ieee_quiet_nan)
    case(9);v%data(1)=ieee_value(0._dp,ieee_positive_inf)
    end select
    original_x=x;original_q=q;original_v=v;call random_seed(get=before)
    z=model%fit_transform(x);query=model%transform(q);back=model%inverse_transform(v)
    call random_seed(get=after)
    print *, 'BEGIN',c
    call emit('components',reshape(model%components_,[size(model%components_)]))
    call emit('train_transform',reshape(z,[size(z)]));call emit('query_transform',reshape(query,[size(query)]))
    call emit('inverse',reshape(back,[size(back)]))
    select type(model)
    type is(sparse_random_projection)
      indicator=model%transform_sparse(q);call indicator%validate()
      call emit('csr_shape',real([indicator%n_rows,indicator%n_cols],dp))
      call emit('csr_indptr',real(indicator%indptr,dp));call emit('csr_indices',real(indicator%indices,dp))
      call emit('csr_data',indicator%data)
      indicator=model%fit_transform_sparse(x);call indicator%validate()
      call emit('csr_train_shape',real([indicator%n_rows,indicator%n_cols],dp))
      call emit('csr_train_indptr',real(indicator%indptr,dp));call emit('csr_train_indices',real(indicator%indices,dp))
      call emit('csr_train_data',indicator%data)
    class default
      call emit('csr_shape',[real(dp)::]);call emit('csr_indptr',[real(dp)::]);call emit('csr_indices',[real(dp)::])
      call emit('csr_data',[real(dp)::]);call emit('csr_train_shape',[real(dp)::])
      call emit('csr_train_indptr',[real(dp)::]);call emit('csr_train_indices',[real(dp)::])
      call emit('csr_train_data',[real(dp)::])
    end select
    if(allocated(model%inverse_components_)) then
      call emit('pinv',reshape(model%inverse_components_,[size(model%inverse_components_)]))
    else
      call emit('pinv',[real(dp)::])
    end if
    call emit('metadata',real([model%n_components_,model%n_features_in_,merge(1,0,allocated(model%inverse_components_))],dp))
    select type(model)
    type is(sparse_random_projection);call emit('density',[model%density_])
    class default;call emit('density',[real(dp)::])
    end select
    call emit('preserved',real([merge(1,0,equal_csr(x,original_x)),merge(1,0,equal_csr(q,original_q)), &
      merge(1,0,equal_csr(v,original_v)),merge(1,0,all(before==after))],dp))
    first=model%components_;call model%fit(x)
    k=merge(1,0,all(first==model%components_))
    model%compute_inverse_components=.false.;call model%fit(x)
    call emit('refit',real([k,merge(1,0,.not.allocated(model%inverse_components_))],dp))
    print *, 'END',c
    deallocate(model)
  end do
contains
  subroutine read_csr(a)
    type(csr_matrix),intent(out)::a
    integer :: nnz
    read(*,*) a%n_rows,a%n_cols,nnz
    allocate(a%indptr(a%n_rows+1),a%indices(nnz),a%data(nnz))
    read(*,*) a%indptr;read(*,*) a%indices;read(*,*) a%data
  end subroutine
  logical function equal_csr(a,b) result(same)
    type(csr_matrix),intent(in)::a,b
    same=a%n_rows==b%n_rows.and.a%n_cols==b%n_cols.and.all(a%indptr==b%indptr).and.all(a%indices==b%indices) &
      .and.all(transfer(a%data,[0_int64])==transfer(b%data,[0_int64]))
  end function
  subroutine emit(name,values)
    character(*),intent(in)::name
    real(dp),intent(in)::values(:)
    print *,name,size(values)
    write(*,'(*(es25.16e3,1x))') values
  end subroutine
end program
