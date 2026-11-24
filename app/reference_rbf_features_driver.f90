program reference_rbf_features_driver
  use sklearn_common, only: dp
  use sklearn_kernel_approximation, only: rbf_sampler
  use sklearn_sparse, only: csr_matrix
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(rbf_sampler) :: model
  type(csr_matrix) :: x,q,original_x,original_q
  real(dp),allocatable :: dense(:,:),query(:,:),train(:,:),z(:,:),first(:,:),offset(:)
  integer,allocatable :: before(:),after(:)
  integer :: cases,c,k,seed,fit_kind,query_kind,scale,mode,mutate,ns
  real(dp) :: gamma
  call random_seed(size=ns);allocate(before(ns),after(ns))
  read(*,*) cases
  do c=1,cases
    read(*,*) k,seed,fit_kind,query_kind,scale,gamma,mode,mutate
    call read_csr(x);call read_csr(q)
    model=rbf_sampler();model%n_components=k;model%random_state=seed
    model%gamma=gamma;model%gamma_scale=scale/=0
    if(mode==3) x%data(1)=ieee_value(0._dp,ieee_quiet_nan)
    if(mode==4) x%data(1)=ieee_value(0._dp,ieee_positive_inf)
    if(mode==5) q%data(1)=ieee_value(0._dp,ieee_quiet_nan)
    if(mode==6) q%data(1)=ieee_value(0._dp,ieee_positive_inf)
    if(mode==7) x%indices(1)=0
    if(mode==8) x%indptr(1)=0
    if(mode==9) x%indptr(2)=size(x%data)+2
    if(mode==1) then
      if(query_kind==0) then
        dense=q%to_dense();z=model%transform(dense)
      else
        z=model%transform(q)
      end if
    end if
    original_x=x;original_q=q;call random_seed(get=before)
    if(fit_kind==0) then
      dense=x%to_dense();train=model%fit_transform(dense)
    else
      train=model%fit_transform(x)
    end if
    if(mutate>0) model%n_components=mutate
    if(query_kind==0) then
      query=q%to_dense();z=model%transform(query)
    else
      z=model%transform(q)
    end if
    call random_seed(get=after)
    print *, 'BEGIN',c
    call emit('weights',reshape(model%random_weights_,[size(model%random_weights_)]))
    call emit('offset',model%random_offset_);call emit('gamma',[model%gamma_])
    call emit('train',reshape(train,[size(train)]));call emit('query',reshape(z,[size(z)]))
    call emit('metadata',real([model%n_features_in_,size(model%random_weights_,2),model%n_components],dp))
    call emit('preserved',real([merge(1,0,equal_csr(x,original_x)),merge(1,0,equal_csr(q,original_q)), &
      merge(1,0,all(before==after))],dp))
    first=model%random_weights_;offset=model%random_offset_;model%n_components=k
    if(fit_kind==0) then
      call model%fit(dense)
    else
      call model%fit(x)
    end if
    call emit('repeat',real([merge(1,0,equal_values(first,model%random_weights_)), &
      merge(1,0,equal_values(reshape(offset,[size(offset),1]),reshape(model%random_offset_,[size(offset),1])))],dp))
    print *, 'END',c
  end do
contains
  subroutine read_csr(a)
    type(csr_matrix),intent(out)::a
    integer::nnz
    read(*,*) a%n_rows,a%n_cols,nnz
    allocate(a%indptr(a%n_rows+1),a%indices(nnz),a%data(nnz))
    read(*,*) a%indptr;read(*,*) a%indices;read(*,*) a%data
  end subroutine
  logical function equal_csr(a,b) result(same)
    type(csr_matrix),intent(in)::a,b
    same=a%n_rows==b%n_rows.and.a%n_cols==b%n_cols.and.all(a%indptr==b%indptr).and.all(a%indices==b%indices) &
      .and.all(transfer(a%data,[0_int64])==transfer(b%data,[0_int64]))
  end function
  logical function equal_values(a,b) result(same)
    real(dp),intent(in)::a(:,:),b(:,:)
    same=all(transfer(a,[0_int64])==transfer(b,[0_int64]))
  end function
  subroutine emit(name,values)
    character(*),intent(in)::name
    real(dp),intent(in)::values(:)
    print *,name,size(values)
    write(*,'(*(es25.16e3,1x))') values
  end subroutine
end program
