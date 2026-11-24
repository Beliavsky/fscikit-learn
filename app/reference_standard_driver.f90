program reference_standard_driver
 use sklearn_common,only:dp
 use sklearn_standard_scaling,only:standard_scaler
 use sklearn_sparse,only:csr_matrix
 use,intrinsic::iso_fortran_env,only:int64
 implicit none
 type(standard_scaler)::model
 type(csr_matrix)::x,b,q,z,original_x,original_b,original_q
 real(dp),allocatable::dense(:,:),a(:,:),w(:),v(:)
 integer,allocatable::before(:),after(:)
 integer::c,cases,fit_kind,batch_kind,query_kind,center,scaling,wm,vm,qcenter,qscaling,ns
 call random_seed(size=ns);allocate(before(ns),after(ns));read(*,*) cases
 do c=1,cases
  read(*,*) fit_kind,batch_kind,query_kind,center,scaling,wm,vm,qcenter,qscaling
  call read_csr(x);call read_csr(b);call read_csr(q)
  if(allocated(w)) deallocate(w,v)
  allocate(w(x%n_rows),v(b%n_rows));read(*,*) w;read(*,*) v
  original_x=x;original_b=b;original_q=q;call random_seed(get=before)
  model=standard_scaler();model%with_mean=center/=0;model%with_std=scaling/=0
  call update(x,fit_kind,wm,w,.true.)
  print *,'BEGIN',c
  call snapshot('fit_')
  if(batch_kind>=0) call update(b,batch_kind,vm,v,.false.)
  call snapshot('batch_')
  model%with_mean=qcenter/=0;model%with_std=qscaling/=0
  if(query_kind==0) then
   dense=q%to_dense();a=model%transform(dense);call dense_output('query',a)
   a=model%inverse_transform(dense);call dense_output('inverse',a)
  else
   z=model%transform(q);call sparse_output('query',z)
   z=model%inverse_transform(q);call sparse_output('inverse',z)
  end if
  call random_seed(get=after)
  call emit('preserved',real([merge(1,0,equal_bits(x,original_x)),merge(1,0,equal_bits(b,original_b)), &
   merge(1,0,equal_bits(q,original_q)),merge(1,0,all(before==after))],dp))
  model%with_mean=center/=0;model%with_std=scaling/=0
  dense=q%to_dense();call model%fit(dense);call snapshot('refit_')
  print *,'END',c
 end do
contains
 subroutine snapshot(prefix)
  character(*),intent(in)::prefix
  call learned(prefix//'mean',model%mean_);call learned(prefix//'var',model%var_)
  call learned(prefix//'scale',model%scale_);call learned(prefix//'counts',model%n_samples_seen_)
  call emit(prefix//'metadata',real([model%n_features_in_,merge(1,0,model%sample_count_scalar_), &
   merge(1,0,model%sample_count_real_)],dp))
 end subroutine
 subroutine update(x,kind,mode,w,first)
  type(csr_matrix),intent(in)::x
  integer,intent(in)::kind,mode
  real(dp),intent(in)::w(:)
  logical,intent(in)::first
  real(dp),allocatable::d(:,:)
  if(kind==0) then
   d=x%to_dense()
   if(first) then
    select case(mode)
    case(0);call model%fit(d)
    case(1);call model%fit(d,w)
    case(2);call model%fit(d,w(1))
    end select
   else
    select case(mode)
    case(0);call model%partial_fit(d)
    case(1);call model%partial_fit(d,w)
    case(2);call model%partial_fit(d,w(1))
    end select
   end if
  else
   if(first) then
    select case(mode)
    case(0);call model%fit(x)
    case(1);call model%fit(x,w)
    case(2);call model%fit(x,w(1))
    end select
   else
    select case(mode)
    case(0);call model%partial_fit(x)
    case(1);call model%partial_fit(x,w)
    case(2);call model%partial_fit(x,w(1))
    end select
   end if
  end if
 end subroutine
 subroutine read_csr(a)
  type(csr_matrix),intent(out)::a
  integer::nnz
  read(*,*) a%n_rows,a%n_cols,nnz;allocate(a%indptr(a%n_rows+1),a%indices(nnz),a%data(nnz))
  read(*,*) a%indptr;read(*,*) a%indices;read(*,*) a%data
 end subroutine
 logical function equal_bits(a,b)
  type(csr_matrix),intent(in)::a,b
  equal_bits=all(a%indptr==b%indptr).and.all(a%indices==b%indices).and. &
   all(transfer(a%data,[0_int64])==transfer(b%data,[0_int64]))
 end function
 subroutine learned(name,a)
  character(*),intent(in)::name
  real(dp),allocatable,intent(in)::a(:)
  if(allocated(a)) then
   call emit(name,a)
  else
   call emit(name,[real(dp)::])
  end if
 end subroutine
 subroutine dense_output(name,a)
  character(*),intent(in)::name
  real(dp),intent(in)::a(:,:)
  call emit(name//'_shape',real(shape(a),dp));call emit(name//'_ptr',[real(dp)::])
  call emit(name//'_idx',[real(dp)::]);call emit(name//'_data',reshape(a,[size(a)]))
 end subroutine
 subroutine sparse_output(name,a)
  character(*),intent(in)::name
  type(csr_matrix),intent(in)::a
  call emit(name//'_shape',real([a%n_rows,a%n_cols],dp));call emit(name//'_ptr',real(a%indptr,dp))
  call emit(name//'_idx',real(a%indices,dp));call emit(name//'_data',a%data)
 end subroutine
 subroutine emit(name,a)
  character(*),intent(in)::name
  real(dp),intent(in)::a(:)
  print *,name,size(a);write(*,'(*(es25.16e3,1x))') a
 end subroutine
end program
