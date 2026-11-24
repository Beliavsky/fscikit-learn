program reference_additive_csr_driver
 use sklearn_common,only:dp
 use sklearn_kernel_approximation,only:additive_chi2_sampler
 use sklearn_sparse,only:csr_matrix
 use,intrinsic::iso_fortran_env,only:int64
 use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan,ieee_positive_inf
 implicit none
 type(additive_chi2_sampler)::model
 type(csr_matrix)::x,q,original_x,original_q,out
 real(dp),allocatable::dense(:,:),z(:,:)
 integer,allocatable::before(:),after(:)
 integer::cases,c,fit_kind,query_kind,steps,query_steps,action,mode,ns
 real(dp)::interval,query_interval
 call random_seed(size=ns);allocate(before(ns),after(ns))
 read(*,*) cases
 do c=1,cases
  read(*,*) fit_kind,query_kind,steps,interval,query_steps,query_interval,action,mode
  call read_csr(x);call read_csr(q)
  if(mode==1) x%data(1)=ieee_value(0._dp,ieee_quiet_nan)
  if(mode==2) x%data(1)=ieee_value(0._dp,ieee_positive_inf)
  if(mode==3) q%data(1)=ieee_value(0._dp,ieee_quiet_nan)
  if(mode==4) q%data(1)=ieee_value(0._dp,ieee_positive_inf)
  if(mode==5) x%indices(1)=0
  if(mode==6) x%indptr(1)=0
  if(mode==7) x%indptr(2)=size(x%data)+2
  model=additive_chi2_sampler();model%sample_steps=steps;model%sample_interval=interval
  original_x=x;original_q=q;call random_seed(get=before)
  print *,'BEGIN',c
  if(fit_kind==0) then
   dense=x%to_dense()
   if(action==0) then
    z=model%fit_transform(dense)
   else
    z=model%transform(dense)
   end if
   call emit_dense('train',z)
  else
   if(action==0) then
    out=model%fit_transform(x)
   else
    out=model%transform(x)
   end if
   call out%validate();call emit_csr('train',out)
  end if
  model%sample_steps=query_steps;model%sample_interval=query_interval
  if(query_kind==0) then
   dense=q%to_dense();z=model%transform(dense);call emit_dense('query',z)
  else
   out=model%transform(q);call out%validate();call emit_csr('query',out)
  end if
  call random_seed(get=after)
  call emit('metadata',real([model%n_features_in_,model%sample_steps],dp))
  call emit('controls',[model%sample_interval])
  call emit('preserved',real([merge(1,0,equal_csr(x,original_x)),merge(1,0,equal_csr(q,original_q)), &
    merge(1,0,all(before==after))],dp))
  print *,'END',c
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
 subroutine emit_dense(prefix,a)
  character(*),intent(in)::prefix
  real(dp),intent(in)::a(:,:)
  call emit(prefix//'_shape',real(shape(a),dp));call emit(prefix//'_ptr',[real(dp)::])
  call emit(prefix//'_idx',[real(dp)::]);call emit(prefix//'_data',reshape(a,[size(a)]))
 end subroutine
 subroutine emit_csr(prefix,a)
  character(*),intent(in)::prefix
  type(csr_matrix),intent(in)::a
  call emit(prefix//'_shape',real([a%n_rows,a%n_cols],dp));call emit(prefix//'_ptr',real(a%indptr,dp))
  call emit(prefix//'_idx',real(a%indices,dp));call emit(prefix//'_data',a%data)
 end subroutine
 subroutine emit(name,a)
  character(*),intent(in)::name
  real(dp),intent(in)::a(:)
  print *,name,size(a)
  write(*,'(*(es25.16e3,1x))') a
 end subroutine
end program
