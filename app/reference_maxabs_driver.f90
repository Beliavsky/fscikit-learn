program reference_maxabs_driver
 use sklearn_common,only:dp
 use sklearn_maxabs_scaling,only:maxabs_scaler,maxabs_scale
 use sklearn_sparse,only:csr_matrix
 use,intrinsic::iso_fortran_env,only:int64
 use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan,ieee_positive_inf
 implicit none
 type(maxabs_scaler)::model
 type(csr_matrix)::x,b,q,z,inv,original_x,original_b,original_q
 real(dp),allocatable::dense(:,:),dense_b(:,:),dense_q(:,:),a(:,:),back(:,:),vector(:)
 integer,allocatable::before(:),after(:)
 integer::cases,c,fit_kind,batch_kind,query_kind,stream,axis,mode,ns,action
 call random_seed(size=ns);allocate(before(ns),after(ns))
 read(*,*) cases
 do c=1,cases
  read(*,*) fit_kind,batch_kind,query_kind,stream,axis,mode,action
  call read_csr(x);call read_csr(b);call read_csr(q)
  model=maxabs_scaler()
  if(mode==1) z=model%transform(q)
  if(mode==2) x%data(1)=ieee_value(0._dp,ieee_positive_inf)
  if(mode==3) q%data(1)=ieee_value(0._dp,ieee_positive_inf)
  if(mode==4) b%data(1)=ieee_value(0._dp,ieee_positive_inf)
  if(mode==5) x%indices(1)=0
  if(mode==6) x%indptr(1)=0
  if(mode==7) x%indptr(2)=size(x%data)+2
  original_x=x;original_b=b;original_q=q;call random_seed(get=before)
  if(fit_kind==0) then
   dense=x%to_dense();call model%fit(dense)
  else
   call model%fit(x)
  end if
  if(mode==8) model%n_samples_seen_=huge(0_int64)
  if(stream==1) then
   if(batch_kind==0) then
    dense_b=b%to_dense();call model%partial_fit(dense_b)
   else
    call model%partial_fit(b)
   end if
  end if
  if(action==1) model%scale_=0
  if(action==2) model%scale_=-model%scale_
  print *, 'BEGIN',c
  call emit('maximum',model%max_abs_);call emit('scale',model%scale_)
  call emit('metadata',real([int(model%n_features_in_,int64),model%n_samples_seen_],dp))
  if(query_kind==0) then
   dense_q=q%to_dense()
   if(action==3) then
    back=model%inverse_transform(dense_q);call emit_dense('inverse',back)
   else
    a=model%transform(dense_q);call emit_dense('mapped',a)
    back=model%inverse_transform(a);call emit_dense('inverse',back)
   end if
  else
   if(action==3) then
    inv=model%inverse_transform(q);call emit_sparse('inverse',inv)
   else
    z=model%transform(q);call emit_sparse('mapped',z)
    inv=model%inverse_transform(z);call emit_sparse('inverse',inv)
   end if
  end if
  if(fit_kind==0) then
   a=maxabs_scale(dense,axis);call emit_dense('free',a)
   vector=reshape(dense,[size(dense)]);call emit('vector',maxabs_scale(vector,axis))
  else
   z=maxabs_scale(x,axis);call emit_sparse('free',z);call emit('vector',[real(dp)::])
  end if
  call random_seed(get=after)
  call emit('preserved',real([merge(1,0,equal_csr(x,original_x)),merge(1,0,equal_csr(b,original_b)), &
    merge(1,0,equal_csr(q,original_q)),merge(1,0,all(before==after))],dp))
  ! A full refit must replace learned dimensions/counts even after streaming.
  call model%fit(q);call emit('refit',real([int(model%n_features_in_,int64),model%n_samples_seen_],dp))
  call emit('refit_maximum',model%max_abs_)
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
 subroutine emit_dense(name,a)
  character(*),intent(in)::name
  real(dp),intent(in)::a(:,:)
  call emit(name//'_shape',real(shape(a),dp));call emit(name//'_ptr',[real(dp)::])
  call emit(name//'_idx',[real(dp)::]);call emit(name//'_data',reshape(a,[size(a)]))
 end subroutine
 subroutine emit_sparse(name,a)
  character(*),intent(in)::name
  type(csr_matrix),intent(in)::a
  call emit(name//'_shape',real([a%n_rows,a%n_cols],dp));call emit(name//'_ptr',real(a%indptr,dp))
  call emit(name//'_idx',real(a%indices,dp));call emit(name//'_data',a%data)
 end subroutine
 subroutine emit(name,a)
  character(*),intent(in)::name
  real(dp),intent(in)::a(:)
  print *,name,size(a)
  write(*,'(*(es25.16e3,1x))') a
 end subroutine
end program
