program reference_robust_driver
 use sklearn_common,only:dp
 use sklearn_robust_scaling,only:robust_scaler,robust_scale
 use sklearn_sparse,only:csr_matrix
 use,intrinsic::iso_fortran_env,only:int64
 use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_positive_inf
 implicit none
 type(robust_scaler)::model
 type(csr_matrix)::x,q,z,original_x,original_q
 real(dp),allocatable::dense(:,:),dense_q(:,:),a(:,:),vector(:)
 integer,allocatable::before(:),after(:)
 integer::c,cases,fit_kind,query_kind,center,scaling,unit,axis,mode,action,qcenter,qscaling,ns
 real(dp)::low,high
 call random_seed(size=ns);allocate(before(ns),after(ns));read(*,*) cases
 do c=1,cases
  read(*,*) fit_kind,query_kind,center,scaling,unit,axis,mode,action,qcenter,qscaling,low,high
  call read_csr(x);call read_csr(q)
  model=robust_scaler();model%with_centering=center/=0;model%with_scaling=scaling/=0
  model%unit_variance=unit/=0;model%quantile_low=low;model%quantile_high=high
  if(mode==1) z=model%transform(q)
  if(mode==2) x%data(1)=ieee_value(0._dp,ieee_positive_inf)
  if(mode==3) q%data(1)=ieee_value(0._dp,ieee_positive_inf)
  if(mode==4) x%indices(1)=0
  if(mode==5) x%indptr(1)=0
  if(mode==6) x%indptr(2)=size(x%data)+2
  original_x=x;original_q=q;call random_seed(get=before)
  if(fit_kind==0) then
   dense=x%to_dense();call model%fit(dense)
  else
   call model%fit(x)
  end if
  print *,'BEGIN',c
  call learned('center',model%center_);call learned('scale',model%scale_)
  call emit('metadata',real([model%n_features_in_,merge(1,0,allocated(model%center_)), &
    merge(1,0,allocated(model%scale_))],dp))
  if(fit_kind==0) then
   a=model%transform(dense);call dense_output('train',a)
  else
   z=model%transform(x);call sparse_output('train',z)
  end if
  model%with_centering=qcenter/=0;model%with_scaling=qscaling/=0
  if(query_kind==0) then
   dense_q=q%to_dense()
   if(action/=1) then
    a=model%transform(dense_q);call dense_output('query',a)
   end if
   a=model%inverse_transform(dense_q);call dense_output('inverse',a)
  else
   if(action/=1) then
    z=model%transform(q);call sparse_output('query',z)
   end if
   z=model%inverse_transform(q);call sparse_output('inverse',z)
  end if
  if(fit_kind==0) then
   a=robust_scale(dense,center/=0,scaling/=0,low,high,unit/=0,axis);call dense_output('free',a)
   vector=reshape(dense,[size(dense)])
   call emit('vector',robust_scale(vector,center/=0,scaling/=0,low,high,unit/=0,axis))
  else
   z=robust_scale(x,center/=0,scaling/=0,low,high,unit/=0,axis);call sparse_output('free',z)
   call emit('vector',[real(dp)::])
  end if
  call random_seed(get=after)
  call emit('preserved',real([merge(1,0,equal_bits(x,original_x)),merge(1,0,equal_bits(q,original_q)), &
    merge(1,0,all(before==after))],dp))
  model%with_centering=center/=0;model%with_scaling=scaling/=0;dense_q=q%to_dense();call model%fit(dense_q)
  call learned('refit_center',model%center_);call learned('refit_scale',model%scale_)
  call emit('refit',real([model%n_features_in_,merge(1,0,allocated(model%center_)), &
    merge(1,0,allocated(model%scale_))],dp))
  print *,'END',c
 end do
contains
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
