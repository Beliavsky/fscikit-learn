program reference_quantile_seeded_driver
 use sklearn_common,only:dp
 use sklearn_quantile_transform,only:quantile_transformer,quantile_transform
 use sklearn_sparse,only:csr_matrix
 use,intrinsic::iso_fortran_env,only:int64
 implicit none
 type(quantile_transformer)::model
 type(csr_matrix)::x,q,probe,z,original_x,original_q,original_probe
 real(dp),allocatable::dense(:,:),dense_q(:,:),dense_probe(:,:),a(:,:)
 integer,allocatable::before(:),after(:)
 integer::c,cases,fit_kind,query_kind,nq,subsample,seed,distribution,ignore,axis,qdist,qignore,mode,ns
 call random_seed(size=ns);allocate(before(ns),after(ns));read(*,*) cases
 do c=1,cases
  read(*,*) fit_kind,query_kind,nq,subsample,seed,distribution,ignore,axis,qdist,qignore,mode
  call read_csr(x);call read_csr(q);call read_csr(probe)
  if(mode==3) x%indices(1)=0
  if(mode==4) x%indptr(1)=0
  original_x=x;original_q=q;original_probe=probe;call random_seed(get=before)
  model=quantile_transformer();call restore()
  if(mode==5) model%output_distribution='invalid'
  if(mode==6) z=quantile_transform(x,output_distribution='uniform         invalid')
  if(mode==2) z=model%transform(q)
  print *,'BEGIN',c
  if(fit_kind==0) then
   dense=x%to_dense();a=model%fit_transform(dense);call dense_output('train',a)
  else
   z=model%fit_transform(x);call sparse_output('train',z)
  end if
  call emit('quantiles',reshape(model%quantiles_,[size(model%quantiles_)]))
  call emit('references',model%references_);call emit('metadata',real([model%n_quantiles_,model%n_features_in_],dp))
  model%ignore_implicit_zeros=qignore/=0
  model%output_distribution='uniform';if(qdist/=0) model%output_distribution='normal'
  if(mode==1) then
   model%n_quantiles=0;model%subsample=-1;model%random_state=-1
  end if
  if(query_kind==0) then
   dense_q=q%to_dense();dense_probe=probe%to_dense()
   a=model%transform(dense_q);call dense_output('query',a)
   a=model%inverse_transform(dense_probe);call dense_output('inverse',a)
  else
   z=model%transform(q);call sparse_output('query',z)
   z=model%inverse_transform(probe);call sparse_output('inverse',z)
  end if
  call restore()
  if(fit_kind==0) then
   a=quantile_transform(dense,axis,nq,model%output_distribution,ignore/=0,subsample,seed)
   call dense_output('free',a)
  else
   z=quantile_transform(x,axis,nq,model%output_distribution,ignore/=0,subsample,seed)
   call sparse_output('free',z)
  end if
  call random_seed(get=after)
  call emit('preserved',real([merge(1,0,equal_bits(x,original_x)),merge(1,0,equal_bits(q,original_q)), &
   merge(1,0,equal_bits(probe,original_probe)),merge(1,0,all(before==after))],dp))
  dense_q=transpose(q%to_dense());call model%fit(dense_q)
  call emit('refit_quantiles',reshape(model%quantiles_,[size(model%quantiles_)]))
  call emit('refit_references',model%references_)
  call emit('refit_metadata',real([model%n_quantiles_,model%n_features_in_],dp))
  print *,'END',c
 end do
contains
 subroutine restore()
  model%n_quantiles=nq;model%subsample=subsample;model%random_state=seed;model%ignore_implicit_zeros=ignore/=0
  model%output_distribution='uniform';if(distribution/=0) model%output_distribution='normal'
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
