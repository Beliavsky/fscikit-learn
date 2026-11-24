program reference_nystroem_csr_driver
 use sklearn_common,only:dp
 use sklearn_kernel_approximation,only:nystroem
 use sklearn_sparse,only:csr_matrix
 use sklearn_nystroem_svd,only:kernel_svd
 use,intrinsic::iso_fortran_env,only:int64
 use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan,ieee_positive_inf
 implicit none
 type(nystroem)::model
 type(csr_matrix)::x,q,original_x,original_q
 real(dp),allocatable::dense(:,:),dense_q(:,:),g(:,:),z(:,:),query(:,:),first(:,:),u(:,:),v(:,:),s(:)
 integer,allocatable::before(:),after(:)
 integer::c,cases,kind,fit_kind,query_kind,k,seed,degree,mode,change,ns,n,status
 real(dp)::gamma,coef0,query_gamma
 character(16),parameter::kernels(*)=[character(16)::'rbf','linear','poly','polynomial','cosine', &
   'laplacian','sigmoid','callable','callable','chi2','additive_chi2','precomputed']
 call random_seed(size=ns);allocate(before(ns),after(ns))
 read(*,*) cases
 do c=1,cases
  read(*,*) fit_kind,query_kind,kind,k,seed,gamma,coef0,degree,query_gamma,change,mode
  call read_csr(x);call read_csr(q)
  n=min(x%n_rows,max(0,k));allocate(g(n,n));read(*,*) g
  model=nystroem();model%kernel='unknown'
  if(kind>=0.and.kind<size(kernels)) model%kernel=kernels(kind+1)
  if(kind==7) model%kernel_callback=>asymmetric_callback
  if(kind==8) model%kernel_callback=>rbf_callback
  model%n_components=k;model%random_state=seed;model%gamma=gamma;model%coef0=coef0;model%degree=degree
  if(mode==1) query=model%transform(q)
  if(mode==2) x%data(1)=ieee_value(0._dp,ieee_quiet_nan)
  if(mode==3) x%data(1)=ieee_value(0._dp,ieee_positive_inf)
  if(mode==4) q%data(1)=ieee_value(0._dp,ieee_quiet_nan)
  if(mode==5) q%data(1)=ieee_value(0._dp,ieee_positive_inf)
  if(mode==6) x%indices(1)=0
  if(mode==7) x%indptr(1)=0
  if(mode==8) x%indptr(2)=size(x%data)+2
  original_x=x;original_q=q;call random_seed(get=before)
  if(fit_kind==0) then
   dense=x%to_dense();z=model%fit_transform(dense)
  else
   z=model%fit_transform(x)
  end if
  model%gamma=query_gamma
  if(change==1) model%kernel='laplacian'
  if(query_kind==0) then
   dense_q=q%to_dense();query=model%transform(dense_q)
  else
   query=model%transform(q)
  end if
  call random_seed(get=after)
  print *,'BEGIN',c
  call emit('indices',real(model%component_indices_,dp))
  if(allocated(model%components_csr_)) then
   call emit('component_shape',real([model%components_csr_%n_rows,model%components_csr_%n_cols],dp))
   call emit('component_ptr',real(model%components_csr_%indptr,dp))
   call emit('component_idx',real(model%components_csr_%indices,dp));call emit('component_data',model%components_csr_%data)
  else
   call emit('component_shape',real(shape(model%components_),dp));call emit('component_ptr',[real(dp)::])
   call emit('component_idx',[real(dp)::]);call emit('component_data',reshape(model%components_,[size(model%components_)]))
  end if
  call emit('normalization',reshape(model%normalization_,[size(model%normalization_)]))
  call emit('norm_gram',reshape(matmul(transpose(model%normalization_),model%normalization_),[size(model%normalization_)]))
  call emit('train',reshape(z,[size(z)]));call emit('query',reshape(query,[size(query)]))
  call emit('train_gram',reshape(matmul(z,transpose(z)),[size(z,1)**2]))
  call emit('query_gram',reshape(matmul(query,transpose(query)),[size(query,1)**2]))
  call emit('cross_gram',reshape(matmul(z,transpose(query)),[size(z,1)*size(query,1)]))
  call emit('metadata',real([model%n_features_in_,size(model%component_indices_), &
    merge(1,0,allocated(model%components_csr_)),merge(1,0,allocated(model%components_))],dp))
  call emit('preserved',real([merge(1,0,equal_csr(x,original_x)),merge(1,0,equal_csr(q,original_q)), &
    merge(1,0,all(before==after))],dp))
  first=model%normalization_;model%kernel=kernels(kind+1);model%gamma=gamma
  if(fit_kind==0) then
   call model%fit(dense)
  else
   call model%fit(x)
  end if
  call emit('repeat',real([merge(1,0,all(first==model%normalization_))],dp))
  allocate(u(n,n),v(n,n),s(n));call kernel_svd(g,u,s,v,status,240)
  call emit('svd_status',[real(status,dp)]);call emit('singular',s)
  call emit('svd_u',reshape(u,[size(u)]));call emit('svd_v',reshape(v,[size(v)]))
  print *,'END',c
  deallocate(g,u,v,s)
 end do
contains
 function asymmetric_callback(a,b) result(value)
  real(dp),intent(in)::a(:),b(:)
  real(dp)::value
  value=dot_product(a,b)+.2_dp*a(1)-.1_dp*b(1)+1
 end function
 function rbf_callback(a,b) result(value)
  real(dp),intent(in)::a(:),b(:)
  real(dp)::value
  value=exp(-.3_dp*sum((a-b)**2))
 end function
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
 subroutine emit(name,a)
  character(*),intent(in)::name
  real(dp),intent(in)::a(:)
  print *,name,size(a)
  write(*,'(*(es25.16e3,1x))') a
 end subroutine
end program
