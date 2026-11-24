program reference_polynomial_csr_driver
 use sklearn_common,only:dp
 use sklearn_polynomial_count_sketch,only:polynomial_count_sketch
 use sklearn_sparse,only:csr_matrix
 use,intrinsic::iso_fortran_env,only:int64
 use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan,ieee_positive_inf
 use,intrinsic::ieee_exceptions,only:ieee_get_flag,ieee_get_halting_mode, &
   ieee_invalid,ieee_divide_by_zero,ieee_overflow
 implicit none
 type(polynomial_count_sketch)::model
 type(csr_matrix)::x,q,original_x,original_q
 real(dp),allocatable::dense(:,:),dense_q(:,:),z(:,:),query(:,:)
 integer,allocatable::before(:),after(:),hash(:,:),bits(:,:)
 integer::c,cases,fit_kind,query_kind,degree,components,seed,qdegree,qcomponents,mode,ns
 real(dp)::gamma,coef0,qgamma,qcoef0
 logical::halt_before(3),halt_after(3),flag_before(3),flag_after(3)
 call random_seed(size=ns);allocate(before(ns),after(ns))
 read(*,*) cases
 do c=1,cases
  read(*,*) fit_kind,query_kind,degree,components,seed,gamma,coef0,qdegree,qcomponents,qgamma,qcoef0,mode
  call read_csr(x);call read_csr(q)
  model=polynomial_count_sketch()
  model%degree=degree;model%n_components=components;model%random_state=seed;model%gamma=gamma;model%coef0=coef0
  if(mode==1) query=model%transform(q)
  if(mode==2) x%data(1)=ieee_value(0._dp,ieee_quiet_nan)
  if(mode==3) x%data(1)=ieee_value(0._dp,ieee_positive_inf)
  if(mode==4) q%data(1)=ieee_value(0._dp,ieee_quiet_nan)
  if(mode==5) q%data(1)=ieee_value(0._dp,ieee_positive_inf)
  if(mode==6) x%indices(1)=0
  if(mode==7) x%indptr(1)=0
  if(mode==8) x%indptr(2)=size(x%data)+2
  original_x=x;original_q=q;call random_seed(get=before);call environment(halt_before,flag_before)
  if(fit_kind==0) then
   dense=x%to_dense();z=model%fit_transform(dense)
  else
   z=model%fit_transform(x)
  end if
  hash=model%index_hash_;bits=model%bit_hash_
  model%degree=qdegree;model%n_components=qcomponents;model%gamma=qgamma;model%coef0=qcoef0
  if(query_kind==0) then
   dense_q=q%to_dense();query=model%transform(dense_q)
  else
   query=model%transform(q)
  end if
  call random_seed(get=after);call environment(halt_after,flag_after)
  print *,'BEGIN',c
  call emit('hash_shape',real(shape(hash),dp));call emit('hash',real(reshape(hash,[size(hash)]),dp))
  call emit('bits',real(reshape(bits,[size(bits)]),dp))
  call emit('train',reshape(z,[size(z)]));call emit('query',reshape(query,[size(query)]))
  call emit('metadata',real([model%n_features_in_,model%n_features_out_,size(query,1),size(query,2)],dp))
  call emit('preserved',real([merge(1,0,equal_csr(x,original_x)),merge(1,0,equal_csr(q,original_q)), &
    merge(1,0,all(before==after))],dp))
  call emit('environment',real([merge(1,0,halt_before.eqv.halt_after),merge(1,0,flag_before.eqv.flag_after)],dp))
  model%degree=degree;model%n_components=components;model%gamma=gamma;model%coef0=coef0
  if(fit_kind==0) then
   call model%fit(x)
  else
   dense=x%to_dense();call model%fit(dense)
  end if
  call emit('repeat',real([merge(1,0,all(hash==model%index_hash_)),merge(1,0,all(bits==model%bit_hash_))],dp))
  print *,'END',c
 end do
contains
 subroutine environment(halting,flags)
  logical,intent(out)::halting(3),flags(3)
  call ieee_get_halting_mode(ieee_invalid,halting(1));call ieee_get_halting_mode(ieee_divide_by_zero,halting(2))
  call ieee_get_halting_mode(ieee_overflow,halting(3))
  call ieee_get_flag(ieee_invalid,flags(1));call ieee_get_flag(ieee_divide_by_zero,flags(2))
  call ieee_get_flag(ieee_overflow,flags(3))
 end subroutine
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
