program reference_kbins_driver
 use sklearn_common,only:dp
 use sklearn_discretization,only:kbins_discretizer
 use sklearn_legacy_rng,only:legacy_random_state
 use sklearn_sparse,only:csr_matrix
 use sklearn_feature_extraction,only:feature_string
 use,intrinsic::iso_fortran_env,only:int64
 implicit none
 type(kbins_discretizer)::model
 type(legacy_random_state)::rng
 type(csr_matrix)::s
 type(feature_string),allocatable::names(:),out(:)
 real(dp),allocatable::x(:,:),q(:,:),w(:),z(:,:),a(:,:),back(:,:)
 integer,allocatable::counts(:)
 integer::n,p,m,subsample,weighted,wn,sparse,named,external,refit,ir,ic,i,j,k,pos
 integer(int64)::seed
 character(16)::strategy,encode,live,refit_encode
 character(32)::text
 read(*,*) n,p,m,subsample,weighted,wn,sparse,named,external,seed,refit,ir,ic,strategy,encode,live,refit_encode
 allocate(x(n,p),q(m,p),counts(p),w(wn),a(ir,ic))
 read(*,*) counts
 do i=1,n
  read(*,*) x(i,:)
 end do
 do i=1,m
  read(*,*) q(i,:)
 end do
 if(wn>0) read(*,*) w
 do i=1,ir
  read(*,*) a(i,:)
 end do
 if(named==1) then
  allocate(names(p))
  do i=1,p
   write(text,'(i0)') i-1;names(i)%text='feature_'//trim(text)
  end do
 end if
 model%strategy=strategy;model%encode=encode;model%subsample=subsample;model%random_state=seed
 call seed_rng()
 if(weighted==1) then
  if(external==1) then
   call model%fit(x,counts,w,names,rng)
  else
   call model%fit(x,counts,w,names)
  end if
 else
  if(external==1) then
   call model%fit(x,counts,feature_names=names,rng=rng)
  else
   call model%fit(x,counts,feature_names=names)
  end if
 end if
 call emit_state('fit')
 call mapped('train',x,.true.)
 if(refit>0) then
  model%encode=refit_encode;model%n_bins=refit;model%subsample=0
  call model%fit(x+.2_dp,feature_names=names)
  call emit_state('refit')
 end if
 model%encode=live
 call mapped('query',q,.false.)
 if(ir>0) then
  if(sparse==1) then
   ! Raw inverse CSR has duplicates, descending columns and explicit zeros.
   s%n_rows=ir;s%n_cols=ic;s%indptr=[(1+(i-1)*2*ic,i=1,ir+1)]
   if(allocated(s%indices)) deallocate(s%indices,s%data)
   allocate(s%indices(2*ir*ic),s%data(2*ir*ic));pos=1
   do i=1,ir
    do j=ic,1,-1
     do k=1,2
      s%indices(pos)=j;s%data(pos)=a(i,j)*merge(.25_dp,.75_dp,k==1);pos=pos+1
     end do
    end do
   end do
   back=model%inverse_transform(s)
  else
   back=model%inverse_transform(a)
  end if
  call emit('probe_inverse',reshape(back,[size(back)]))
 end if
 if(external==1) call emit('rng',real([(rng%raw_uint32(),i=1,8)],dp))
contains
 subroutine seed_rng()
  use sklearn_legacy_rng,only:seed_legacy_uint32
  call seed_legacy_uint32(rng,seed)
 end subroutine
 subroutine emit(name,v)
  character(*),intent(in)::name
  real(dp),intent(in)::v(:)
  write(*,'(a,*(1x,es25.16e3))') name,v
 end subroutine
 subroutine emit_names(prefix,v)
  character(*),intent(in)::prefix
  type(feature_string),intent(in)::v(:)
  integer::j,k
  call emit(prefix//'_length',real([(len(v(j)%text),j=1,size(v))],dp))
  write(*,'(a)',advance='no') prefix//'_codes'
  do j=1,size(v)
   do k=1,len(v(j)%text)
    write(*,'(1x,i0)',advance='no') iachar(v(j)%text(k:k))
   end do
  end do
  write(*,*)
 end subroutine
 subroutine emit_state(prefix)
  character(*),intent(in)::prefix
  character(32)::num
  integer::j
  call emit(prefix//'_counts',real(model%n_bins_,dp))
  call emit(prefix//'_meta',real([model%n_features_in_,model%n_features_out_],dp))
  do j=1,size(model%n_bins_)
   write(num,'(i0)') j;call emit(prefix//'_edges'//trim(num),model%bin_edges_(j)%values)
  end do
  out=model%get_feature_names_out();call emit_names(prefix//'_names',out)
 end subroutine
 subroutine mapped(prefix,data,inverse)
  character(*),intent(in)::prefix
  real(dp),intent(in)::data(:,:)
  logical,intent(in)::inverse
  if(sparse==1.and.model%encode/='ordinal') then
   s=model%transform_csr(data);z=s%to_dense()
   call emit(prefix//'_ptr',real(s%indptr,dp));call emit(prefix//'_idx',real(s%indices,dp))
   call emit(prefix//'_data',s%data)
   if(inverse) back=model%inverse_transform(s)
  else
   z=model%transform(data)
   if(inverse) back=model%inverse_transform(z)
  end if
  call emit(prefix//'_shape',real(shape(z),dp));call emit(prefix//'_dense',reshape(z,[size(z)]))
  if(inverse) call emit(prefix//'_inverse',reshape(back,[size(back)]))
 end subroutine
end program
