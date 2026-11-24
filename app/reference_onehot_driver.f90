program reference_onehot_driver
 use sklearn_common,only:dp
 use sklearn_one_hot_encoding,only:one_hot_encoder,onehot_inverse_result,onehot_category,onehot_name_value
 use sklearn_sparse,only:csr_matrix
 use sklearn_feature_extraction,only:feature_string
 implicit none
 type(one_hot_encoder)::model
 type(onehot_inverse_result)::decoded
 type(csr_matrix)::s
 type(feature_string),allocatable::names(:),out(:)
 real(dp),allocatable::x(:,:),q(:,:),a(:,:),z(:,:)
 integer::n,p,m,mincount,maxcats,sparse,named,explicit,drop_explicit,inv,ir,ic,callback,refit,i,j,k,pos,cn
 real(dp)::minfraction
 character(24)::unknown,drop,live
 character(32)::text
 read(*,*) n,p,m,mincount,minfraction,maxcats,sparse,named,explicit,drop_explicit,inv,ir,ic,callback,refit,unknown,drop,live
 allocate(x(n,p),q(m,p),a(ir,ic))
 do i=1,n
  read(*,*) x(i,:)
 end do
 do i=1,m
  read(*,*) q(i,:)
 end do
 if(explicit==1) then
  allocate(model%categories(p))
  do i=1,p
   read(*,*) cn;allocate(model%categories(i)%values(cn))
   if(cn>0) read(*,*) model%categories(i)%values
  end do
 end if
 if(drop_explicit==1) then
  allocate(model%drop_values(p));read(*,*) model%drop_values
 end if
 do i=1,ir
  read(*,*) a(i,:)
 end do
 if(named==1) then
  allocate(names(p))
  do i=1,p
   write(text,'(i0)') i-1;names(i)%text='feature_'//trim(text)
  end do
 end if
 model%min_frequency_count=mincount;model%min_frequency_fraction=minfraction;model%max_categories=maxcats
 model%drop=drop;model%handle_unknown=unknown
 if(callback==1) model%feature_name_combiner=>name_callback
 call model%fit(x,names);call state('fit')
 call mapped('train',x)
 if(refit==1) then
  call model%fit(x+.5_dp);call state('refit')
 end if
 model%handle_unknown=live;call mapped('query',q)
 if(inv==1) then
  if(sparse==1) then
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
   decoded=model%inverse_transform(s)
  else
   decoded=model%inverse_transform(a)
  end if
  call emit('inverse_shape',real(shape(decoded%values),dp))
  call emit('inverse_values',reshape(decoded%values,[size(decoded%values)]))
  call emit('inverse_none',real(reshape(merge(1,0,decoded%is_none),[size(decoded%is_none)]),dp))
  call emit('inverse_infrequent',real(reshape(merge(1,0,decoded%is_infrequent),[size(decoded%is_infrequent)]),dp))
 end if
contains
 function name_callback(feature,category) result(value)
  character(*),intent(in)::feature
  type(onehot_name_value),intent(in)::category
  type(feature_string)::value
  value%text=feature//'__number'
  if(category%is_string) then
   if(category%text=='infrequent_sklearn') value%text=feature//'__rare'
  end if
 end function
 subroutine emit(name,v)
  character(*),intent(in)::name
  real(dp),intent(in)::v(:)
  write(*,'(a,*(1x,es25.16e3))') name,v
 end subroutine
 subroutine state(prefix)
  character(*),intent(in)::prefix
  character(32)::suffix
  integer::j
  call emit(prefix//'_meta',real([model%n_features_in_,model%n_features_out_],dp))
  call emit(prefix//'_flags',real([merge(1,0,allocated(model%drop_idx_)), &
   merge(1,0,allocated(model%infrequent_categories_)),merge(1,0,allocated(model%feature_names_in_))],dp))
  if(allocated(model%drop_idx_)) call emit(prefix//'_drop',real(model%drop_idx_,dp))
  do j=1,size(model%categories_)
   write(suffix,'(i0)') j
   call emit(prefix//'_cat'//trim(suffix),model%categories_(j)%values)
   if(allocated(model%infrequent_categories_)) then
    if(allocated(model%infrequent_categories_(j)%values)) then
     call emit(prefix//'_rare'//trim(suffix),model%infrequent_categories_(j)%values)
    else
     call emit(prefix//'_rare'//trim(suffix),[real(dp)::])
    end if
   end if
  end do
  out=model%get_feature_names_out();call emit_names(prefix//'_names',out)
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
 subroutine mapped(prefix,values)
  character(*),intent(in)::prefix
  real(dp),intent(in)::values(:,:)
  if(sparse==1) then
   s=model%transform_csr(values);z=s%to_dense()
   call emit(prefix//'_ptr',real(s%indptr,dp));call emit(prefix//'_idx',real(s%indices,dp));call emit(prefix//'_data',s%data)
  else
   z=model%transform(values)
  end if
  call emit(prefix//'_shape',real(shape(z),dp));call emit(prefix//'_dense',reshape(z,[size(z)]))
 end subroutine
end program
