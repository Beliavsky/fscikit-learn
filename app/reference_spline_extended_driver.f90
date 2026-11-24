program reference_spline_extended_driver
 use sklearn_common,only:dp
 use sklearn_spline,only:spline_transformer
 use sklearn_feature_extraction,only:feature_string
 use sklearn_sparse,only:csr_matrix
 implicit none
 type(spline_transformer)::model
 type(csr_matrix)::s
 type(feature_string),allocatable::names(:),out(:)
 real(dp),allocatable::x(:,:),q(:,:),w(:),knots(:,:),z(:,:)
 integer::n,p,m,nk,d,bias,weighted,explicit,sparse,named,ld,lb,i
 character(16)::mode,extra,live
 character(32)::number
 read(*,*) n,p,m,nk,d,bias,weighted,explicit,sparse,named,mode,extra,ld,lb,live
 allocate(x(n,p),q(m,p),w(n),knots(nk,p))
 do i=1,n
  read(*,*) x(i,:)
 end do
 do i=1,m
  read(*,*) q(i,:)
 end do
 read(*,*) w
 do i=1,nk
  read(*,*) knots(i,:)
 end do
 if(named==1) then
  allocate(names(p))
  do i=1,p
   write(number,'(i0)') i-1;names(i)%text='feature_'//trim(number)
  end do
 end if
 model%n_knots=nk;model%degree=d;model%include_bias=bias==1;model%knots=mode;model%extrapolation=extra
 if(explicit==1) then
  if(weighted==1) then
   call model%fit(x,knots,w,names)
  else
   call model%fit(x,knots=knots,feature_names=names)
  end if
 else
  if(weighted==1) then
   call model%fit(x,sample_weight=w,feature_names=names)
  else
   call model%fit(x,feature_names=names)
  end if
 end if
 call emit('base',reshape(model%base_knots_,[size(model%base_knots_)]))
 call emit('knots',reshape(model%knots_,[size(model%knots_)]))
 call emit('meta',real([model%n_features_in_,model%n_features_out_],dp))
 call mapped('train',x)
 model%degree=ld;model%include_bias=lb==1;model%extrapolation=live
 call mapped('query',q)
 out=model%get_feature_names_out();call emit_names(out)
contains
 subroutine emit(name,v)
  character(*),intent(in)::name
  real(dp),intent(in)::v(:)
  write(*,'(a,*(1x,es25.16e3))') name,v
 end subroutine
 subroutine mapped(name,a)
  character(*),intent(in)::name
  real(dp),intent(in)::a(:,:)
  if(sparse==1) then
   s=model%transform_csr(a);call emit(name//'_shape',real([s%n_rows,s%n_cols],dp))
   call emit(name//'_ptr',real(s%indptr,dp));call emit(name//'_idx',real(s%indices,dp))
   call emit(name//'_data',s%data);z=s%to_dense()
  else
   z=model%transform(a);call emit(name//'_shape',real(shape(z),dp))
  end if
  call emit(name//'_dense',reshape(z,[size(z)]))
 end subroutine
 subroutine emit_names(a)
  type(feature_string),intent(in)::a(:)
  integer::j,k
  call emit('names_length',real([(len(a(j)%text),j=1,size(a))],dp))
  write(*,'(a)',advance='no') 'names_codes'
  do j=1,size(a)
   do k=1,len(a(j)%text)
    write(*,'(1x,i0)',advance='no') iachar(a(j)%text(k:k))
   end do
  end do
  write(*,*)
 end subroutine
end program
