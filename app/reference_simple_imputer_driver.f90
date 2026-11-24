program reference_simple_imputer_driver
 use sklearn_common,only:dp
 use sklearn_simple_impute,only:simple_imputer
 use sklearn_sparse,only:csr_matrix
 use sklearn_feature_extraction,only:feature_string
 use,intrinsic::iso_fortran_env,only:int64
 implicit none
 type(simple_imputer)::model
 type(csr_matrix)::x,q,probe,z,original_x,original_q,original_probe
 type(feature_string),allocatable::names(:),output_names(:)
 real(dp),allocatable::dense(:,:),dense_q(:,:),a(:,:),inverse_input(:,:)
 integer,allocatable::before(:),after(:)
 integer::c,cases,fit_kind,query_kind,strategy,keep,add,named,qkeep,qadd,qstrategy,mode,ns,j,calls
 real(dp)::marker,fill,qmarker
 call random_seed(size=ns);allocate(before(ns),after(ns));read(*,*) cases
 do c=1,cases
  read(*,*) fit_kind,query_kind,strategy,keep,add,named,qkeep,qadd,qstrategy,mode,marker,fill,qmarker
  call read_csr(x);call read_csr(q);call read_csr(probe)
  if(mode==3) x%indices(1)=0
  if(mode==4) x%indptr(1)=0
  if(allocated(names)) deallocate(names)
  allocate(names(x%n_cols))
  do j=1,size(names)
   names(j)%text='f'//integer_text(j-1)
  end do
  original_x=x;original_q=q;original_probe=probe;call random_seed(get=before);calls=0
  model=simple_imputer();call restore()
  if(mode==1) model%strategy='invalid'
  if(mode==7) nullify(model%statistic)
  if(mode==2) a=model%transform(q%to_dense())
  print *,'BEGIN',c
  if(fit_kind==0) then
   dense=x%to_dense()
   if(named/=0) then
    a=model%fit_transform(dense,names)
   else
    a=model%fit_transform(dense)
   end if
   call dense_output('train',a)
  else
   if(named/=0) then
    z=model%fit_transform(x,names)
   else
    z=model%fit_transform(x)
   end if
   call sparse_output('train',z)
  end if
  if(mode==6) a=model%inverse_transform(probe%to_dense())
  call emit('statistics',model%statistics_);call metadata('fit')
  call names_output('names_default',model%get_feature_names_out())
  call names_output('names_explicit',model%get_feature_names_out(names))
  model%keep_empty_features=qkeep/=0;model%add_indicator=qadd/=0;model%strategy=strategy_name(qstrategy)
  model%missing_values=qmarker
  if(mode==8) names(1)%text='other'
  if(query_kind==0) then
   dense_q=q%to_dense()
   if(named/=0) then
    a=model%transform(dense_q,names)
   else
    a=model%transform(dense_q)
   end if
   call dense_output('query',a);inverse_input=a
  else
   z=model%transform(q);call sparse_output('query',z);inverse_input=z%to_dense()
  end if
  if(qadd/=0) then
   if(mode==5) inverse_input=probe%to_dense()
   a=model%inverse_transform(inverse_input)
   call emit('inverse_shape',real(shape(a),dp));call emit('inverse_data',reshape(a,[size(a)]))
  else
   call emit('inverse_shape',[0._dp,0._dp]);call emit('inverse_data',[real(dp)::])
  end if
  call random_seed(get=after)
  call emit('preserved',real([merge(1,0,equal_bits(x,original_x)),merge(1,0,equal_bits(q,original_q)), &
   merge(1,0,equal_bits(probe,original_probe)),merge(1,0,all(before==after))],dp))
  call restore();calls=0;dense_q=transpose(q%to_dense());call model%fit(dense_q)
  call emit('refit_statistics',model%statistics_);call metadata('refit')
  call names_output('refit_names',model%get_feature_names_out())
  print *,'END',c
 end do
contains
 function strategy_name(code) result(name)
  integer,intent(in)::code
  character(16)::name
  select case(code)
  case(0);name='mean'
  case(1);name='median'
  case(2);name='most_frequent'
  case(3);name='constant'
  case(4);name='callable'
  case default;name='invalid'
  end select
 end function
 subroutine restore()
  model%strategy=strategy_name(strategy);model%keep_empty_features=keep/=0;model%add_indicator=add/=0
  model%missing_values=marker;model%fill_value=fill
  if(strategy==4) model%statistic=>count_values
 end subroutine
 function count_values(v) result(value)
  real(dp),intent(in)::v(:)
  real(dp)::value
  calls=calls+1;value=real(size(v),dp)+2
 end function
 subroutine metadata(prefix)
  character(*),intent(in)::prefix
  integer::present_indicator,nindicator,names_count
  present_indicator=0;nindicator=0;names_count=0
  if(allocated(model%indicator_%features_)) then
   present_indicator=1;nindicator=size(model%indicator_%features_)
   call emit(prefix//'_indicator',real(model%indicator_%features_,dp))
  else
   call emit(prefix//'_indicator',[real(dp)::])
  end if
  if(allocated(model%feature_names_in_)) names_count=size(model%feature_names_in_)
  call emit(prefix//'_metadata',real([model%n_features_in_,present_indicator,nindicator,names_count,calls],dp))
 end subroutine
 subroutine names_output(prefix,names)
  character(*),intent(in)::prefix
  type(feature_string),intent(in)::names(:)
  integer::i,j,n,k
  real(dp),allocatable::lengths(:),codes(:)
  allocate(lengths(size(names)));n=0
  do i=1,size(names)
   lengths(i)=len(names(i)%text);n=n+len(names(i)%text)
  end do
  allocate(codes(n));k=0
  do i=1,size(names)
   do j=1,len(names(i)%text)
    k=k+1;codes(k)=iachar(names(i)%text(j:j))
   end do
  end do
  call emit(prefix//'_lengths',lengths);call emit(prefix//'_codes',codes)
 end subroutine
 function integer_text(i) result(text)
  integer,intent(in)::i
  character(:),allocatable::text
  character(48)::buffer
  write(buffer,'(i0)') i;text=trim(buffer)
 end function
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
