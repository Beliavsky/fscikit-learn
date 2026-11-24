program reference_scale_driver
 use sklearn_common,only:dp
 use sklearn_standard_scaling,only:scale
 use sklearn_sparse,only:csr_matrix
 use,intrinsic::iso_fortran_env,only:int64
 implicit none
 type(csr_matrix)::x,z,original
 real(dp),allocatable::dense(:,:),mapped(:,:),v(:)
 integer,allocatable::before(:),after(:)
 integer::c,cases,kind,axis,center,std,mode,ns
 logical::first,second
 call random_seed(size=ns);allocate(before(ns),after(ns));read(*,*) cases
 do c=1,cases
  read(*,*) kind,axis,center,std,mode;call read_csr(x)
  if(mode==2) x%indices(1)=0
  if(mode==3) x%indptr(1)=0
  original=x;call random_seed(get=before)
  print *,'BEGIN',c
  if(kind==0) then
   dense=x%to_dense()
   if(mode==1) then
    v=scale(reshape(dense,[size(dense)]),axis,center/=0,std/=0,first,second)
    call emit('only_vector',v)
   else
    mapped=scale(dense,axis,center/=0,std/=0,first,second);call dense_output('out',mapped)
   end if
  else
   z=scale(x,axis,center/=0,std/=0,first,second);call sparse_output('out',z)
  end if
  call emit('corrections',real([merge(1,0,first),merge(1,0,second)],dp))
  if(kind==0.and.axis==0) then
   v=scale(reshape(dense,[size(dense)]),axis,center/=0,std/=0,first,second);call emit('vector',v)
   call emit('vector_corrections',real([merge(1,0,first),merge(1,0,second)],dp))
  else
   call emit('vector',[real(dp)::]);call emit('vector_corrections',[real(dp)::])
  end if
  call random_seed(get=after)
  call emit('preserved',real([merge(1,0,equal_bits(x,original)),merge(1,0,all(before==after))],dp))
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
