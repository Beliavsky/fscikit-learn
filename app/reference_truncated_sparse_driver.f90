program reference_truncated_sparse_driver
  use sklearn_common, only: dp
  use sklearn_decomposition, only: truncated_svd
  use sklearn_sparse, only: csr_matrix,csr_from_arrays,csr_from_dense
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  implicit none
  type(truncated_svd) :: model
  type(csr_matrix) :: sx,sq
  real(dp), allocatable :: x(:,:),q(:,:),z(:,:),values(:),before(:)
  integer, allocatable :: pointer(:),indices(:),saved_pointer(:),saved_indices(:)
  integer :: kind,n,p,m,k,algorithm,power,extra,normalizer,seed,nnz
  real(dp) :: bad
  character(8), parameter :: norms(4)=[character(8)::'auto','none','LU','QR']
  character(80) :: argument
  if(command_argument_count()>0) then
    call get_command_argument(1,argument);call reject(trim(argument));stop 2
  end if
  read(*,*) kind,n,p,m,k,algorithm,power,extra,normalizer,seed
  model%n_components=k;model%n_iter=power;model%n_oversamples=extra;model%random_state=seed
  model%power_iteration_normalizer=norms(normalizer+1)
  if(algorithm==1) model%algorithm='randomized'
  if(kind==0) then
    allocate(x(n,p),q(m,p));read(*,*) x;read(*,*) q;z=model%fit_transform(x)
  else
    read(*,*) nnz;allocate(pointer(n+1),indices(nnz),values(nnz))
    read(*,*) pointer;read(*,*) indices;read(*,*) values;sx=csr_from_arrays(n,p,pointer,indices,values)
    deallocate(pointer,indices,values)
    read(*,*) nnz;allocate(pointer(m+1),indices(nnz),values(nnz))
    read(*,*) pointer;read(*,*) indices;read(*,*) values;sq=csr_from_arrays(m,p,pointer,indices,values)
    before=sx%data;saved_pointer=sx%indptr;saved_indices=sx%indices;z=model%fit_transform(sx)
  end if
  call matrix(model%components_);call vector(model%singular_values_)
  call vector(model%explained_variance_);call vector(model%explained_variance_ratio_)
  call vector(real([model%n_components_,model%n_features_in_],dp));call matrix(z)
  if(kind==0) then
    call matrix(model%transform(x));z=model%transform(q)
  else
    call matrix(model%transform(sx));z=model%transform(sq)
    if(any(before/=sx%data).or.any(saved_pointer/=sx%indptr).or.any(saved_indices/=sx%indices)) stop 3
  end if
  call matrix(z);call matrix(model%inverse_transform(z));call vector([1._dp])
contains
  subroutine vector(a)
    real(dp), intent(in) :: a(:)
    print *,size(a);print '(*(es26.17e3,1x))',a
  end subroutine
  subroutine matrix(a)
    real(dp), intent(in) :: a(:,:)
    print *,shape(a);print '(*(es26.17e3,1x))',reshape(a,[size(a)])
  end subroutine
  subroutine reject(name)
    character(*), intent(in) :: name
    real(dp) :: data(4,3)
    data=reshape([1._dp,2._dp,0._dp,1._dp,0._dp,1._dp,2._dp,0._dp,1._dp,0._dp,1._dp,2._dp],[4,3])
    sx=csr_from_dense(data);model%algorithm='randomized';bad=ieee_value(0._dp,ieee_quiet_nan)
    select case(name)
    case('algorithm');model%algorithm='unknown'
    case('power');model%n_iter=-1
    case('oversampling');model%n_oversamples=0
    case('seed');model%random_state=-1
    case('normalizer');model%power_iteration_normalizer='unknown'
    case('rank');model%n_components=0
    case('rank_large');model%n_components=4
    case('capacity');model%n_oversamples=huge(1)
    case('single_feature');call model%fit(data(:,:1));return
    case('dense_nan');data(1,1)=bad;call model%fit(data);return
    case('csr_full');model%algorithm='full'
    case('csr_nan');sx%data(1)=bad
    case('csr_empty');sx=csr_from_arrays(0,3,[1],[integer::],[real(dp)::])
    case('csr_pointer');sx%indptr(1)=2
    case('csr_column');sx%indices(1)=4
    case('csr_unfitted');z=model%transform(sx);return
    case('csr_features');call model%fit(sx);sq=csr_from_dense(data(:,:2));z=model%transform(sq);return
    case('csr_query_nan');call model%fit(sx);sx%data(1)=bad;z=model%transform(sx);return
    case('inverse_features');call model%fit(sx);z=model%inverse_transform(data);return
    case default;stop 2
    end select
    call model%fit(sx)
  end subroutine
end program
