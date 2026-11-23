program reference_isomap_graphs_driver
  use sklearn_common, only: dp,require
  use sklearn_feature_extraction, only: feature_string
  use sklearn_manifold, only: isomap
  use sklearn_sparse, only: csr_matrix,csr_from_dense
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(isomap) :: model
  type(feature_string), allocatable :: names(:)
  integer(int64), allocatable :: name_codes(:,:)
  integer :: name_index,letter
  type(csr_matrix) :: train,query,original
  real(dp), allocatable :: x(:,:),q(:,:),z(:,:),projected(:,:),again(:,:),before(:,:),qbefore(:,:)
  integer :: kind,query_kind,n,p,m,width,method,metric,mutate
  character(40) :: arg
  call get_command_argument(1,arg)
  if(len_trim(arg)>0) then
    call reject(trim(arg));error stop 'invalid Isomap input accepted'
  end if
  read(*,*) kind,query_kind,n,p,m,width,model%n_neighbors,model%n_components,model%radius,method,metric,model%p,mutate
  model%path_method='auto';if(method==1) model%path_method='FW'
  if(method==2) model%path_method='D'
  select case(metric)
  case(0);model%metric='euclidean'
  case(1);model%metric='manhattan'
  case(2);model%metric='minkowski'
  case(3);model%metric='chebyshev'
  case(4);model%metric='precomputed'
  end select
  if(kind==2) then
    call read_sparse(train);original=train;z=model%fit_transform(train)
  else
    allocate(x(n,p));read(*,*) x;before=x;z=model%fit_transform(x)
  end if
  if(query_kind==1) then
    call read_sparse(query)
  else
    allocate(q(m,width));read(*,*) q;qbefore=q
  end if
  if(mutate/=0) then
    if(model%n_neighbors>0) model%n_neighbors=1
    model%radius=0;model%metric='cosine';model%p=99
  end if
  if(query_kind==1) then
    projected=model%transform(query)
  else
    projected=model%transform(q)
  end if
  if(kind==2) then
    again=model%transform(train)
    call require(all(train%data==original%data).and.all(train%indices==original%indices).and. &
      all(train%indptr==original%indptr),'Isomap CSR input changed')
  else
    again=model%transform(x);call require(all(x==before),'Isomap input changed')
  end if
  if(query_kind==0) call require(all(q==qbefore),'Isomap query changed')
  call integers(reshape(int([model%n_features_in_,model%n_samples_fit_,model%n_features_out_, &
    model%n_connected_components_],int64),[4,1]))
  call matrix(model%dist_matrix_);call matrix(z);call matrix(projected);call matrix(again)
  call matrix(reshape(model%kernel_pca_%eigenvalues_,[size(z,2),1]))
  call matrix(model%kernel_pca_%eigenvectors_)
  call matrix(reshape(model%kernel_pca_%k_fit_rows_,[n,1]))
  call matrix(reshape([model%reconstruction_error()],[1,1]));call integers(reshape([1_int64],[1,1]))
  names=model%get_feature_names_out();allocate(name_codes(size(names),48));name_codes=iachar(' ')
  do name_index=1,size(names)
    do letter=1,len(names(name_index)%text)
      name_codes(name_index,letter)=iachar(names(name_index)%text(letter:letter))
    end do
  end do
  call integers(name_codes)
contains
  subroutine read_sparse(a)
    type(csr_matrix), intent(out) :: a
    integer :: stored
    read(*,*) a%n_rows,a%n_cols,stored
    allocate(a%indptr(a%n_rows+1),a%indices(stored),a%data(stored))
    read(*,*) a%indptr;read(*,*) a%indices;read(*,*) a%data
  end subroutine
  subroutine matrix(a)
    real(dp), intent(in) :: a(:,:)
    print *, 'F',shape(a,kind=int64)
    print '(8(es25.16e3,1x))',a
  end subroutine
  subroutine integers(a)
    integer(int64), intent(in) :: a(:,:)
    print *, 'I',shape(a,kind=int64)
    print '(20(i0,1x))',a
  end subroutine
  subroutine reject(name)
    character(*), intent(in) :: name
    real(dp) :: small(4,1),bad(4,4),value
    real(dp), allocatable :: result(:,:)
    type(csr_matrix) :: a
    small(:,1)=[0._dp,1._dp,3._dp,6._dp];model%n_neighbors=2
    bad=0
    select case(name)
    case('neighbors_negative');model%n_neighbors=-1
    case('neighbors_large');model%n_neighbors=4
    case('both');model%radius=1
    case('neither');model%n_neighbors=0
    case('radius_negative');model%radius=-2
    case('radius_inf');model%n_neighbors=0;model%radius=ieee_value(0._dp,ieee_positive_inf)
    case('radius_nan');model%radius=ieee_value(0._dp,ieee_quiet_nan)
    case('components');model%n_components=0
    case('path');model%path_method='J'
    case('p_small');model%p=.5_dp
    case('p_inf');model%p=ieee_value(0._dp,ieee_positive_inf)
    case('p_nan');model%p=ieee_value(0._dp,ieee_quiet_nan)
    case('pre_shape');model%metric='precomputed'
    case('pre_negative');model%metric='precomputed';bad(1,2)=-1;call model%fit(bad);return
    case('pre_nan');model%metric='precomputed';bad(1,2)=ieee_value(0._dp,ieee_quiet_nan)
      call model%fit(bad);return
    case('csr_metric');a=csr_from_dense(bad,.true.);call model%fit(a);return
    case('csr_shape');model%metric='precomputed';a=csr_from_dense(small,.true.);call model%fit(a);return
    case('csr_neighbors');model%metric='precomputed';a=csr_from_dense(bad);call model%fit(a);return
    case('csr_disconnected');model%metric='precomputed';model%n_neighbors=0;model%radius=0
      bad(1,1)=1;bad(2,2)=1;bad(3,3)=1;bad(4,4)=1;a=csr_from_dense(bad);call model%fit(a);return
    case('unfitted_names');names=model%get_feature_names_out();return
    case('names_length');call model%fit(small);names=model%get_feature_names_out(['a','b']);return
    case('unfitted');result=model%transform(small);return
    case('unfitted_csr');a=csr_from_dense(bad,.true.);result=model%transform(a);return
    case('unfitted_error');value=model%reconstruction_error();return
    case('query_width');call model%fit(small);result=model%transform(bad);return
    case('query_negative');model%metric='precomputed';call model%fit(bad);bad(1,2)=-1
      result=model%transform(bad);return
    case('query_empty_radius');model%n_neighbors=0;model%radius=1;call model%fit(small)
      result=model%transform(small+100);return
    case('query_csr_metric');call model%fit(small);a=csr_from_dense(bad,.true.);result=model%transform(a);return
    case('query_csr_width');model%metric='precomputed';call model%fit(bad)
      a=csr_from_dense(small,.true.);result=model%transform(a);return
    case('query_csr_neighbors');model%metric='precomputed';call model%fit(bad)
      a=csr_from_dense(bad);result=model%transform(a);return
    case('query_inf');call model%fit(small);small(1,1)=ieee_value(0._dp,ieee_positive_inf)
      result=model%transform(small);return
    case default;error stop 'unknown Isomap rejection case'
    end select
    call model%fit(small)
  end subroutine
end program
