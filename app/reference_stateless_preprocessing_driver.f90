program reference_stateless_preprocessing_driver
  use sklearn_stateless_preprocessing, only: normalizer,binarizer,normalize,binarize
  use sklearn_transformer_protocol, only: dense_transformer
  use sklearn_feature_extraction, only: feature_string
  use sklearn_sparse, only: csr_matrix,csr_from_arrays,csr_from_dense
  use sklearn_common, only: dp,require
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  real(dp), allocatable :: x(:,:),q(:,:),z(:,:),scales(:),before(:,:),qbefore(:,:)
  type(csr_matrix) :: xs,qs,result,xcopy,qcopy
  type(feature_string), allocatable :: names(:),output_names(:)
  class(dense_transformer), allocatable :: model
  integer :: kind,sparse,norm,axis,return_norm,n,p,m,named,i
  real(dp) :: threshold
  character(3), parameter :: norms(3)=[character(3)::'l1','l2','max']
  character(40) :: argument
  character(32) :: text
  call get_command_argument(1,argument)
  if(len_trim(argument)>0) then
    call reject(trim(argument));error stop 'invalid input accepted'
  end if
  read(*,*) kind,sparse,norm,axis,return_norm,n,p,m,named,threshold
  allocate(names(p))
  do i=1,p
    write(text,'(i0)') i-1;names(i)%text='axis '//trim(text)
  end do
  if(sparse==0) then
    allocate(x(n,p),q(m,p));read(*,*) x;read(*,*) q;before=x;qbefore=q
  else
    call read_sparse(n,p,xs);call read_sparse(m,p,qs);xcopy=xs;qcopy=qs
  end if
  if(kind<2) then
    if(sparse==0) then
      if(kind==0) then
        if(return_norm/=0) then
          z=normalize(x,norms(norm+1),axis,scales)
        else
          z=normalize(x,norms(norm+1),axis)
        end if
      else
        z=binarize(x,threshold)
      end if
      call matrix(z)
    else
      if(kind==0) then
        if(return_norm/=0) then
          result=normalize(xs,norms(norm+1),axis,scales)
        else
          result=normalize(xs,norms(norm+1),axis)
        end if
      else
        result=binarize(xs,threshold)
      end if
      call emit_sparse(result)
    end if
    if(allocated(scales)) call matrix(reshape(scales,[size(scales),1]))
  else
    if(kind==2) then
      allocate(normalizer::model)
      select type(model)
      type is(normalizer);model%norm=norms(norm+1)
      end select
    else
      allocate(binarizer::model)
      select type(model)
      type is(binarizer);model%threshold=threshold
      end select
    end if
    if(sparse==0) then
      call matrix(model%map_data(x));call matrix(model%map_data(q))
      if(named/=0) then
        z=model%fit_map_data(x,names)
      else
        z=model%fit_map_data(x)
      end if
      call matrix(z);call matrix(model%map_data(q))
    else
      select type(model)
      type is(normalizer)
        call emit_sparse(model%transform(xs));call emit_sparse(model%transform(qs))
        if(named/=0) then
          result=model%fit_transform(xs,names)
        else
          result=model%fit_transform(xs)
        end if
        call emit_sparse(result);call emit_sparse(model%transform(qs))
      type is(binarizer)
        call emit_sparse(model%transform(xs));call emit_sparse(model%transform(qs))
        if(named/=0) then
          result=model%fit_transform(xs,names)
        else
          result=model%fit_transform(xs)
        end if
        call emit_sparse(result);call emit_sparse(model%transform(qs))
      end select
    end if
    call emit_names(model%feature_names())
    if(named/=0) call emit_names(model%feature_names(names))
    if(sparse==0) then
      call model%fit_data(x)
    else
      select type(model)
      type is(normalizer);call model%fit(xs)
      type is(binarizer);call model%fit(xs)
      end select
    end if
    call emit_names(model%feature_names())
    select type(model)
    type is(normalizer);i=model%n_features_in_
    type is(binarizer);i=model%n_features_in_
    end select
    call matrix(reshape(real([0,i,merge(1,0,model%can_inverse())],dp),[3,1]))
  end if
  if(sparse==0) then
    call require(all(x==before).and.all(q==qbefore),'dense input changed')
  else
    call require(all(xs%data==xcopy%data).and.all(qs%data==qcopy%data),'sparse values changed')
    call require(all(xs%indices==xcopy%indices).and.all(qs%indices==qcopy%indices),'sparse indices changed')
    call require(all(xs%indptr==xcopy%indptr).and.all(qs%indptr==qcopy%indptr),'sparse pointers changed')
  end if
contains
  subroutine matrix(a)
    real(dp), intent(in) :: a(:,:)
    print *,shape(a)
    print '(*(es25.16e3,1x))',a
  end subroutine
  subroutine emit_sparse(a)
    type(csr_matrix), intent(in) :: a
    call a%validate();call matrix(a%to_dense())
    call matrix(reshape(real([a%n_rows,a%n_cols],dp),[2,1]))
    call matrix(reshape(real(a%indptr,dp),[size(a%indptr),1]))
    call matrix(reshape(real(a%indices,dp),[size(a%indices),1]))
    call matrix(reshape(a%data,[size(a%data),1]))
  end subroutine
  subroutine read_sparse(rows,cols,a)
    integer, intent(in) :: rows,cols
    type(csr_matrix), intent(out) :: a
    integer :: count
    read(*,*) count;a%n_rows=rows;a%n_cols=cols
    allocate(a%indptr(rows+1),a%indices(count),a%data(count));read(*,*) a%indptr
    if(count>0) then
      read(*,*) a%indices;read(*,*) a%data
    end if
  end subroutine
  subroutine emit_names(values)
    type(feature_string), intent(in) :: values(:)
    real(dp), allocatable :: codes(:,:)
    integer :: i,j,width
    width=0
    do i=1,size(values)
      width=max(width,len(values(i)%text))
    end do
    allocate(codes(width,size(values)));codes=-1
    do i=1,size(values)
      do j=1,len(values(i)%text)
        codes(j,i)=iachar(values(i)%text(j:j))
      end do
    end do
    call matrix(codes)
  end subroutine
  subroutine reject(name)
    character(*), intent(in) :: name
    type(normalizer) :: unit
    type(binarizer) :: binary
    type(feature_string) :: names(2)
    type(feature_string), allocatable :: output_names(:)
    real(dp) :: data(3,2),bad
    real(dp), allocatable :: z(:,:),scales(:)
    type(csr_matrix) :: s,r
    data=1;names(1)%text='a';names(2)%text='b'
    bad=ieee_value(0._dp,ieee_quiet_nan);s=csr_from_dense(data)
    select case(name)
    case('norm');z=normalize(data,'bad')
    case('axis');z=normalize(data,axis=2)
    case('threshold_nan');z=binarize(data,bad)
    case('threshold_inf');z=binarize(data,ieee_value(0._dp,ieee_positive_inf))
    case('sparse_negative');r=binarize(s,-1._dp)
    case('sparse_l1_norms');r=normalize(s,'l1',norms=scales)
    case('sparse_l2_norms');r=normalize(s,'l2',norms=scales)
    case('empty_rows');z=normalize(data(:0,:))
    case('empty_features');z=binarize(data(:,:0))
    case('dense_nan');data(1,1)=bad;z=normalize(data)
    case('dense_inf');data(1,1)=ieee_value(0._dp,ieee_positive_inf);z=binarize(data)
    case('sparse_nan');s%data(1)=bad;r=normalize(s)
    case('sparse_inf');s%data(1)=ieee_value(0._dp,ieee_positive_inf);r=binarize(s)
    case('csr_pointer');s%indptr(1)=0;r=normalize(s)
    case('csr_index');s%indices(1)=0;r=binarize(s)
    case('csr_empty');s=csr_from_arrays(0,2,[1],[integer::],[real(dp)::]);r=normalize(s)
    case('norm_fit');unit%norm='invalid';call unit%fit(data)
    case('binary_fit');binary%threshold=bad;call binary%fit(data)
    case('norm_query_width');call unit%fit(data);z=unit%transform(data(:,:1))
    case('binary_query_width');call binary%fit(data);z=binary%transform(data(:,:1))
    case('sparse_query_width');call unit%fit(data);s=csr_from_dense(data(:,:1));r=unit%transform(s)
    case('fit_name_count');call unit%fit(data,names(:1))
    case('fit_name_unallocated');deallocate(names(1)%text);call binary%fit(s,names)
    case('unfitted_names');output_names=unit%get_feature_names_out()
    case('unfitted_supplied_names');output_names=unit%get_feature_names_out(names)
    case('get_name_count');call unit%fit(data,names);output_names=unit%get_feature_names_out(names(:1))
    case('get_name_text');call binary%fit(data,names);names(1)%text='c';output_names=binary%get_feature_names_out(names)
    case('get_name_length');call unit%fit(data,names);names(1)%text='aa';output_names=unit%get_feature_names_out(names)
    case('get_name_unallocated')
      call unit%fit(data);deallocate(names(1)%text);output_names=unit%get_feature_names_out(names)
    case default;error stop 'unknown rejection case'
    end select
  end subroutine
end program
