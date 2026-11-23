program reference_polynomial_driver
  use sklearn_polynomial, only: polynomial_features
  use sklearn_common, only: dp,require
  use sklearn_sparse, only: csr_matrix,csr_from_arrays,csr_from_dense
  use sklearn_feature_extraction, only: feature_string
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(polynomial_features) :: model
  type(csr_matrix) :: xs,qs,result,xcopy,qcopy
  type(feature_string), allocatable :: names(:)
  real(dp), allocatable :: x(:,:),q(:,:),z(:,:),before(:,:),qbefore(:,:)
  integer :: mode,sparse,n,p,m,bias,interaction,named,i
  character(40) :: argument
  character(32) :: text
  call get_command_argument(1,argument)
  if(len_trim(argument)>0) then
    call reject(trim(argument));error stop 'invalid polynomial input accepted'
  end if
  read(*,*) mode,sparse,n,p,m,model%min_degree,model%degree,bias,interaction,named
  model%include_bias=bias/=0;model%interaction_only=interaction/=0
  if(mode/=1) then
    allocate(names(p))
    do i=1,p
      write(text,'(i0)') i-1;names(i)%text='axis '//trim(text)
    end do
    if(named==2) then
      names(1)%text=''
      if(p>=2) names(2)%text='heat flux'
      if(p>=3) names(3)%text=achar(206)//achar(178)
    end if
  end if
  if(sparse==0) then
    allocate(x(n,p),q(m,p));read(*,*) x;read(*,*) q;before=x;qbefore=q
    if(named/=0.and.mode/=1) then
      call model%fit(x,names)
    else
      call model%fit(x)
    end if
  else
    call read_sparse(n,p,xs);call read_sparse(m,p,qs);xcopy=xs;qcopy=qs
    if(named/=0.and.mode/=1) then
      call model%fit(xs,names)
    else
      call model%fit(xs)
    end if
  end if
  call integers(reshape([int(model%n_features_in_,int64),model%n_output_features_, &
    model%min_degree_,model%max_degree_],[4,1]))
  if(mode==0) then
    if(sparse==0) then
      call matrix(model%transform(x));call matrix(model%transform(q))
      if(named/=0) then
        z=model%fit_transform(x,names)
      else
        z=model%fit_transform(x)
      end if
      call matrix(z)
      call matrix(polynomial_features(x,int(model%degree),model%include_bias,model%interaction_only,int(model%min_degree)))
    else
      call emit_sparse(model%transform(xs));call emit_sparse(model%transform(qs))
      if(named/=0) then
        result=model%fit_transform(xs,names)
      else
        result=model%fit_transform(xs)
      end if
      call emit_sparse(result)
    end if
  end if
  if(mode/=1) then
    call integers(model%get_powers());call integers(model%powers_())
    call emit_names(model%get_feature_names_out());call emit_names(model%get_feature_names_out(names))
    if(sparse==0) then
      call model%fit(x)
    else
      call model%fit(xs)
    end if
    call emit_names(model%get_feature_names_out())
  end if
  if(sparse==0) then
    call require(all(x==before).and.all(q==qbefore),'polynomial dense input changed')
  else
    call require(all(xs%data==xcopy%data).and.all(qs%data==qcopy%data),'polynomial sparse data changed')
    call require(all(xs%indices==xcopy%indices).and.all(qs%indices==qcopy%indices),'polynomial sparse indices changed')
    call require(all(xs%indptr==xcopy%indptr).and.all(qs%indptr==qcopy%indptr),'polynomial sparse pointers changed')
  end if
contains
  subroutine matrix(a)
    real(dp), intent(in) :: a(:,:)
    print *, 'F',shape(a,kind=int64)
    print '(*(es25.16e3,1x))',a
  end subroutine
  subroutine integers(a)
    integer(int64), intent(in) :: a(:,:)
    print *, 'I',shape(a,kind=int64)
    print '(*(i0,1x))',a
  end subroutine
  subroutine emit_sparse(a)
    type(csr_matrix), intent(in) :: a
    call a%validate();call matrix(a%to_dense())
    call integers(reshape(int([a%n_rows,a%n_cols],int64),[2,1]))
    call integers(reshape(int(a%indptr,int64),[size(a%indptr),1]))
    call integers(reshape(int(a%indices,int64),[size(a%indices),1]))
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
    integer(int64), allocatable :: codes(:,:)
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
    call integers(codes)
  end subroutine
  subroutine reject(name)
    character(*), intent(in) :: name
    type(polynomial_features) :: t
    type(csr_matrix) :: s,r
    type(feature_string) :: names(2)
    type(feature_string), allocatable :: output_names(:)
    real(dp) :: data(3,2),bad
    real(dp), allocatable :: a(:,:)
    integer(int64), allocatable :: powers(:,:)
    data=1;bad=ieee_value(0._dp,ieee_quiet_nan);names(1)%text='a';names(2)%text='b';s=csr_from_dense(data)
    select case(name)
    case('negative_degree');t%degree=-1
    case('negative_min');t%min_degree=-1
    case('degree_order');t%min_degree=3;t%degree=2
    case('zero_no_bias');t%degree=0;t%include_bias=.false.
    case('empty_rows');call t%fit(data(:0,:));return
    case('empty_features');call t%fit(data(:,:0));return
    case('nan');data(1,1)=bad
    case('inf');data(1,1)=ieee_value(0._dp,ieee_positive_inf)
    case('sparse_nan');s%data(1)=bad;call t%fit(s);return
    case('csr_index');s%indices(1)=0;call t%fit(s);return
    case('csr_pointer');s%indptr(1)=0;call t%fit(s);return
    case('unfitted');a=t%transform(data);return
    case('unfitted_powers');powers=t%get_powers();return
    case('unfitted_names');output_names=t%get_feature_names_out();return
    case('query_width');call t%fit(data);a=t%transform(data(:,:1));return
    case('sparse_width');call t%fit(data);s=csr_from_dense(data(:,:1));r=t%transform(s);return
    case('changed_degree');call t%fit(data);t%degree=3;a=t%transform(data);return
    case('changed_bias');call t%fit(data);t%include_bias=.false.;powers=t%get_powers();return
    case('changed_interaction');call t%fit(data);t%interaction_only=.true.;output_names=t%get_feature_names_out();return
    case('fit_name_count');call t%fit(data,names(:1));return
    case('fit_name_unallocated');deallocate(names(1)%text);call t%fit(data,names);return
    case('name_count');call t%fit(data);output_names=t%get_feature_names_out(names(:1));return
    case('name_text');call t%fit(data,names);names(1)%text='c';output_names=t%get_feature_names_out(names);return
    case('name_length');call t%fit(data,names);names(1)%text='aa';output_names=t%get_feature_names_out(names);return
    case('name_unallocated');call t%fit(data);deallocate(names(1)%text);output_names=t%get_feature_names_out(names);return
    case('count_overflow')
      s=csr_from_arrays(1,100000,[1,1],[integer::],[real(dp)::]);t%degree=10;call t%fit(s);return
    case('csr_output_range')
      s=csr_from_arrays(1,100000,[1,1],[integer::],[real(dp)::]);call t%fit(s);r=t%transform(s);return
    case('dense_address_range')
      t%degree=huge(1_int64)-1;call t%fit(data(:1,:1));a=t%transform(data(:1,:1));return
    case('powers_address_range')
      t%degree=huge(1_int64)-1;call t%fit(data(:1,:1));powers=t%get_powers();return
    case('names_address_range')
      t%degree=huge(1_int64)-1;call t%fit(data(:1,:1));output_names=t%get_feature_names_out();return
    case('canonical_overflow')
      s=csr_from_arrays(1,1,[1,3],[1,1],[1.e308_dp,1.e308_dp]);call t%fit(s);r=t%transform(s);return
    case default;error stop 'unknown polynomial rejection'
    end select
    call t%fit(data)
  end subroutine
end program
