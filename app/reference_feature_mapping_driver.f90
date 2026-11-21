program reference_feature_mapping_driver
  use sklearn_feature_extraction
  use sklearn_murmurhash, only: murmurhash3_32
  use sklearn_sparse, only: csr_matrix
  use sklearn_common, only: dp
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  type(dict_vectorizer) :: vectorizer
  type(feature_hasher) :: hasher
  type(feature_row), allocatable :: train(:),query(:),inverse(:)
  type(feature_string), allocatable :: names(:)
  type(csr_matrix) :: a,b
  real(dp), allocatable :: dense(:,:)
  integer, allocatable :: support(:)
  integer :: kind,n,m,sorting,operation,nfeatures,alternate,restriction,ns,i
  integer(int64) :: seed
  character(:), allocatable :: separator,key
  character(8) :: mode
  read(*,*) kind,n,m,sorting,operation,nfeatures,alternate,restriction,seed
  if(kind==2) then
    key=read_text();print *,murmurhash3_32(key,seed);stop
  end if
  separator=read_text();read(*,*) mode
  call read_rows(n,train);call read_rows(m,query)
  if(kind==0) then
    vectorizer%sort=sorting==1;vectorizer%separator=separator
    if(operation==0) then
      a=vectorizer%fit_transform(train)
    else
      call vectorizer%fit(train)
      if(n>0) then
        a=vectorizer%transform(train)
      else
        a%n_rows=0;a%n_cols=vectorizer%n_features_out_
        a%indptr=[1];allocate(a%indices(0),a%data(0))
      end if
    end if
    names=vectorizer%get_feature_names_out();call print_names(names);call print_csr(a)
    if(a%n_cols>0.and.a%n_rows>0.and.all(ieee_is_finite(a%data))) then
      inverse=vectorizer%inverse_transform(a);call print_inverse(inverse)
      dense=a%to_dense()
      if(all(ieee_is_finite(dense))) then
        inverse=vectorizer%inverse_transform(dense);call print_inverse(inverse)
      else
        print *,-1
      end if
    else
      print *,-1;print *,-1
    end if
    if(restriction>0) then
      read(*,*) ns;allocate(support(ns));read(*,*) support
      if(restriction==1) then
        call vectorizer%restrict(support/=0)
      else
        call vectorizer%restrict(support)
      end if
    end if
    if(operation==2) vectorizer%separator='::'
    names=vectorizer%get_feature_names_out();call print_names(names)
    b=vectorizer%transform(query);call print_csr(b)
    print *, (vectorizer%vocabulary_index(names(i)%text),i=1,size(names))
  else
    hasher%input_type=mode;hasher%n_features=nfeatures;hasher%alternate_sign=alternate==1
    if(operation==0) then
      a=hasher%fit_transform(train)
    else
      call hasher%fit();a=hasher%transform(train)
    end if
    print *,0;call print_csr(a);print *,-1;print *,-1;print *,0
    b=hasher%transform(query);call print_csr(b);print *
  end if
contains
  function read_text() result(text)
    character(:), allocatable :: text,hex
    integer :: length,j,byte
    read(*,*) length
    allocate(character(length)::text);allocate(character(2*length)::hex)
    read(*,'(a)') hex
    do j=1,length
      read(hex(2*j-1:2*j),'(z2)') byte;text(j:j)=achar(byte)
    end do
  end function
  subroutine print_text(text)
    character(*), intent(in) :: text
    character(:), allocatable :: hex
    integer :: j
    allocate(character(2*len(text))::hex)
    do j=1,len(text)
      write(hex(2*j-1:2*j),'(z2.2)') iachar(text(j:j))
    end do
    print *,len(text);print '(a)',hex
  end subroutine
  subroutine read_rows(count,rows)
    integer, intent(in) :: count
    type(feature_row), allocatable, intent(out) :: rows(:)
    integer :: r,j,k,ni,nt,tag,nc
    allocate(rows(count))
    do r=1,count
      read(*,*) ni,nt
      if(ni>=0) then
        allocate(rows(r)%items(ni))
        do j=1,ni
          rows(r)%items(j)%key=read_text();read(*,*) tag
          select case(tag)
          case(0);read(*,*) rows(r)%items(j)%value
          case(1);rows(r)%items(j)%category=read_text()
          case(2)
            read(*,*) nc;allocate(rows(r)%items(j)%categories(nc))
            do k=1,nc
              rows(r)%items(j)%categories(k)%text=read_text()
            end do
          case(3);rows(r)%items(j)%is_none=.true.
          end select
        end do
      end if
      if(nt>=0) then
        allocate(rows(r)%tokens(nt))
        do j=1,nt
          rows(r)%tokens(j)%text=read_text()
        end do
      end if
    end do
  end subroutine
  subroutine print_names(names)
    type(feature_string), intent(in) :: names(:)
    integer :: j
    print *,size(names)
    do j=1,size(names)
      call print_text(names(j)%text)
    end do
  end subroutine
  subroutine print_csr(matrix)
    type(csr_matrix), intent(in) :: matrix
    real(dp), allocatable :: values(:,:)
    print *,matrix%n_rows,matrix%n_cols,matrix%nnz(),merge(1,0,matrix%n_cols<=1024)
    print *,matrix%indptr
    print *,matrix%indices
    print '(*(es25.16e3,1x))',matrix%data
    if(matrix%n_cols<=1024) then
      values=matrix%to_dense();print '(*(es25.16e3,1x))',values
    else
      print *
    end if
  end subroutine
  subroutine print_inverse(rows)
    type(feature_row), intent(in) :: rows(:)
    integer :: r,j
    print *,size(rows)
    do r=1,size(rows)
      print *,size(rows(r)%items)
      do j=1,size(rows(r)%items)
        call print_text(rows(r)%items(j)%key)
        print '(*(es25.16e3,1x))',rows(r)%items(j)%value
      end do
    end do
  end subroutine
end program
